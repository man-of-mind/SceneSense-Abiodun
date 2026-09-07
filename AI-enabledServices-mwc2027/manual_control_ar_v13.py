#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Allows controlling a vehicle with a keyboard.

Welcome to CARLA manual control.

Version 13 adds cooperative, display-only occlusion reasoning to the
ego-following spatial map. Selected virtual camera/radar pairs draw their
horizontal fields of view, and cached Town10 building footprints plus current
vehicle footprints cast 2-D ground-plane visibility shadows. The highlighted
remaining occlusion is the ego blind region minus coverage visible to active
peer sites, so activating a useful complementary site reduces that region.
This is an idealized geometry demonstrator fed by CARLA ground-truth poses; it
does not claim measured split-inference accuracy and it never starts a virtual
map sensor. ``[``/``-`` and ``]``/``=`` decrease/increase the live site budget.
The camera and radar horizontal FoVs can be configured separately, or one
pair-wide command-line override can assign the same FoV to both modalities.

A virtual pair is inventoried for every nearby live vehicle/walker actor plus
the configured traffic-light roots; only the operator-selected subset is
active. A spawn_blocker rogue pedestrian is withheld from both the spatial map
and AR overlays while it remains unresolved. Visibility is tested over the
pedestrian's ground footprint rather than only its centre: any in-FoV footprint
probe with a clear line of sight is a detection. Either the active ego pair or
an active peer pair can authorize the existing red box, PEDESTRIAN label, and
proximity warning. Peer-only recovery is labelled OCCLUDED PEDESTRIAN on the
spatial map. A target's own mounted virtual pair cannot detect itself. In a
network-degraded zone, the same reasoning consumes only the radar markers that
the existing radar-priority policy marks active.

Version 12 extends the ego-following spatial map with display-only, virtual RGB
camera/radar sites. No CARLA camera or radar actor is created or discovered for
these map nodes. A co-located virtual pair follows the ego vehicle, every
strictly role-labelled spawn_blocker_v5 static/reactive vehicle and pedestrian,
and each configured infrastructure traffic-light root (with catalog fallback).
The map chooses a deterministic, hysteretic nearby subset of those sites.
Active cameras use cyan directional triangles, active radars use orange
diamonds, and the configured infrastructure inventory defaults to traffic
lights 14, 24, and 11. The ego-view RGB camera, collision/lane/GNSS/IMU
sensors, and the explicitly toggled G-key radar remain real because they serve
the driving client rather than the spatial-map visualization.

Version 12 also adds configurable cellular-network degradation zones.  Their
translucent map layer is drawn below routes, actors, sensors, and occlusion
labels.  Inside a zone the metrics window applies clearly labelled,
display-only penalties to map latency, map accuracy error, and sense-to-act
latency.  Inside a degraded zone every camera marker is shown inactive and only
the selected nearby radar markers are shown active. A committed world profile
published by spawn_blocker_v5.py supplies vehicle/pedestrian zones dynamically;
its legacy stream opt-in flag is ignored. Repeat ``--network-degradation-zone
X Y RADIUS`` to override only the published/default zone locations locally, or
use ``--disable-network-degradation`` to force no zones. This passive client
never changes the clock, world settings, or Traffic Manager. Spatial-map
sensor markers cannot collect data because the represented sites are virtual
and carry no CARLA actor proxy.

To bound spatial-map work, virtual sensor sites are restricted to a
configurable forward region of interest. The default region is a 40 m, +/-45
degree planar cone ahead of the ego vehicle. Side, rear, and distant virtual
sites are neither drawn nor eligible for visual activation.

Version 11 highlights the occluded hazard actors owned by spawn_blocker_v5.py
inside the ego-following spatial map.  Reactive vehicles and blocker
pedestrians use red markers with adjacent OCCLUDED VEHICLE or OCCLUDED
PEDESTRIAN labels, and blocker-pedestrian circles are enlarged for rapid map
comprehension.  Static occluder vehicles and ordinary traffic retain their
existing colors.  Actor roles are evaluated on every map refresh, so replaced
blocker generations remain highlighted without caching transient actor IDs.

Version 10 adds a proximity-aware AR warning for the occluded blocker
pedestrians spawned by spawn_blocker_v5.py.  While U visualizations are on, a
blocker pedestrian inside the configured warning radius is outlined in red and
labelled PEDESTRIAN.  A top-right warning card says SLOW DOWN, changing to
APPLY BRAKES inside the independently configurable close-range radius.  The
alert clears after the ego passes the pedestrian.  Version 9's route arrows,
actor boxes, top-down map, and live metrics remain grouped under U.

Version 9 adds a compact live Physical AI metrics window to Version 8.  The
U key now toggles that window together with the route arrows, actor boxes, and
ego-following top-down map.  Metrics backed by this client are labelled LIVE;
spatial-map accuracy and AI-reasoning latency are labelled DEMO because this
manual client has no spatial-map estimator or reasoning service to measure.

Version 8 loads coordinate-based ego routes exported by
physical_ai_scenario_controller_ui_v2.py.  A loaded route is visual guidance
for the human driver: it does not silently enable autopilot.  Perspective
arrows are projected onto the road in the RGB stream and the U key toggles the
route guidance together with the actor boxes and ego-following top-down map.
Version 7's exact ego-vehicle blueprint selection, hidden startup HUD, and
front-mounted initial RGB camera are preserved.

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
    Y            : respawn ego at the configured CLI or saved-route start

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry
    U            : toggle route arrows, actor boxes, rogue-pedestrian warnings,
                   top-down map, sensor nodes/network zone, and metrics
    [ or -       : decrease active cooperative virtual sensor sites
    ] or =       : increase active cooperative virtual sensor sites
    SHIFT + key  : change the cooperative site budget in steps of five

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
import json
import logging
import math
import random
import re
import os
import sys
import time
import weakref

from ego_route_config import load_route_config, maps_match
from cooperative_occlusion_v1 import (
    evaluate_cooperative_targets,
    modality_limits,
    occlusion_shadow_polygon_xy,
    sensor_fov_polygon_xy,
)

try:
    from network_degradation_profile_v1 import (
        NetworkProfileError,
        discover_network_degradation_profile,
    )
except ImportError:
    # Keep the client usable when copied without the optional shared-profile
    # helper. The built-in zone remains available and stream activation stays
    # fail-safe off.
    NetworkProfileError = ValueError
    discover_network_degradation_profile = None

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
    from pygame.locals import K_LEFTBRACKET
    from pygame.locals import K_PAGEDOWN
    from pygame.locals import K_PAGEUP
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_RIGHTBRACKET
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

# The default zone overlaps spawn_blocker_v5 pedestrian #2 and the first leg
# of the bundled Town10 ego route.  Influence follows a deterministic radial
# smoothstep: zero at the boundary and one at the centre.
DEFAULT_NETWORK_DEGRADATION_ZONE = (
    8.827261924743652,
    62.21647644042969,
    18.0,
)
MAX_NETWORK_DEGRADATION_ZONE_RADIUS_M = 10000.0
DEFAULT_NETWORK_MAP_LATENCY_PENALTY_MS = 12.0
DEFAULT_NETWORK_MAP_ACCURACY_PENALTY_CM = 1.2
DEFAULT_NETWORK_SENSE_TO_ACT_PENALTY_MS = 18.0
NETWORK_DEGRADATION_OVERLAY_ALPHA = 0.13

# The infrastructure inventory is represented virtually at these Town10
# traffic-light roots. Live root transforms override the catalog fallback.
DEFAULT_INFRASTRUCTURE_SENSOR_TRAFFIC_LIGHT_IDS = (14, 24, 11)
TRAFFIC_LIGHT_SENSOR_CATALOG_FILENAME = 'traffic_lights_data.json'
VIRTUAL_TRAFFIC_LIGHT_SENSOR_HEIGHT_M = 15.0
VIRTUAL_SENSOR_ID_NAMESPACE_STRIDE = 1000000000000
VIRTUAL_SENSOR_DYNAMIC_HOST_NAMESPACE = 1
VIRTUAL_SENSOR_TRAFFIC_LIGHT_NAMESPACE = 2
VIRTUAL_SENSOR_FRONT_MARGIN_M = 0.05

DEFAULT_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS = 5
DEFAULT_SPATIAL_MAP_DEGRADED_RADAR_PAIRS = 5
MAX_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS = 1000
DEFAULT_SPATIAL_MAP_SENSOR_FORWARD_RANGE_M = 40.0
DEFAULT_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG = 45.0
MAX_SPATIAL_MAP_SENSOR_FORWARD_RANGE_M = 10000.0
MAX_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG = 90.0
DEFAULT_COOPERATIVE_CAMERA_HORIZONTAL_FOV_DEG = 90.0
DEFAULT_COOPERATIVE_CAMERA_RANGE_M = 60.0
DEFAULT_COOPERATIVE_RADAR_HORIZONTAL_FOV_DEG = 35.0
DEFAULT_COOPERATIVE_RADAR_RANGE_M = 80.0
MAX_COOPERATIVE_SENSOR_RANGE_M = 10000.0
MAX_COOPERATIVE_SENSOR_FOV_DEG = 180.0
COOPERATIVE_VISIBILITY_MAX_AGE_SECONDS = 0.75
COOPERATIVE_SENSOR_COUNT_FAST_STEP = 5
COOPERATIVE_OCCLUSION_MASK_DOWNSAMPLE = 4
SENSOR_FORWARD_REGION_EPSILON_M = 1.0e-6
SPATIAL_SENSOR_SELECTION_HYSTERESIS_M = 2.0
NETWORK_RADAR_PRIORITY_EXIT_REFRESHES = 3
NETWORK_PROFILE_REFRESH_SECONDS = 0.25
NETWORK_PROFILE_MISSING_HYSTERESIS_REFRESHES = 2

# The live metrics strip is deliberately light enough to share the main render
# loop with the existing top-down OpenCV window.  The two DEMO-only fields use
# a dedicated, seeded RNG so they never perturb CARLA blueprint/spawn choices.
LIVE_METRICS_REFRESH_HZ = 5.0
LIVE_METRICS_WINDOW_WIDTH = 1200
LIVE_METRICS_WINDOW_HEIGHT = 184
DEFAULT_METRICS_PLACEHOLDER_SEED = 20260817
METRICS_PLACEHOLDER_REFRESH_SECONDS = 0.5
METRICS_PLACEHOLDER_EWMA_ALPHA = 0.25
METRICS_MEASURED_EWMA_ALPHA = 0.25
METRICS_MAX_SENSOR_AGE_SECONDS = 2.0
METRICS_MAX_MAP_SAMPLE_AGE_SECONDS = 2.0
METRICS_ACCURACY_MIN_CM = 1.0
METRICS_ACCURACY_MAX_CM = 4.0
METRICS_ACCURACY_MEAN_CM = 2.3
METRICS_ACCURACY_SIGMA_CM = 0.45
METRICS_REASONING_MIN_MS = 30.0
METRICS_REASONING_MAX_MS = 60.0
METRICS_REASONING_MEAN_MS = 45.0
METRICS_REASONING_SIGMA_MS = 5.0

# spawn_blocker_v5.py assigns every blocker walker a stable role of the form
# pedestrian_blocker_v5_<positive index>.  Actor IDs are deliberately not used
# because the blocker lifecycle replaces an actor on every respawn.
DEFAULT_ROGUE_PEDESTRIAN_ROLE_PREFIX = 'pedestrian_blocker_v5'
DEFAULT_ROGUE_VEHICLE_ROLE_PREFIX = 'reactive_blocker_v5'
DEFAULT_STATIC_BLOCKER_VEHICLE_ROLE_PREFIX = 'static_blocker_v5'
DEFAULT_ROGUE_PEDESTRIAN_WARNING_RADIUS_M = 45.0
DEFAULT_ROGUE_PEDESTRIAN_BRAKE_RADIUS_M = 12.0
ROGUE_PEDESTRIAN_PASS_MARGIN_M = 1.0
ROGUE_PEDESTRIAN_PASS_EXTRA_CLEARANCE_M = 0.25
ROGUE_PEDESTRIAN_REVERSE_SPEED_THRESHOLD_MPS = 0.25
ROGUE_PEDESTRIAN_DISTANCE_EPSILON_M = 1.0e-4
ROGUE_PEDESTRIAN_BOX_COLOR = (255, 35, 35)
ROGUE_PEDESTRIAN_WARNING_COLOR = (255, 48, 48)
ROGUE_PEDESTRIAN_CAUTION_COLOR = (255, 190, 55)
ROGUE_PEDESTRIAN_LABEL = 'PEDESTRIAN'
ROGUE_PEDESTRIAN_WARNING_TITLE = 'APPROACHING PEDESTRIAN'
ROGUE_PEDESTRIAN_SLOW_ACTION = 'SLOW DOWN'
ROGUE_PEDESTRIAN_BRAKE_ACTION = 'APPLY BRAKES'
ROGUE_WARNING_CARD_WIDTH = 460
ROGUE_WARNING_CARD_HEIGHT = 116

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

# Camera-space guidance is sampled in world space so every arrow shrinks and
# turns naturally under perspective, like paint attached to the road plane.
ROUTE_GUIDANCE_LOOKAHEAD_M = 65.0
ROUTE_ARROW_SPACING_M = 9.0
ROUTE_ARROW_START_M = 5.0
ROUTE_ARROW_LENGTH_M = 3.2
ROUTE_ARROW_WIDTH_M = 1.35
ROUTE_CONFIG_ENDPOINT_TOLERANCE_M = 12.0
ROUTE_CONFIG_CONTROL_TOLERANCE_M = 5.0

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
TOPDOWN_COLOR_OCCLUDED_ACTOR = (45, 45, 245)
TOPDOWN_COLOR_NETWORK_ZONE_FILL = (142, 72, 132)
TOPDOWN_COLOR_NETWORK_ZONE_EDGE = (224, 118, 222)
TOPDOWN_COLOR_CAMERA_ACTIVE = (235, 224, 45)
TOPDOWN_COLOR_CAMERA_INACTIVE = (104, 101, 91)
TOPDOWN_COLOR_RADAR_ACTIVE = (42, 181, 255)
TOPDOWN_COLOR_RADAR_PRIORITY = (42, 245, 255)
TOPDOWN_COLOR_RADAR_INACTIVE = (92, 101, 112)
TOPDOWN_COLOR_SENSOR_OUTLINE = (238, 242, 247)
TOPDOWN_COLOR_LABEL_TEXT = (245, 245, 245)
TOPDOWN_COLOR_LABEL_BACKGROUND = (18, 23, 30)
TOPDOWN_COLOR_CAMERA_FOV = (188, 168, 35)
TOPDOWN_COLOR_RADAR_FOV = (38, 139, 225)
TOPDOWN_COLOR_REMAINING_OCCLUSION = (108, 48, 188)
TOPDOWN_CAMERA_FOV_ALPHA = 0.08
TOPDOWN_RADAR_FOV_ALPHA = 0.07
TOPDOWN_REMAINING_OCCLUSION_ALPHA = 0.30
TOPDOWN_OCCLUDED_PEDESTRIAN_OUTER_RADIUS_PX = 9
TOPDOWN_OCCLUDED_PEDESTRIAN_INNER_RADIUS_PX = 7
TOPDOWN_OCCLUDED_LABEL_FONT_SCALE = 0.42
TOPDOWN_OCCLUDED_LABEL_PADDING_PX = 4
TOPDOWN_OCCLUDED_LABEL_GAP_PX = 10
TOPDOWN_OCCLUDED_VEHICLE_LABEL = 'OCCLUDED VEHICLE'
TOPDOWN_OCCLUDED_PEDESTRIAN_LABEL = 'OCCLUDED PEDESTRIAN'

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


def positive_finite_float(value):
    """Argparse converter for positive, finite distance values."""
    parsed = finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return parsed


def nonnegative_finite_float(value):
    """Argparse converter for a finite value that may be zero."""
    parsed = finite_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError('must be zero or greater')
    return parsed


def nonnegative_int(value):
    """Argparse converter for a base-10 integer that may be zero."""
    try:
        parsed = int(str(value), 10)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('must be an integer')
    if parsed < 0:
        raise argparse.ArgumentTypeError('must be zero or greater')
    return parsed


def resolve_cooperative_sensor_fovs(
        pair_horizontal_fov,
        camera_horizontal_fov,
        radar_horizontal_fov):
    """Resolve pair-wide versus modality-specific horizontal FoV options."""
    if (
            pair_horizontal_fov is not None
            and (camera_horizontal_fov is not None
                 or radar_horizontal_fov is not None)):
        raise ValueError(
            '--cooperative-sensor-pair-horizontal-fov cannot be combined '
            'with --cooperative-camera-horizontal-fov or '
            '--cooperative-radar-horizontal-fov')
    if pair_horizontal_fov is not None:
        camera_horizontal_fov = pair_horizontal_fov
        radar_horizontal_fov = pair_horizontal_fov
    if camera_horizontal_fov is None:
        camera_horizontal_fov = DEFAULT_COOPERATIVE_CAMERA_HORIZONTAL_FOV_DEG
    if radar_horizontal_fov is None:
        radar_horizontal_fov = DEFAULT_COOPERATIVE_RADAR_HORIZONTAL_FOV_DEG
    resolved = (
        float(camera_horizontal_fov),
        float(radar_horizontal_fov),
    )
    if not all(math.isfinite(value) and 0.0 < value <=
               MAX_COOPERATIVE_SENSOR_FOV_DEG for value in resolved):
        raise ValueError(
            'cooperative sensor horizontal FoV must be greater than zero '
            'and at most {:.1f} degrees'.format(
                MAX_COOPERATIVE_SENSOR_FOV_DEG))
    return resolved


def infrastructure_sensor_traffic_light_ids(value):
    """Parse a unique comma-separated list of positive traffic-light IDs."""
    if isinstance(value, (tuple, list)):
        raw_values = list(value)
    else:
        text = str(value).strip()
        if text.lower() in ('none', 'off', 'disabled'):
            return ()
        raw_values = text.split(',')
    identifiers = []
    for raw_value in raw_values:
        try:
            identifier = int(str(raw_value).strip())
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                'must be comma-separated positive integer actor IDs or "none"')
        if identifier <= 0:
            raise argparse.ArgumentTypeError(
                'traffic-light actor IDs must be positive integers')
        if identifier not in identifiers:
            identifiers.append(identifier)
    if not identifiers:
        raise argparse.ArgumentTypeError(
            'must contain an actor ID or use "none"')
    return tuple(identifiers)


def normalize_network_degradation_zones(zones):
    """Validate and normalize ``(x, y, radius)`` network zones."""
    normalized = []
    for zone in zones or ():
        if len(zone) != 3:
            raise ValueError('each network degradation zone needs X Y RADIUS')
        x_coord, y_coord, radius = [float(value) for value in zone]
        if not all(math.isfinite(value) for value in (x_coord, y_coord, radius)):
            raise ValueError('network degradation zone values must be finite')
        if radius <= 0.0 or radius > MAX_NETWORK_DEGRADATION_ZONE_RADIUS_M:
            raise ValueError(
                'network degradation zone radius must be greater than zero '
                'and at most {:.1f} m'.format(
                    MAX_NETWORK_DEGRADATION_ZONE_RADIUS_M))
        normalized.append((x_coord, y_coord, radius))
    return tuple(normalized)


def network_degradation_strength(location, zones):
    """Return the strongest smooth radial influence from zero through one."""
    try:
        x_coord = float(location.x)
        y_coord = float(location.y)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if not math.isfinite(x_coord) or not math.isfinite(y_coord):
        return 0.0
    strongest = 0.0
    for zone_x, zone_y, radius in zones or ():
        distance = math.hypot(x_coord - zone_x, y_coord - zone_y)
        if distance >= radius:
            continue
        inward_fraction = max(0.0, min(1.0, 1.0 - (distance / radius)))
        # Smoothstep removes a visible/metric discontinuity at the boundary.
        influence = inward_fraction * inward_fraction * (
            3.0 - (2.0 * inward_fraction))
        strongest = max(strongest, influence)
    return strongest


def point_is_in_forward_sensor_region(
        origin_x,
        origin_y,
        forward_x,
        forward_y,
        point_x,
        point_y,
        maximum_distance_m,
        half_angle_degrees):
    """Return whether an XY point lies in the bounded ego-forward cone."""
    values = (
        origin_x, origin_y, forward_x, forward_y, point_x, point_y,
        maximum_distance_m, half_angle_degrees)
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    (
        origin_x, origin_y, forward_x, forward_y, point_x, point_y,
        maximum_distance_m, half_angle_degrees,
    ) = values
    if (
            maximum_distance_m <= 0.0
            or half_angle_degrees <= 0.0
            or half_angle_degrees >
            MAX_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG):
        return False
    forward_norm = math.hypot(forward_x, forward_y)
    if forward_norm <= SENSOR_FORWARD_REGION_EPSILON_M:
        return False
    forward_x /= forward_norm
    forward_y /= forward_norm
    delta_x = point_x - origin_x
    delta_y = point_y - origin_y
    distance = math.hypot(delta_x, delta_y)
    if distance > maximum_distance_m + SENSOR_FORWARD_REGION_EPSILON_M:
        return False
    # A truly co-located point has no meaningful bearing. Keep it so an ego
    # mount at the actor origin cannot disappear through numerical jitter.
    if distance <= SENSOR_FORWARD_REGION_EPSILON_M:
        return True
    forward_projection = delta_x * forward_x + delta_y * forward_y
    minimum_projection = distance * math.cos(
        math.radians(half_angle_degrees))
    return (
        forward_projection + SENSOR_FORWARD_REGION_EPSILON_M
        >= minimum_projection)


def rogue_pedestrian_role_prefix(value):
    """Normalize the role stem used for spawn_blocker pedestrian matching."""
    prefix = str(value).strip().rstrip('_')
    if not prefix:
        raise argparse.ArgumentTypeError('must not be empty')
    if any(character.isspace() for character in prefix):
        raise argparse.ArgumentTypeError('must not contain whitespace')
    return prefix


def actor_has_indexed_role(actor, type_prefix, role_prefix):
    """Match one exact ``<role_prefix>_<positive index>`` actor role."""
    try:
        type_id = str(actor.type_id)
        role_name = str(actor.attributes.get('role_name', ''))
    except (AttributeError, RuntimeError, TypeError):
        return False
    if not type_id.startswith(str(type_prefix)):
        return False
    indexed_prefix = '{}_'.format(role_prefix)
    if not role_name.startswith(indexed_prefix):
        return False
    index_text = role_name[len(indexed_prefix):]
    return index_text.isdigit() and int(index_text) > 0


def actor_is_rogue_pedestrian(actor, role_prefix):
    """Return True only for indexed blocker walkers from the configured role."""
    return actor_has_indexed_role(
        actor,
        'walker.pedestrian.',
        role_prefix)


def actor_is_rogue_vehicle(
        actor,
        role_prefix=DEFAULT_ROGUE_VEHICLE_ROLE_PREFIX):
    """Return True only for indexed reactive vehicles from spawn_blocker_v5."""
    return actor_has_indexed_role(actor, 'vehicle.', role_prefix)


def actor_is_static_blocker_vehicle(
        actor,
        role_prefix=DEFAULT_STATIC_BLOCKER_VEHICLE_ROLE_PREFIX):
    """Return True only for indexed static vehicles from spawn_blocker_v5."""
    return actor_has_indexed_role(actor, 'vehicle.', role_prefix)


def actor_planar_bounding_radius(actor):
    """Return a conservative XY support radius without issuing an RPC."""
    try:
        bounding_box = actor.bounding_box
        center_offset = math.hypot(
            float(bounding_box.location.x),
            float(bounding_box.location.y))
        extent_radius = math.hypot(
            float(bounding_box.extent.x),
            float(bounding_box.extent.y))
        radius = center_offset + extent_radius
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0.0
    if not math.isfinite(radius) or radius < 0.0:
        return 0.0
    return radius


def rogue_pedestrian_alert_distance(
        ego_transform,
        pedestrian_transform,
        warning_radius,
        pass_margin=ROGUE_PEDESTRIAN_PASS_MARGIN_M,
        travel_forward=None):
    """Return planar alert distance, or None once outside/past the alert zone."""
    ego_location = ego_transform.location
    pedestrian_location = pedestrian_transform.location
    delta_x = float(pedestrian_location.x - ego_location.x)
    delta_y = float(pedestrian_location.y - ego_location.y)
    distance = math.hypot(delta_x, delta_y)
    if (
            not math.isfinite(distance) or
            distance > (
                float(warning_radius) +
                ROGUE_PEDESTRIAN_DISTANCE_EPSILON_M)):
        return None
    forward = (
        ego_transform.get_forward_vector()
        if travel_forward is None else travel_forward)
    longitudinal_distance = delta_x * float(forward.x) + delta_y * float(forward.y)
    if (
            not math.isfinite(longitudinal_distance) or
            longitudinal_distance < -float(pass_margin)):
        return None
    return distance


def rogue_pedestrian_warning_action(distance, brake_radius):
    """Select the exact warning action for one validated alert distance."""
    if float(distance) <= (
            float(brake_radius) + ROGUE_PEDESTRIAN_DISTANCE_EPSILON_M):
        return ROGUE_PEDESTRIAN_BRAKE_ACTION
    return ROGUE_PEDESTRIAN_SLOW_ACTION


def rogue_pedestrian_warning_is_sensor_authorized(visibility_state):
    """Allow warning styling after any active ego/peer sensor detection."""
    return str(visibility_state) in (
        'EGO_VISIBLE',
        'COOPERATIVELY_REVEALED',
    )


def rogue_pedestrian_warning_is_cooperatively_authorized(
        cooperative_state):
    """Compatibility alias for the v13 any-active-sensor warning policy."""
    return rogue_pedestrian_warning_is_sensor_authorized(cooperative_state)


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

    # Keep off-screen corner projections and clip the 12 box edges against the
    # camera near plane.  Filtering corners to the viewport first makes a box
    # collapse and disappear as a pedestrian gets very close—the exact moment
    # the APPLY BRAKES alert must remain visible.
    near_plane = 0.05
    bbox_edges = (
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    )
    clipped_points = [
        point for point in camera_corners if float(point[0]) >= near_plane
    ]
    for first_index, second_index in bbox_edges:
        first = camera_corners[first_index]
        second = camera_corners[second_index]
        first_depth = float(first[0])
        second_depth = float(second[0])
        if (first_depth >= near_plane) == (second_depth >= near_plane):
            continue
        ratio = (near_plane - first_depth) / (second_depth - first_depth)
        clipped_points.append(first + ratio * (second - first))
    if not clipped_points:
        return None

    clipped_points = np.asarray(clipped_points, dtype=np.float32)
    depths = clipped_points[:, 0]
    u_values = (
        calibration[0, 2] +
        (clipped_points[:, 1] / depths) * calibration[0, 0])
    v_values = (
        calibration[1, 2] -
        (clipped_points[:, 2] / depths) * calibration[1, 1])
    finite_pixels = np.isfinite(u_values) & np.isfinite(v_values)
    if not np.any(finite_pixels):
        return None
    u_values = u_values[finite_pixels]
    v_values = v_values[finite_pixels]

    x1 = max(0, int(math.floor(float(u_values.min()))))
    y1 = max(0, int(math.floor(float(v_values.min()))))
    x2 = min(width - 1, int(math.ceil(float(u_values.max()))))
    y2 = min(height - 1, int(math.ceil(float(v_values.max()))))
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


class LiveMetricsModel(object):
    """Collect measured client proxies and reproducible demo-only values."""

    def __init__(
            self,
            placeholder_seed=DEFAULT_METRICS_PLACEHOLDER_SEED,
            network_map_latency_penalty_ms=(
                DEFAULT_NETWORK_MAP_LATENCY_PENALTY_MS),
            network_map_accuracy_penalty_cm=(
                DEFAULT_NETWORK_MAP_ACCURACY_PENALTY_CM),
            network_sense_to_act_penalty_ms=(
                DEFAULT_NETWORK_SENSE_TO_ACT_PENALTY_MS)):
        self._rng = random.Random(int(placeholder_seed))
        self._network_map_latency_penalty_ms = float(
            network_map_latency_penalty_ms)
        self._network_map_accuracy_penalty_cm = float(
            network_map_accuracy_penalty_cm)
        self._network_sense_to_act_penalty_ms = float(
            network_sense_to_act_penalty_ms)
        for name, value in (
                ('map latency', self._network_map_latency_penalty_ms),
                ('map accuracy', self._network_map_accuracy_penalty_cm),
                ('sense-to-act', self._network_sense_to_act_penalty_ms)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    '{} degradation penalty must be finite and nonnegative'.format(
                        name))
        self._next_placeholder_update_at = 0.0
        self._spatial_accuracy_cm = METRICS_ACCURACY_MEAN_CM
        self._ai_reasoning_ms = METRICS_REASONING_MEAN_MS
        self.reset_live_measurements()

    @staticmethod
    def _ewma(previous, current, alpha):
        if previous is None:
            return float(current)
        return ((1.0 - float(alpha)) * float(previous) +
                float(alpha) * float(current))

    def _bounded_gaussian(self, mean, sigma, lower, upper):
        return min(
            float(upper),
            max(float(lower), self._rng.gauss(float(mean), float(sigma))))

    def _update_placeholders(self, now):
        if now + 1.0e-9 < self._next_placeholder_update_at:
            return
        self._next_placeholder_update_at = (
            float(now) + METRICS_PLACEHOLDER_REFRESH_SECONDS)
        accuracy_sample = self._bounded_gaussian(
            METRICS_ACCURACY_MEAN_CM,
            METRICS_ACCURACY_SIGMA_CM,
            METRICS_ACCURACY_MIN_CM,
            METRICS_ACCURACY_MAX_CM)
        reasoning_sample = self._bounded_gaussian(
            METRICS_REASONING_MEAN_MS,
            METRICS_REASONING_SIGMA_MS,
            METRICS_REASONING_MIN_MS,
            METRICS_REASONING_MAX_MS)
        self._spatial_accuracy_cm = min(
            METRICS_ACCURACY_MAX_CM,
            max(
                METRICS_ACCURACY_MIN_CM,
                self._ewma(
                    self._spatial_accuracy_cm,
                    accuracy_sample,
                    METRICS_PLACEHOLDER_EWMA_ALPHA)))
        self._ai_reasoning_ms = min(
            METRICS_REASONING_MAX_MS,
            max(
                METRICS_REASONING_MIN_MS,
                self._ewma(
                    self._ai_reasoning_ms,
                    reasoning_sample,
                    METRICS_PLACEHOLDER_EWMA_ALPHA)))

    def reset_live_measurements(self):
        self.events_detected = 0
        self._events_detected_sampled_at = None
        self.spatial_map_latency_ms = None
        self._spatial_map_latency_sampled_at = None
        self.sense_to_act_latency_ms = None
        self._sense_to_act_sensed_at = None

    def note_events_detected(self, count, sampled_at=None):
        self.events_detected = max(0, int(count))
        self._events_detected_sampled_at = (
            time.perf_counter() if sampled_at is None else float(sampled_at))

    def note_spatial_map_latency(self, latency_ms, sampled_at=None):
        try:
            latency_ms = float(latency_ms)
        except (TypeError, ValueError):
            return
        if not math.isfinite(latency_ms) or latency_ms < 0.0:
            return
        self.spatial_map_latency_ms = self._ewma(
            self.spatial_map_latency_ms,
            latency_ms,
            METRICS_MEASURED_EWMA_ALPHA)
        self._spatial_map_latency_sampled_at = (
            time.perf_counter() if sampled_at is None else float(sampled_at))

    def note_control_submitted(self, sensed_at, submitted_at=None):
        if sensed_at is None:
            return
        if submitted_at is None:
            submitted_at = time.perf_counter()
        try:
            latency_seconds = float(submitted_at) - float(sensed_at)
        except (TypeError, ValueError):
            return
        if (
                not math.isfinite(latency_seconds)
                or latency_seconds < 0.0
                or latency_seconds > METRICS_MAX_SENSOR_AGE_SECONDS):
            if (
                    math.isfinite(latency_seconds)
                    and latency_seconds > METRICS_MAX_SENSOR_AGE_SECONDS):
                self.sense_to_act_latency_ms = None
                self._sense_to_act_sensed_at = None
            return
        self.sense_to_act_latency_ms = self._ewma(
            self.sense_to_act_latency_ms,
            latency_seconds * 1000.0,
            METRICS_MEASURED_EWMA_ALPHA)
        self._sense_to_act_sensed_at = float(sensed_at)

    def snapshot(
            self,
            collision_count=0,
            now=None,
            network_degradation_strength_value=0.0,
            network_sensor_policy=None):
        if now is None:
            now = time.perf_counter()
        self._update_placeholders(float(now))
        try:
            degradation_strength = float(
                network_degradation_strength_value)
        except (TypeError, ValueError):
            degradation_strength = 0.0
        if not math.isfinite(degradation_strength):
            degradation_strength = 0.0
        degradation_strength = max(0.0, min(1.0, degradation_strength))
        if network_sensor_policy not in ('BALANCED', 'RADAR PRIORITY'):
            network_sensor_policy = (
                'RADAR PRIORITY'
                if degradation_strength > 0.0 else
                'BALANCED')
        collision_count = max(0, int(collision_count))
        events_detected = int(self.events_detected)
        if (
                self._events_detected_sampled_at is None
                or float(now) - self._events_detected_sampled_at
                > METRICS_MAX_SENSOR_AGE_SECONDS):
            events_detected = None
        spatial_map_latency_ms = self.spatial_map_latency_ms
        if (
                self._spatial_map_latency_sampled_at is None
                or float(now) - self._spatial_map_latency_sampled_at
                > METRICS_MAX_MAP_SAMPLE_AGE_SECONDS):
            spatial_map_latency_ms = None
        sense_to_act_latency_ms = self.sense_to_act_latency_ms
        if (
                self._sense_to_act_sensed_at is None
                or float(now) - self._sense_to_act_sensed_at
                > METRICS_MAX_SENSOR_AGE_SECONDS):
            sense_to_act_latency_ms = None
        if spatial_map_latency_ms is not None:
            spatial_map_latency_ms += (
                self._network_map_latency_penalty_ms * degradation_strength)
        spatial_map_accuracy_cm = (
            float(self._spatial_accuracy_cm)
            + self._network_map_accuracy_penalty_cm * degradation_strength)
        if sense_to_act_latency_ms is not None:
            sense_to_act_latency_ms += (
                self._network_sense_to_act_penalty_ms * degradation_strength)
        return {
            'events_detected': events_detected,
            'spatial_map_latency_ms': spatial_map_latency_ms,
            'spatial_map_accuracy_cm': spatial_map_accuracy_cm,
            'ai_reasoning_ms': float(self._ai_reasoning_ms),
            'sense_to_act_latency_ms': sense_to_act_latency_ms,
            'outcome': (
                'HAZARD NOT AVOIDED'
                if collision_count > 0 else
                'HAZARD AVOIDED'),
            'network_degradation_strength': degradation_strength,
            'network_sensor_policy': network_sensor_policy,
        }


class LiveMetricsRenderer(object):
    """Compact six-column OpenCV window modelled on the supplied reference."""

    def __init__(
            self,
            width=LIVE_METRICS_WINDOW_WIDTH,
            height=LIVE_METRICS_WINDOW_HEIGHT,
            refresh_hz=LIVE_METRICS_REFRESH_HZ):
        self._width = int(width)
        self._height = int(height)
        self._window_name = 'CARLA Live Physical AI Metrics'
        self._window_created = False
        self._last_refresh_at = None
        self._refresh_period = 1.0 / max(0.1, float(refresh_hz))
        self.ready = cv2 is not None

    @staticmethod
    def _format_measurement(value, unit):
        if value is None:
            return '-- {}'.format(unit)
        return '{:.1f} {}'.format(float(value), unit)

    @staticmethod
    def _draw_centered_text(
            image,
            text,
            center_x,
            baseline_y,
            font_scale,
            color,
            thickness=1):
        size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness)
        origin_x = int(round(center_x - size[0] * 0.5))
        cv2.putText(
            image,
            text,
            (origin_x, int(baseline_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA)

    def build_frame(self, metrics):
        frame = np.full(
            (self._height, self._width, 3),
            (18, 15, 13),
            dtype=np.uint8)
        degradation_strength = max(
            0.0,
            min(1.0, float(metrics.get('network_degradation_strength', 0.0))))
        network_degraded = degradation_strength > 0.0
        sensor_policy = str(
            metrics.get('network_sensor_policy', 'BALANCED'))
        modeled_live_source = 'LIVE + MODEL' if network_degraded else 'LIVE LOCAL'
        modeled_demo_source = 'DEMO + MODEL' if network_degraded else 'DEMO'
        header_height = 27
        footer_height = 23
        card_top = header_height
        card_bottom = self._height - footer_height
        card_width = float(self._width) / 6.0
        columns = (
            ('SENSE', 'Events Detected', (
                '--' if metrics['events_detected'] is None
                else str(int(metrics['events_detected']))),
             (218, 58, 255), 'LIVE'),
            ('SPATIAL MAP', 'Spatial Map Latency', self._format_measurement(
                metrics['spatial_map_latency_ms'], 'ms'),
             (239, 190, 48), modeled_live_source),
            ('SPATIAL MAP', 'Spatial Map Accuracy', '{:.1f} cm*'.format(
                metrics['spatial_map_accuracy_cm']),
             (105, 220, 87), modeled_demo_source),
            ('AI REASONING', 'Reasoning Latency', '{:.1f} ms*'.format(
                metrics['ai_reasoning_ms']),
             (54, 215, 249), 'DEMO'),
            ('ACT', 'Sense-to-Act Latency', self._format_measurement(
                metrics['sense_to_act_latency_ms'], 'ms'),
             (51, 149, 255), modeled_live_source),
            ('OUTCOME', 'Collision Proxy', metrics['outcome'],
             ((88, 225, 120) if metrics['outcome'] == 'HAZARD AVOIDED'
              else (77, 88, 255)), 'LIVE PROXY'),
        )

        cv2.putText(
            frame,
            'LIVE PHYSICAL AI PIPELINE METRICS',
            (14, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 224, 230),
            1,
            cv2.LINE_AA)
        network_status = (
            'NETWORK MODEL: DEGRADED {:3.0f}% | {}'.format(
                degradation_strength * 100.0,
                sensor_policy)
            if network_degraded else
            'NETWORK MODEL: NORMAL | {} SENSORS'.format(sensor_policy))
        status_size, _ = cv2.getTextSize(
            network_status,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1)
        cv2.putText(
            frame,
            network_status,
            (self._width - status_size[0] - 14, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            ((55, 202, 255) if network_degraded else (130, 170, 140)),
            1,
            cv2.LINE_AA)
        for index, (category, label, value, color, source) in enumerate(columns):
            left = int(round(index * card_width))
            right = int(round((index + 1) * card_width))
            center_x = (left + right) * 0.5
            if index > 0:
                cv2.line(
                    frame,
                    (left, card_top + 5),
                    (left, card_bottom - 5),
                    (56, 52, 50),
                    1,
                    cv2.LINE_AA)
            cv2.rectangle(
                frame,
                (left + 9, card_top + 9),
                (left + 14, card_top + 26),
                color,
                -1)
            cv2.putText(
                frame,
                category,
                (left + 22, card_top + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA)
            self._draw_centered_text(
                frame, label, center_x, card_top + 50, 0.42,
                (182, 187, 194))
            value_scale = 0.55 if index == 5 else 0.74
            self._draw_centered_text(
                frame, value, center_x, card_top + 91, value_scale,
                color, 2)
            self._draw_centered_text(
                frame, source, center_x, card_bottom - 8, 0.32,
                (112, 117, 124))

        footer = (
            '* Accuracy/reasoning are bounded placeholders; network penalties '
            'are deterministic display models (no induced delay)'
            if network_degraded else
            '* Deterministic bounded placeholder; other values are local client proxies')
        cv2.putText(
            frame,
            footer,
            (12, self._height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (137, 142, 150),
            1,
            cv2.LINE_AA)
        return frame

    def render(self, metrics):
        if not self.ready:
            return False
        now = time.perf_counter()
        if (
                self._last_refresh_at is not None
                and now - self._last_refresh_at < self._refresh_period):
            return True
        self._last_refresh_at = now

        try:
            if self._window_created:
                visible = cv2.getWindowProperty(
                    self._window_name, cv2.WND_PROP_VISIBLE)
                if visible < 1.0:
                    self.close()
                    self.ready = False
                    return False
            else:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    self._window_name, self._width, self._height)
                self._window_created = True
            cv2.imshow(self._window_name, self.build_frame(metrics))
            cv2.waitKey(1)
        except cv2.error:
            self.close()
            self.ready = False
            return False
        return True

    def close(self):
        self._last_refresh_at = None
        if cv2 is None or not self._window_created:
            return
        try:
            cv2.destroyWindow(self._window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._window_created = False


class SelectedSensorStreamController(object):
    """No-op compatibility controller for display-only virtual nodes."""

    def reconcile(self, selected_sensor_actors, enabled):
        # Shared profile permission cannot subscribe virtual map markers.
        return None

    def stop_all(self, reason='disabled'):
        return None

    def stats_snapshot(self):
        return {}

    @property
    def owned_actor_ids(self):
        return ()

    @property
    def listening_actor_ids(self):
        return ()


class TopDownMapRenderer(object):
    """Ego-centred, world-aligned view of static geometry and live actors."""

    def __init__(
            self,
            carla_world,
            carla_map,
            zoom_radius_m,
            width=960,
            height=960,
            refresh_hz=TOPDOWN_MAP_REFRESH_HZ,
            rogue_pedestrian_role_prefix=(
                DEFAULT_ROGUE_PEDESTRIAN_ROLE_PREFIX),
            rogue_vehicle_role_prefix=DEFAULT_ROGUE_VEHICLE_ROLE_PREFIX,
            network_degradation_zones=(),
            infrastructure_sensor_traffic_light_ids=(
                DEFAULT_INFRASTRUCTURE_SENSOR_TRAFFIC_LIGHT_IDS),
            traffic_light_catalog_path=None,
            active_sensor_pairs=DEFAULT_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS,
            degraded_active_radars=(
                DEFAULT_SPATIAL_MAP_DEGRADED_RADAR_PAIRS),
            sensor_forward_range_m=(
                DEFAULT_SPATIAL_MAP_SENSOR_FORWARD_RANGE_M),
            sensor_forward_half_angle_degrees=(
                DEFAULT_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG),
            cooperative_camera_horizontal_fov_degrees=(
                DEFAULT_COOPERATIVE_CAMERA_HORIZONTAL_FOV_DEG),
            cooperative_camera_range_m=DEFAULT_COOPERATIVE_CAMERA_RANGE_M,
            cooperative_radar_horizontal_fov_degrees=(
                DEFAULT_COOPERATIVE_RADAR_HORIZONTAL_FOV_DEG),
            cooperative_radar_range_m=DEFAULT_COOPERATIVE_RADAR_RANGE_M):
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
        self.last_update_latency_ms = None
        self.last_network_degradation_strength = 0.0
        self._refresh_period_ms = max(1, int(round(1000.0 / float(refresh_hz))))
        self._header_height = 82
        self._footer_height = 58
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
        self._sensor_forward_x = 1.0
        self._sensor_forward_y = 0.0
        self._rogue_pedestrian_role_prefix = str(
            rogue_pedestrian_role_prefix)
        self._rogue_vehicle_role_prefix = str(rogue_vehicle_role_prefix)
        self._network_degradation_zones = normalize_network_degradation_zones(
            network_degradation_zones)
        self._infrastructure_sensor_traffic_light_ids = tuple(
            int(identifier)
            for identifier in infrastructure_sensor_traffic_light_ids)
        self._active_sensor_pairs = int(active_sensor_pairs)
        self._degraded_active_radars = int(degraded_active_radars)
        self._sensor_forward_range_m = float(sensor_forward_range_m)
        self._sensor_forward_half_angle_degrees = float(
            sensor_forward_half_angle_degrees)
        self._cooperative_modality_config = {
            'camera': {
                'horizontal_fov_degrees': float(
                    cooperative_camera_horizontal_fov_degrees),
                'range_m': float(cooperative_camera_range_m),
            },
            'radar': {
                'horizontal_fov_degrees': float(
                    cooperative_radar_horizontal_fov_degrees),
                'range_m': float(cooperative_radar_range_m),
            },
        }
        if not 0 <= self._active_sensor_pairs <= MAX_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS:
            raise ValueError('spatial-map active sensor-pair count is out of range')
        if not 0 <= self._degraded_active_radars <= MAX_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS:
            raise ValueError('degraded active-radar count is out of range')
        if (
                not math.isfinite(self._sensor_forward_range_m)
                or self._sensor_forward_range_m <= 0.0
                or self._sensor_forward_range_m
                > MAX_SPATIAL_MAP_SENSOR_FORWARD_RANGE_M):
            raise ValueError(
                'spatial-map sensor forward range must be greater than zero '
                'and at most {:.1f} meters'.format(
                    MAX_SPATIAL_MAP_SENSOR_FORWARD_RANGE_M))
        if (
                not math.isfinite(
                    self._sensor_forward_half_angle_degrees)
                or self._sensor_forward_half_angle_degrees <= 0.0
                or self._sensor_forward_half_angle_degrees
                > MAX_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG):
            raise ValueError(
                'spatial-map sensor forward half-angle must be greater than '
                'zero and at most {:.1f} degrees'.format(
                    MAX_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG))
        for modality, config in self._cooperative_modality_config.items():
            if (
                    not math.isfinite(config['range_m'])
                    or config['range_m'] <= 0.0
                    or config['range_m'] > MAX_COOPERATIVE_SENSOR_RANGE_M):
                raise ValueError(
                    'cooperative {} range must be greater than zero and at '
                    'most {:.1f} meters'.format(
                        modality, MAX_COOPERATIVE_SENSOR_RANGE_M))
            if (
                    not math.isfinite(config['horizontal_fov_degrees'])
                    or config['horizontal_fov_degrees'] <= 0.0
                    or config['horizontal_fov_degrees']
                    > MAX_COOPERATIVE_SENSOR_FOV_DEG):
                raise ValueError(
                    'cooperative {} horizontal field of view must be greater '
                    'than zero and at most {:.1f} degrees'.format(
                        modality, MAX_COOPERATIVE_SENSOR_FOV_DEG))
        if traffic_light_catalog_path is None:
            traffic_light_catalog_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                TRAFFIC_LIGHT_SENSOR_CATALOG_FILENAME)
        self._traffic_light_catalog = self._load_traffic_light_catalog(
            traffic_light_catalog_path)
        self._selected_live_sensor_actors = ()
        self._selected_site_keys_by_policy = {
            'balanced': (),
            'radar': (),
        }
        self._radar_priority_latched = False
        self._radar_priority_exit_refreshes = 0
        self._stream_activation_requested = False
        self._owned_stream_count = 0
        self._listening_stream_count = 0
        self._road_polylines = []
        self._building_footprints = []
        self.last_cooperative_visibility = self._empty_cooperative_snapshot()
        self.last_cooperative_sensor_summary = {
            'requested_sites': self._active_sensor_pairs,
            'available_sites': 0,
            'active_sites': 0,
            'remaining_occlusion_percent': None,
            'ego_coverage_active': False,
        }
        self.ready = cv2 is not None
        if self.ready:
            self._build_static_geometry()

    @property
    def selected_live_sensor_actors(self):
        """Compatibility property; virtual markers never expose actor proxies."""
        return tuple(self._selected_live_sensor_actors)

    @property
    def radar_priority_active(self):
        """Expose the same latched policy used for virtual marker styling."""
        return bool(self._radar_priority_latched)

    def clear_selected_sensor_actors(self):
        self._selected_live_sensor_actors = ()
        self._selected_site_keys_by_policy = {
            'balanced': (),
            'radar': (),
        }
        self.clear_cooperative_visibility()

    @staticmethod
    def _empty_cooperative_snapshot(sampled_at=0.0):
        return {
            'sampled_at': float(sampled_at),
            'actors': {},
            'cooperatively_detected_actor_ids': (),
            'ego_visible_actor_ids': (),
        }

    def clear_cooperative_visibility(self):
        """Clear warning authority whenever selection/rendering is reset."""
        self.last_cooperative_visibility = self._empty_cooperative_snapshot()
        self.last_cooperative_sensor_summary = {
            'requested_sites': self._active_sensor_pairs,
            'available_sites': 0,
            'active_sites': 0,
            'remaining_occlusion_percent': None,
            'ego_coverage_active': False,
        }

    @property
    def active_sensor_pair_limit(self):
        return int(self._active_sensor_pairs)

    def set_active_sensor_pair_limits(self, normal_pairs, degraded_radars=None):
        """Apply a live virtual-site budget without touching CARLA actors."""
        normal_pairs = max(
            0, min(MAX_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS, int(normal_pairs)))
        if degraded_radars is None:
            degraded_radars = normal_pairs
        degraded_radars = max(
            0,
            min(MAX_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS, int(degraded_radars)))
        changed = (
            normal_pairs != self._active_sensor_pairs
            or degraded_radars != self._degraded_active_radars)
        if not changed:
            return False
        self._active_sensor_pairs = normal_pairs
        self._degraded_active_radars = degraded_radars
        self._selected_site_keys_by_policy = {
            'balanced': (),
            'radar': (),
        }
        self.clear_cooperative_visibility()
        self._last_refresh_ms = None
        return True

    def set_stream_activation_requested(
            self,
            requested,
            owned_stream_count=0,
            listening_stream_count=0):
        """Update authorization, ownership, and confirmed-listening status."""
        self._stream_activation_requested = bool(requested)
        normalized_counts = []
        for count in (owned_stream_count, listening_stream_count):
            try:
                count = int(count)
            except (TypeError, ValueError, OverflowError):
                count = 0
            normalized_counts.append(max(0, count))
        self._owned_stream_count, self._listening_stream_count = (
            normalized_counts)

    def set_network_degradation_zones(self, zones):
        """Apply a newly committed shared profile without recreating the map."""
        normalized = normalize_network_degradation_zones(zones)
        if normalized == self._network_degradation_zones:
            return False
        self._network_degradation_zones = normalized
        self._radar_priority_latched = False
        self._radar_priority_exit_refreshes = 0
        return True

    @staticmethod
    def _load_traffic_light_catalog(path):
        """Load the existing Town10 traffic-light locations as a fallback."""
        if not path:
            return {}
        try:
            with open(path, 'r') as catalog_file:
                entries = json.load(catalog_file)
        except (IOError, OSError, ValueError) as exc:
            logging.warning(
                'Unable to load infrastructure sensor catalog %s: %s',
                path,
                exc)
            return {}
        catalog = {}
        if not isinstance(entries, list):
            logging.warning(
                'Infrastructure sensor catalog %s must contain a JSON list',
                path)
            return catalog
        for entry in entries:
            try:
                identifier = int(entry['id'])
                location = entry['location']
                coordinates = (
                    float(location['x']),
                    float(location['y']),
                    float(location.get('z', 0.0)),
                )
                if not all(math.isfinite(value) for value in coordinates):
                    continue
                catalog[identifier] = coordinates
            except (KeyError, TypeError, ValueError):
                continue
        return catalog

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

    def _actor_footprint_in_view(self, actor, actor_transform):
        """Return the visible-map portion test for any bounded CARLA actor."""
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

    def _draw_network_degradation_zones(self, image):
        """Draw unobtrusive zone fill and segmented boundaries below actors."""
        visible_bounds = self._visible_world_bounds()
        for zone_x, zone_y, radius in self._network_degradation_zones:
            zone_bounds = (
                zone_x - radius,
                zone_y - radius,
                zone_x + radius,
                zone_y + radius,
            )
            if not self._bounds_intersect(zone_bounds, visible_bounds):
                continue
            center = self._world_xy_to_pixel(zone_x, zone_y)
            radius_pixels = max(1, int(round(radius * self._scale)))
            overlay = image.copy()
            cv2.circle(
                overlay,
                center,
                radius_pixels,
                TOPDOWN_COLOR_NETWORK_ZONE_FILL,
                -1,
                lineType=cv2.LINE_AA)
            cv2.addWeighted(
                overlay,
                NETWORK_DEGRADATION_OVERLAY_ALPHA,
                image,
                1.0 - NETWORK_DEGRADATION_OVERLAY_ALPHA,
                0.0,
                dst=image)
            # Alternating arc segments keep the boundary visible without a
            # solid line that could be mistaken for a road or planned route.
            for start_angle in range(0, 360, 30):
                cv2.ellipse(
                    image,
                    center,
                    (radius_pixels, radius_pixels),
                    0.0,
                    float(start_angle),
                    float(start_angle + 18),
                    TOPDOWN_COLOR_NETWORK_ZONE_EDGE,
                    2,
                    lineType=cv2.LINE_AA)

    def _set_sensor_forward_axis(self, hero_transform):
        """Latch a finite normalized XY heading for this render refresh."""
        try:
            forward = hero_transform.get_forward_vector()
            forward_x = float(forward.x)
            forward_y = float(forward.y)
            forward_norm = math.hypot(forward_x, forward_y)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            forward_norm = 0.0
        if (
                not math.isfinite(forward_norm)
                or forward_norm <= SENSOR_FORWARD_REGION_EPSILON_M):
            self._sensor_forward_x = 0.0
            self._sensor_forward_y = 0.0
            return False
        self._sensor_forward_x = forward_x / forward_norm
        self._sensor_forward_y = forward_y / forward_norm
        return True

    def _sensor_xy_is_in_forward_region(self, x_coord, y_coord):
        try:
            if (
                    abs(float(x_coord) - self._center_x)
                    > self._zoom_radius_m
                    or abs(float(y_coord) - self._center_y)
                    > self._zoom_radius_m):
                return False
        except (TypeError, ValueError):
            return False
        return point_is_in_forward_sensor_region(
            self._center_x,
            self._center_y,
            self._sensor_forward_x,
            self._sensor_forward_y,
            x_coord,
            y_coord,
            self._sensor_forward_range_m,
            self._sensor_forward_half_angle_degrees)

    @staticmethod
    def _virtual_sensor_actor_id(parent_id, modality, namespace):
        """Create a stable negative display ID without claiming a CARLA ID."""
        parent_id = abs(int(parent_id))
        modality_offset = 1 if modality == 'camera' else 2
        return -(
            int(namespace) * VIRTUAL_SENSOR_ID_NAMESPACE_STRIDE
            + parent_id * 2
            + modality_offset)

    @staticmethod
    def _virtual_front_mount(actor, actor_transform):
        """Compute one shared front mount in world coordinates without spawning."""
        type_id = str(actor.type_id)
        is_pedestrian = type_id.startswith('walker.pedestrian.')
        front_x = 0.35 if is_pedestrian else 2.50
        lateral_y = 0.0
        height_z = 1.55 if is_pedestrian else 1.00
        try:
            bounding_box = actor.bounding_box
            box_location = bounding_box.location
            extent = bounding_box.extent
            values = (
                float(box_location.x),
                float(box_location.y),
                float(box_location.z),
                float(extent.x),
                float(extent.z),
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError('non-finite virtual sensor host bounds')
            front_x = (
                values[0]
                + max(0.0, values[3])
                + VIRTUAL_SENSOR_FRONT_MARGIN_M)
            lateral_y = values[1]
            upper_center_z = values[2] + 0.5 * max(0.0, values[4])
            height_z = (
                max(1.45, min(2.00, upper_center_z))
                if is_pedestrian else
                max(0.55, upper_center_z))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        rotation = rotation_matrix_from_carla_rotation(
            actor_transform.rotation)
        local_mount = np.asarray(
            (front_x, lateral_y, height_z), dtype=np.float64)
        parent_location = np.asarray((
            float(actor_transform.location.x),
            float(actor_transform.location.y),
            float(actor_transform.location.z),
        ), dtype=np.float64)
        world_mount = np.dot(rotation, local_mount) + parent_location
        values = (
            float(world_mount[0]),
            float(world_mount[1]),
            float(world_mount[2]),
            float(actor_transform.rotation.yaw),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError('non-finite virtual sensor mount')
        return values

    def _virtual_sensor_pair(
            self,
            parent_id,
            parent_type_id,
            host_category,
            mount,
            sampled_at,
            namespace=VIRTUAL_SENSOR_DYNAMIC_HOST_NAMESPACE,
            traffic_light_id=None):
        """Return a display-only co-located RGB/radar pair inside the ROI."""
        x_coord, y_coord, z_coord, yaw = (
            float(value) for value in mount)
        if not self._sensor_xy_is_in_forward_region(x_coord, y_coord):
            return []
        parent_id = int(parent_id)
        site_key = ('virtual', int(namespace), parent_id)
        markers = []
        for modality, type_id, offset in (
                ('camera', 'virtual.sensor.camera.rgb', (-8, 0)),
                ('radar', 'virtual.sensor.other.radar', (8, 0))):
            marker_id = self._virtual_sensor_actor_id(
                parent_id, modality, namespace)
            marker = {
                'actor_id': marker_id,
                'actor_ids': {marker_id},
                'sensor': None,
                'type_id': type_id,
                'role_name': 'virtual_{}_{}'.format(
                    host_category, modality),
                'parent_id': parent_id,
                'parent_type_id': str(parent_type_id),
                'modality': modality,
                'x': x_coord,
                'y': y_coord,
                'z': z_coord,
                'yaw': yaw,
                'active': False,
                'source': 'virtual',
                'selectable': True,
                'stream_eligible': False,
                'site_key': site_key,
                'host_category': str(host_category),
                'last_seen_at': float(sampled_at),
                'pixel_offset': offset,
            }
            if traffic_light_id is not None:
                marker['traffic_light_id'] = int(traffic_light_id)
            markers.append(marker)
        return markers

    def _virtual_host_sensor_pair(
            self,
            actor,
            actor_transform,
            host_category,
            sampled_at):
        try:
            mount = self._virtual_front_mount(actor, actor_transform)
            return self._virtual_sensor_pair(
                int(actor.id),
                str(actor.type_id),
                host_category,
                mount,
                sampled_at)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return []

    def _traffic_light_locations(self, traffic_lights):
        locations = {}
        for identifier in self._infrastructure_sensor_traffic_light_ids:
            catalog_location = self._traffic_light_catalog.get(identifier)
            if catalog_location is not None:
                locations[identifier] = (
                    catalog_location[0],
                    catalog_location[1],
                    catalog_location[2],
                    0.0,
                )
        for traffic_light in traffic_lights:
            try:
                identifier = int(traffic_light.id)
                if identifier not in self._infrastructure_sensor_traffic_light_ids:
                    continue
                transform = traffic_light.get_transform()
                locations[identifier] = (
                    float(transform.location.x),
                    float(transform.location.y),
                    float(transform.location.z),
                    float(transform.rotation.yaw) + 90.0,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return locations

    def _virtual_traffic_light_sensor_pairs(
            self,
            traffic_lights,
            sampled_at):
        markers = []
        expected_locations = self._traffic_light_locations(traffic_lights)
        for identifier in self._infrastructure_sensor_traffic_light_ids:
            anchor = expected_locations.get(identifier)
            if anchor is None:
                continue
            mount = (
                anchor[0],
                anchor[1],
                anchor[2] + VIRTUAL_TRAFFIC_LIGHT_SENSOR_HEIGHT_M,
                anchor[3],
            )
            markers.extend(self._virtual_sensor_pair(
                identifier,
                'traffic.traffic_light',
                'traffic_light',
                mount,
                sampled_at,
                namespace=VIRTUAL_SENSOR_TRAFFIC_LIGHT_NAMESPACE,
                traffic_light_id=identifier))
        return markers

    def _prepare_sensor_markers(
            self,
            virtual_host_markers,
            traffic_lights,
            sampled_at,
            radar_priority_active):
        markers = list(virtual_host_markers)
        markers.extend(self._virtual_traffic_light_sensor_pairs(
            traffic_lights,
            sampled_at))
        markers = [
            marker
            for marker in markers
            if self._sensor_xy_is_in_forward_region(
                marker['x'], marker['y'])
        ]
        return self._apply_visual_sensor_policy(
            markers,
            radar_priority_active)

    @staticmethod
    def _visual_sensor_site_key(marker):
        explicit_key = marker.get('site_key')
        if explicit_key is not None:
            return tuple(explicit_key)
        location_key = (
            int(round(float(marker['x']) * 10.0)),
            int(round(float(marker['y']) * 10.0)),
            int(round(float(marker.get('z', 0.0)) * 10.0)),
        )
        parent_id = marker.get('parent_id')
        if parent_id is not None:
            try:
                # A parent can host multiple mounts. Keep the parent identity
                # while requiring camera/radar actors to be co-located within
                # the same decimetre before treating them as one site.
                return ('parent-location', int(parent_id)) + location_key
            except (TypeError, ValueError):
                pass
        # Parentless sensors can still form a co-located pair. Ten-centimetre
        # quantization is tighter than the marker de-duplication radius.
        return ('location',) + location_key

    def _apply_visual_sensor_policy(self, markers, radar_priority_active):
        """Select marker colors deterministically with distance hysteresis."""
        sites = {}
        for marker in markers:
            marker['active'] = False
            if not bool(marker.get('selectable', False)):
                continue
            if (
                    abs(float(marker['x']) - self._center_x)
                    > self._zoom_radius_m
                    or abs(float(marker['y']) - self._center_y)
                    > self._zoom_radius_m):
                continue
            key = self._visual_sensor_site_key(marker)
            sites.setdefault(key, []).append(marker)

        def site_sort_key(item):
            key, site_markers = item
            minimum_distance = min(
                math.hypot(
                    float(marker['x']) - self._center_x,
                    float(marker['y']) - self._center_y)
                for marker in site_markers)
            # The ego pair is the perspective-defining site. Pin it first for
            # every positive budget even if another actor overlaps the ego
            # centre more closely than the ego's front mount.
            if any(
                    marker.get('host_category') == 'ego_vehicle'
                    for marker in site_markers):
                minimum_distance = -1.0
            minimum_actor_id = min(
                int(marker.get('actor_id', 0))
                for marker in site_markers)
            return minimum_distance, repr(key), minimum_actor_id

        ordered_sites = sorted(sites.items(), key=site_sort_key)
        for _, site_markers in ordered_sites:
            modalities = set(
                marker.get('modality') for marker in site_markers)
            if 'camera' in modalities and 'radar' in modalities:
                for marker in site_markers:
                    marker['pixel_offset'] = (
                        (-8, 0)
                        if marker['modality'] == 'camera'
                        else (8, 0))

        if radar_priority_active:
            radar_sites = [
                site
                for site in ordered_sites
                if any(
                    marker.get('modality') == 'radar'
                    for marker in site[1])]
            selected_sites = self._select_sensor_sites_with_hysteresis(
                radar_sites,
                self._degraded_active_radars,
                'radar',
                site_sort_key)
            for _, site_markers in selected_sites:
                radars = [marker for marker in site_markers
                          if marker.get('modality') == 'radar']
                # De-duplication normally leaves one radar per parent/site.
                radar = min(
                    radars,
                    key=lambda marker: int(marker.get('actor_id', 0)))
                radar['active'] = True
            return markers

        paired_sites = [
            site
            for site in ordered_sites
            if {'camera', 'radar'}.issubset(
                set(marker.get('modality') for marker in site[1]))]
        selected_sites = self._select_sensor_sites_with_hysteresis(
            paired_sites,
            self._active_sensor_pairs,
            'balanced',
            site_sort_key)
        for _, site_markers in selected_sites:
            for marker in site_markers:
                if marker.get('modality') in ('camera', 'radar'):
                    marker['active'] = True
        return markers

    @staticmethod
    def _stable_live_sensor_site_key(site_markers):
        """Return an actor-based identity that survives a moving host pose."""
        actor_ids = set()
        for marker in site_markers:
            values = marker.get('actor_ids')
            if values is None:
                values = (marker.get('actor_id'),)
            for actor_id in values:
                try:
                    actor_ids.add(int(actor_id))
                except (TypeError, ValueError):
                    continue
        return ('actors', tuple(sorted(actor_ids)))

    def _select_sensor_sites_with_hysteresis(
            self,
            ordered_sites,
            limit,
            policy_name,
            site_sort_key):
        """Retain selected sites until they exceed the cutoff by 2 metres."""
        limit = max(0, int(limit))
        if limit == 0 or not ordered_sites:
            self._selected_site_keys_by_policy[policy_name] = ()
            return []

        candidates = []
        for site in ordered_sites:
            stable_key = self._stable_live_sensor_site_key(site[1])
            sort_key = site_sort_key(site)
            candidates.append((stable_key, sort_key, site))
        candidates.sort(key=lambda candidate: candidate[1])
        cutoff_index = min(limit, len(candidates)) - 1
        cutoff_distance = float(candidates[cutoff_index][1][0])

        previous_keys = tuple(
            self._selected_site_keys_by_policy.get(policy_name, ()))
        previous_order = {
            stable_key: index
            for index, stable_key in enumerate(previous_keys)}
        retained = [
            candidate
            for candidate in candidates
            if (
                candidate[0] in previous_order
                and float(candidate[1][0])
                <= cutoff_distance + SPATIAL_SENSOR_SELECTION_HYSTERESIS_M)]
        retained.sort(key=lambda candidate: (
            previous_order[candidate[0]], candidate[1]))
        selected = retained[:limit]
        selected_keys = set(candidate[0] for candidate in selected)
        for candidate in candidates:
            if len(selected) >= limit:
                break
            if candidate[0] in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(candidate[0])

        self._selected_site_keys_by_policy[policy_name] = tuple(
            candidate[0] for candidate in selected)
        return [candidate[2] for candidate in selected]

    def _draw_sensor_marker(
            self,
            image,
            marker,
            radar_priority=False):
        center_x, center_y = self._world_xy_to_pixel(
            marker['x'], marker['y'])
        offset_x, offset_y = marker.get('pixel_offset', (0, 0))
        center = (center_x + int(offset_x), center_y + int(offset_y))
        active = bool(marker['active'])
        yaw_radians = math.radians(float(marker.get('yaw', 0.0)))
        forward = np.asarray(
            (math.cos(yaw_radians), math.sin(yaw_radians)),
            dtype=np.float32)
        perpendicular = np.asarray((-forward[1], forward[0]), dtype=np.float32)

        if marker['modality'] == 'camera':
            color = (
                TOPDOWN_COLOR_CAMERA_INACTIVE
                if not active else
                TOPDOWN_COLOR_CAMERA_ACTIVE)
            center_array = np.asarray(center, dtype=np.float32)
            points = np.rint(np.asarray((
                center_array + 10.0 * forward,
                center_array - 5.0 * forward + 6.0 * perpendicular,
                center_array - 5.0 * forward - 6.0 * perpendicular,
            ))).astype(np.int32)
            if active:
                cv2.fillPoly(image, [points], color, lineType=cv2.LINE_AA)
            cv2.polylines(
                image,
                [points],
                True,
                (TOPDOWN_COLOR_SENSOR_OUTLINE if active else color),
                2,
                lineType=cv2.LINE_AA)
            cv2.circle(image, center, 2, color, -1, lineType=cv2.LINE_AA)
            return

        color = (
            TOPDOWN_COLOR_RADAR_INACTIVE
            if not active else
            (TOPDOWN_COLOR_RADAR_PRIORITY
             if radar_priority else
             TOPDOWN_COLOR_RADAR_ACTIVE))
        diamond = np.asarray((
            (center[0], center[1] - 8),
            (center[0] + 8, center[1]),
            (center[0], center[1] + 8),
            (center[0] - 8, center[1]),
        ), dtype=np.int32)
        if radar_priority:
            cv2.circle(
                image, center, 15, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(
                image, center, 12, TOPDOWN_COLOR_SENSOR_OUTLINE, 1,
                lineType=cv2.LINE_AA)
        if active:
            cv2.fillPoly(image, [diamond], color, lineType=cv2.LINE_AA)
        cv2.polylines(
            image,
            [diamond],
            True,
            (TOPDOWN_COLOR_SENSOR_OUTLINE if active else color),
            2,
            lineType=cv2.LINE_AA)
        cv2.circle(image, center, 2, color, -1, lineType=cv2.LINE_AA)

    def _radar_priority_is_active(self, degradation_strength):
        """Enter immediately and leave after stable out-of-zone refreshes."""
        if float(degradation_strength) > 0.0:
            self._radar_priority_latched = True
            self._radar_priority_exit_refreshes = 0
        elif self._radar_priority_latched:
            self._radar_priority_exit_refreshes += 1
            if (
                    self._radar_priority_exit_refreshes
                    >= NETWORK_RADAR_PRIORITY_EXIT_REFRESHES):
                self._radar_priority_latched = False
                self._radar_priority_exit_refreshes = 0
        else:
            self._radar_priority_exit_refreshes = 0
        return self._radar_priority_latched

    def _draw_sensors(
            self,
            image,
            virtual_host_markers,
            traffic_lights,
            radar_priority_active,
            sampled_at,
            prepared_markers=None):
        markers = (
            list(prepared_markers)
            if prepared_markers is not None else
            self._prepare_sensor_markers(
                virtual_host_markers,
                traffic_lights,
                sampled_at,
                radar_priority_active))
        counts = {
            'camera_active': 0,
            'camera_inactive': 0,
            'radar_active': 0,
            'radar_inactive': 0,
        }
        degraded = bool(radar_priority_active)
        for marker in markers:
            if (
                    abs(float(marker['x']) - self._center_x)
                    > self._zoom_radius_m
                    or abs(float(marker['y']) - self._center_y)
                    > self._zoom_radius_m):
                continue
            active = bool(marker['active'])
            self._draw_sensor_marker(
                image,
                marker,
                radar_priority=(
                    degraded
                    and active
                    and marker['modality'] == 'radar'))
            counts['{}_{}'.format(
                marker['modality'],
                'active' if active else 'inactive')] += 1
        # Virtual markers intentionally never expose a CARLA actor proxy, so
        # the optional legacy stream controller has nothing it can subscribe.
        self._selected_live_sensor_actors = ()
        return counts

    def _draw_route(self, image, route_trace, route_path=None):
        if not route_trace and not route_path:
            return
        route_points = []
        locations = (
            list(route_path)
            if route_path else
            [waypoint.transform.location for waypoint, _ in route_trace])
        for index, location in enumerate(locations):
            if index % 3 != 0 and index != len(locations) - 1:
                continue
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

    def _draw_pedestrian(
            self,
            image,
            actor_transform,
            color,
            ego=False,
            occluded=False):
        center = self._world_to_pixel(actor_transform.location)
        outer_radius = (
            TOPDOWN_OCCLUDED_PEDESTRIAN_OUTER_RADIUS_PX
            if occluded else (5 if ego else 4))
        inner_radius = (
            TOPDOWN_OCCLUDED_PEDESTRIAN_INNER_RADIUS_PX
            if occluded else (4 if ego else 3))
        cv2.circle(image, center, outer_radius, (18, 23, 30), -1)
        cv2.circle(
            image,
            center,
            inner_radius,
            color,
            -1,
            lineType=cv2.LINE_AA)

    @staticmethod
    def _label_overlap_area(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        if right <= left or bottom <= top:
            return 0
        return (right - left) * (bottom - top)

    def _draw_occluded_label(
            self,
            image,
            anchor,
            text,
            occupied_label_bounds):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = TOPDOWN_OCCLUDED_LABEL_FONT_SCALE
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness)
        padding = TOPDOWN_OCCLUDED_LABEL_PADDING_PX
        box_width = text_width + 2 * padding
        box_height = text_height + baseline + 2 * padding
        gap = TOPDOWN_OCCLUDED_LABEL_GAP_PX
        anchor_x, anchor_y = anchor
        raw_candidates = (
            (anchor_x + gap, anchor_y - box_height - gap),
            (anchor_x + gap, anchor_y + gap),
            (anchor_x - box_width - gap, anchor_y - box_height - gap),
            (anchor_x - box_width - gap, anchor_y + gap),
        )
        maximum_x = max(0, image.shape[1] - box_width - 1)
        maximum_y = max(0, image.shape[0] - box_height - 1)
        candidates = []

        def add_candidate(left, top):
            left = max(0, min(int(left), maximum_x))
            top = max(0, min(int(top), maximum_y))
            bounds = (left, top, left + box_width, top + box_height)
            if bounds not in candidates:
                candidates.append(bounds)

        for left, top in raw_candidates:
            add_candidate(left, top)
        # Clamping can collapse all four preferred positions at a corner.
        # Add edge-aligned stacking slots across the plot so multiple nearby
        # hazards remain readable whenever non-overlapping space exists.
        x_positions = [0, maximum_x]
        y_positions = [0, maximum_y]
        x_positions.extend(range(0, maximum_x + 1, box_width + gap))
        y_positions.extend(range(0, maximum_y + 1, box_height + gap))
        for left in x_positions:
            for top in y_positions:
                add_candidate(left, top)

        def placement_score(candidate):
            overlap_area = sum(
                self._label_overlap_area(candidate, occupied)
                for occupied in occupied_label_bounds)
            left, top, right, bottom = candidate
            covers_anchor = (
                left - 2 <= anchor_x <= right + 2
                and top - 2 <= anchor_y <= bottom + 2)
            center_x = (left + right) * 0.5
            center_y = (top + bottom) * 0.5
            distance_squared = (
                (center_x - anchor_x) ** 2
                + (center_y - anchor_y) ** 2)
            return (
                overlap_area > 0,
                overlap_area,
                covers_anchor,
                distance_squared)

        bounds = min(candidates, key=placement_score)
        occupied_label_bounds.append(bounds)
        left, top, right, bottom = bounds
        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            TOPDOWN_COLOR_LABEL_BACKGROUND,
            -1,
            lineType=cv2.LINE_AA)
        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            TOPDOWN_COLOR_OCCLUDED_ACTOR,
            1,
            lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            text,
            (left + padding, top + padding + text_height),
            font,
            scale,
            TOPDOWN_COLOR_LABEL_TEXT,
            thickness,
            cv2.LINE_AA)

    def _visible_building_occluders(self):
        visible_bounds = self._visible_world_bounds()
        return [
            {
                'actor_id': None,
                'occluder_key': ('building', index),
                'polygon': points,
                'bounds': bounds,
            }
            for index, (points, bounds) in enumerate(self._building_footprints)
            if self._bounds_intersect(bounds, visible_bounds)
        ]

    def _collect_live_scene(
            self,
            carla_world,
            hero_actor,
            hero_transform,
            radar_priority_active):
        """Query actors once and derive one consistent cooperative snapshot."""
        hero_id = int(hero_actor.id)
        sampled_at = time.perf_counter()
        virtual_host_markers = []
        vehicle_records = []
        pedestrian_records = []
        targets = []
        occluders = self._visible_building_occluders()
        self._set_sensor_forward_axis(hero_transform)
        try:
            actors = carla_world.get_actors()
            vehicles = actors.filter('vehicle.*')
            pedestrians = actors.filter('walker.pedestrian.*')
            traffic_lights = actors.filter('traffic.traffic_light')
        except RuntimeError:
            vehicles = []
            pedestrians = []
            traffic_lights = []

        if str(hero_actor.type_id).startswith('vehicle.'):
            virtual_host_markers.extend(self._virtual_host_sensor_pair(
                hero_actor,
                hero_transform,
                'ego_vehicle',
                sampled_at))
            try:
                hero_footprint = get_actor_footprint_points(
                    hero_actor, hero_transform)
                occluders.append({
                    'actor_id': hero_id,
                    'occluder_key': ('actor', hero_id),
                    'polygon': hero_footprint,
                    'bounds': self._geometry_entry(hero_footprint)[1],
                })
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        for vehicle in vehicles:
            if int(vehicle.id) == hero_id:
                continue
            try:
                actor_transform = vehicle.get_transform()
                actor_id = int(vehicle.id)
                is_rogue = actor_is_rogue_vehicle(
                    vehicle, self._rogue_vehicle_role_prefix)
                is_static_blocker = actor_is_static_blocker_vehicle(vehicle)
                host_category = (
                    'reactive_blocker_vehicle'
                    if is_rogue else
                    ('static_blocker_vehicle'
                     if is_static_blocker else 'ambient_vehicle'))
                virtual_host_markers.extend(
                    self._virtual_host_sensor_pair(
                        vehicle,
                        actor_transform,
                        host_category,
                        sampled_at))
                footprint = self._actor_footprint_in_view(
                    vehicle, actor_transform)
                if footprint is None:
                    continue
                record = {
                    'actor': vehicle,
                    'actor_id': actor_id,
                    'transform': actor_transform,
                    'footprint': footprint,
                    'is_rogue': is_rogue,
                    'is_static_blocker': is_static_blocker,
                }
                vehicle_records.append(record)
                occluders.append({
                    'actor_id': actor_id,
                    'occluder_key': ('actor', actor_id),
                    'polygon': footprint,
                    'bounds': self._geometry_entry(footprint)[1],
                })
                if is_rogue:
                    targets.append({
                        'actor_id': actor_id,
                        'actor_kind': 'vehicle',
                        'x': float(actor_transform.location.x),
                        'y': float(actor_transform.location.y),
                        'footprint': footprint,
                    })
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

        for pedestrian in pedestrians:
            if int(pedestrian.id) == hero_id:
                continue
            try:
                actor_transform = pedestrian.get_transform()
                actor_id = int(pedestrian.id)
                is_rogue = actor_is_rogue_pedestrian(
                    pedestrian, self._rogue_pedestrian_role_prefix)
                virtual_host_markers.extend(
                    self._virtual_host_sensor_pair(
                        pedestrian,
                        actor_transform,
                        ('blocker_pedestrian'
                         if is_rogue else 'ambient_pedestrian'),
                        sampled_at))
                footprint = self._actor_footprint_in_view(
                    pedestrian, actor_transform)
                if footprint is None:
                    continue
                pedestrian_records.append({
                    'actor': pedestrian,
                    'actor_id': actor_id,
                    'transform': actor_transform,
                    'footprint': footprint,
                    'is_rogue': is_rogue,
                })
                # Other pedestrians are valid 2-D occluders. The evaluator
                # excludes the current target and sensor host by actor ID.
                occluders.append({
                    'actor_id': actor_id,
                    'occluder_key': ('actor', actor_id),
                    'polygon': footprint,
                    'bounds': self._geometry_entry(footprint)[1],
                })
                if is_rogue:
                    targets.append({
                        'actor_id': actor_id,
                        'actor_kind': 'pedestrian',
                        'x': float(actor_transform.location.x),
                        'y': float(actor_transform.location.y),
                        'footprint': footprint,
                    })
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

        markers = self._prepare_sensor_markers(
            virtual_host_markers,
            traffic_lights,
            sampled_at,
            radar_priority_active)
        actor_visibility = evaluate_cooperative_targets(
            markers,
            targets,
            occluders,
            hero_id,
            self._cooperative_modality_config)
        self.last_cooperative_visibility = {
            'sampled_at': sampled_at,
            'actors': actor_visibility,
            'cooperatively_detected_actor_ids': tuple(sorted(
                actor_id
                for actor_id, result in actor_visibility.items()
                if result['cooperatively_detected'])),
            'ego_visible_actor_ids': tuple(sorted(
                actor_id
                for actor_id, result in actor_visibility.items()
                if result['visible_to_ego'])),
        }
        return {
            'sampled_at': sampled_at,
            'virtual_host_markers': virtual_host_markers,
            'sensor_markers': markers,
            'traffic_lights': traffic_lights,
            'vehicles': vehicle_records,
            'pedestrians': pedestrian_records,
            'occluders': occluders,
            'visibility': actor_visibility,
        }

    @staticmethod
    def _blend_color_through_mask(image, mask, color, alpha):
        if not np.any(mask):
            return
        color_layer = np.empty_like(image)
        color_layer[:] = color
        blended = cv2.addWeighted(
            color_layer, float(alpha), image, 1.0 - float(alpha), 0.0)
        image[mask > 0] = blended[mask > 0]

    def _draw_cooperative_visibility_layer(
            self,
            image,
            sensor_markers,
            occluders,
            hero_id):
        """Draw active FoVs and the unresolved portion of the ego blind area."""
        mask_size = max(
            32,
            int(math.ceil(
                float(self._plot_size)
                / float(COOPERATIVE_OCCLUSION_MASK_DOWNSAMPLE))))
        mask_shape = (mask_size, mask_size)
        mask_pixel_scale = (
            float(mask_size - 1) / float(max(1, self._plot_size - 1)))
        ego_raw_fov = np.zeros(mask_shape, dtype=np.uint8)
        ego_visible = np.zeros(mask_shape, dtype=np.uint8)
        peer_visible = np.zeros(mask_shape, dtype=np.uint8)
        camera_fov_union = np.zeros(mask_shape, dtype=np.uint8)
        radar_fov_union = np.zeros(mask_shape, dtype=np.uint8)
        active_markers = [
            marker for marker in sensor_markers
            if bool(marker.get('active', False))]

        def to_mask_pixels(full_pixels):
            return np.rint(
                np.asarray(full_pixels, dtype=np.float32)
                * mask_pixel_scale).astype(np.int32)

        for marker in active_markers:
            limits = modality_limits(marker, self._cooperative_modality_config)
            if limits is None:
                continue
            horizontal_fov, maximum_range = limits
            sensor_xy = (float(marker['x']), float(marker['y']))
            yaw = float(marker.get('yaw', 0.0))
            fov_polygon = sensor_fov_polygon_xy(
                sensor_xy,
                yaw,
                horizontal_fov,
                maximum_range)
            fov_pixels = self._points_to_pixels(np.asarray(
                fov_polygon, dtype=np.float32))
            fov_mask_pixels = to_mask_pixels(fov_pixels)
            fov_mask = np.zeros(mask_shape, dtype=np.uint8)
            cv2.fillPoly(
                fov_mask, [fov_mask_pixels], 255, lineType=cv2.LINE_8)

            shadow_mask = np.zeros(mask_shape, dtype=np.uint8)
            parent_id = int(marker.get('parent_id', 0))
            for occluder in occluders:
                try:
                    if (
                            occluder.get('actor_id') is not None
                            and int(occluder['actor_id']) == parent_id):
                        continue
                except (TypeError, ValueError):
                    pass
                bounds = occluder.get('bounds')
                if bounds is not None:
                    delta_x = max(
                        float(bounds[0]) - sensor_xy[0],
                        0.0,
                        sensor_xy[0] - float(bounds[2]))
                    delta_y = max(
                        float(bounds[1]) - sensor_xy[1],
                        0.0,
                        sensor_xy[1] - float(bounds[3]))
                    if math.hypot(delta_x, delta_y) >= maximum_range:
                        continue
                shadow_polygon = occlusion_shadow_polygon_xy(
                    sensor_xy,
                    yaw,
                    horizontal_fov,
                    maximum_range,
                    occluder.get('polygon', ()))
                if not shadow_polygon:
                    continue
                shadow_pixels = to_mask_pixels(self._points_to_pixels(
                    np.asarray(shadow_polygon, dtype=np.float32)))
                cv2.fillPoly(
                    shadow_mask,
                    [shadow_pixels],
                    255,
                    lineType=cv2.LINE_8)
            visible_mask = cv2.bitwise_and(
                fov_mask, cv2.bitwise_not(shadow_mask))
            if parent_id == int(hero_id):
                ego_raw_fov = cv2.bitwise_or(ego_raw_fov, fov_mask)
                ego_visible = cv2.bitwise_or(ego_visible, visible_mask)
            else:
                peer_visible = cv2.bitwise_or(peer_visible, visible_mask)

            modality = marker.get('modality')
            if modality == 'camera':
                camera_fov_union = cv2.bitwise_or(
                    camera_fov_union, fov_mask)
                color = TOPDOWN_COLOR_CAMERA_FOV
            else:
                radar_fov_union = cv2.bitwise_or(
                    radar_fov_union, fov_mask)
                color = TOPDOWN_COLOR_RADAR_FOV
            cv2.polylines(
                image,
                [fov_pixels],
                True,
                color,
                1,
                lineType=cv2.LINE_AA)

        ego_blind = cv2.bitwise_and(
            ego_raw_fov, cv2.bitwise_not(ego_visible))
        remaining_occlusion = cv2.bitwise_and(
            ego_blind, cv2.bitwise_not(peer_visible))

        def expand_mask(mask):
            return cv2.resize(
                mask,
                (self._plot_size, self._plot_size),
                interpolation=cv2.INTER_NEAREST)

        self._blend_color_through_mask(
            image,
            expand_mask(camera_fov_union),
            TOPDOWN_COLOR_CAMERA_FOV,
            TOPDOWN_CAMERA_FOV_ALPHA)
        self._blend_color_through_mask(
            image,
            expand_mask(radar_fov_union),
            TOPDOWN_COLOR_RADAR_FOV,
            TOPDOWN_RADAR_FOV_ALPHA)
        remaining_occlusion_display = expand_mask(remaining_occlusion)
        self._blend_color_through_mask(
            image,
            remaining_occlusion_display,
            TOPDOWN_COLOR_REMAINING_OCCLUSION,
            TOPDOWN_REMAINING_OCCLUSION_ALPHA)
        contours_result = cv2.findContours(
            remaining_occlusion_display,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2]
        if contours:
            cv2.drawContours(
                image,
                contours,
                -1,
                TOPDOWN_COLOR_REMAINING_OCCLUSION,
                1,
                lineType=cv2.LINE_AA)

        available_sites = set(
            self._visual_sensor_site_key(marker)
            for marker in sensor_markers
            if bool(marker.get('selectable', False)))
        active_sites = set(
            self._visual_sensor_site_key(marker)
            for marker in active_markers)
        blind_pixels = int(np.count_nonzero(ego_blind))
        remaining_pixels = int(np.count_nonzero(remaining_occlusion))
        remaining_percent = (
            100.0 * float(remaining_pixels) / float(blind_pixels)
            if blind_pixels else None)
        self.last_cooperative_sensor_summary = {
            'requested_sites': (
                self._degraded_active_radars
                if self.radar_priority_active else
                self._active_sensor_pairs),
            'available_sites': len(available_sites),
            'active_sites': len(active_sites),
            'remaining_occlusion_percent': remaining_percent,
            'ego_coverage_active': bool(np.count_nonzero(ego_raw_fov)),
        }

    def _draw_live_actors(
            self,
            image,
            hero_actor,
            hero_transform,
            radar_priority_active,
            scene):
        visible_vehicle_count = 0
        visible_pedestrian_count = 0
        occluded_labels = []
        visibility = scene['visibility']

        for record in scene['vehicles']:
            is_rogue = record['is_rogue']
            result = visibility.get(record['actor_id'], {})
            cooperatively_detected = bool(
                result.get('cooperatively_detected', False))
            ego_visible = bool(result.get('visible_to_ego', False))
            network_detected = bool(result.get(
                'network_detected', cooperatively_detected or ego_visible))
            if is_rogue and not network_detected:
                continue
            self._draw_vehicle(
                image,
                record['actor'],
                record['transform'],
                (TOPDOWN_COLOR_OCCLUDED_ACTOR
                 if cooperatively_detected else TOPDOWN_COLOR_VEHICLE),
                footprint=record['footprint'])
            if cooperatively_detected:
                occluded_labels.append((
                    self._world_to_pixel(record['transform'].location),
                    TOPDOWN_OCCLUDED_VEHICLE_LABEL))
            visible_vehicle_count += 1

        for record in scene['pedestrians']:
            is_rogue = record['is_rogue']
            result = visibility.get(record['actor_id'], {})
            cooperatively_detected = bool(
                result.get('cooperatively_detected', False))
            ego_visible = bool(result.get('visible_to_ego', False))
            network_detected = bool(result.get(
                'network_detected', cooperatively_detected or ego_visible))
            if is_rogue and not network_detected:
                # Suppress CARLA-ground-truth leakage until an active sensor
                # actually supplies the spatial-map object.
                continue
            self._draw_pedestrian(
                image,
                record['transform'],
                (TOPDOWN_COLOR_OCCLUDED_ACTOR
                 if cooperatively_detected else TOPDOWN_COLOR_PEDESTRIAN),
                occluded=cooperatively_detected)
            if cooperatively_detected:
                occluded_labels.append((
                    self._world_to_pixel(record['transform'].location),
                    TOPDOWN_OCCLUDED_PEDESTRIAN_LABEL))
            visible_pedestrian_count += 1

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
        sensor_counts = self._draw_sensors(
            image,
            scene['virtual_host_markers'],
            scene['traffic_lights'],
            radar_priority_active,
            scene['sampled_at'],
            prepared_markers=scene['sensor_markers'])
        occupied_label_bounds = []
        for anchor, text in occluded_labels:
            self._draw_occluded_label(
                image, anchor, text, occupied_label_bounds)
        return visible_vehicle_count, visible_pedestrian_count, sensor_counts

    def _draw_status(
            self,
            frame,
            ego_location,
            visible_vehicle_count,
            visible_pedestrian_count,
            sensor_counts,
            degradation_strength,
            radar_priority_active):
        title = (
            'Top-down r={:.1f} m | sensor ROI {:.0f} m ahead +/-{:.0f} deg'
        ).format(
                self._zoom_radius_m,
                self._sensor_forward_range_m,
                self._sensor_forward_half_angle_degrees)
        cooperative_summary = self.last_cooperative_sensor_summary
        remaining_percent = cooperative_summary.get(
            'remaining_occlusion_percent')
        blind_area_text = (
            'N/A (NO ACTIVE EGO SENSOR)'
            if remaining_percent is None else
            '{:.1f}%'.format(float(remaining_percent)))
        coordinates = (
            'ego x={:.2f}  y={:.2f} | pairs {}/{} (requested {}) | '
            'remaining blind area {}'
        ).format(
            float(ego_location.x),
            float(ego_location.y),
            cooperative_summary.get('active_sites', 0),
            cooperative_summary.get('available_sites', 0),
            cooperative_summary.get('requested_sites', 0),
            blind_area_text)
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

        in_degraded_zone = float(degradation_strength) > 0.0
        degraded = bool(radar_priority_active)
        stream_status = 'VIRTUAL NODES | NO STREAMS'
        if in_degraded_zone:
            network_status = (
                'CELLULAR: DEGRADED {:3.0f}% | RADAR-ONLY | {} | '
                'METRICS DEGRADED'.format(
                    float(degradation_strength) * 100.0,
                    stream_status))
        elif degraded:
            network_status = (
                'CELLULAR: RECOVERING | RADAR-ONLY HYSTERESIS | {}'.format(
                    stream_status))
        else:
            network_status = (
                'CELLULAR: NORMAL | CAMERA+RADAR SUBSET | {}'.format(
                    stream_status))
        cv2.putText(
            frame,
            network_status,
            (self._plot_left, 69),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            ((55, 202, 255) if in_degraded_zone else (130, 170, 140)),
            1,
            cv2.LINE_AA)

        legend_y = self._height - 35
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
            ('COOP RECOVERED', TOPDOWN_COLOR_OCCLUDED_ACTOR),
            ('REMAINING BLIND', TOPDOWN_COLOR_REMAINING_OCCLUSION),
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
        if self._network_degradation_zones:
            label = 'CELLULAR ZONE'
            cv2.circle(
                frame,
                (x_coord + 5, legend_y - 4),
                6,
                TOPDOWN_COLOR_NETWORK_ZONE_EDGE,
                2,
                lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                label,
                (x_coord + 15, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (215, 220, 227),
                1,
                cv2.LINE_AA)

        sensor_y = self._height - 12
        camera_entries = (
            ('CAM ACTIVE {}'.format(sensor_counts['camera_active']),
             TOPDOWN_COLOR_CAMERA_ACTIVE, True),
            ('CAM INACTIVE {}'.format(sensor_counts['camera_inactive']),
             TOPDOWN_COLOR_CAMERA_INACTIVE, False),
        )
        radar_entries = (
            ('RADAR ACTIVE {}'.format(sensor_counts['radar_active']),
             (TOPDOWN_COLOR_RADAR_PRIORITY
              if degraded else TOPDOWN_COLOR_RADAR_ACTIVE), True),
            ('RADAR INACTIVE {}'.format(sensor_counts['radar_inactive']),
             TOPDOWN_COLOR_RADAR_INACTIVE, False),
        )
        x_coord = self._plot_left
        for label, color, active in camera_entries:
            center = (x_coord + 6, sensor_y - 4)
            triangle = np.asarray((
                (center[0] + 7, center[1]),
                (center[0] - 5, center[1] - 5),
                (center[0] - 5, center[1] + 5),
            ), dtype=np.int32)
            if active:
                cv2.fillPoly(frame, [triangle], color, lineType=cv2.LINE_AA)
            cv2.polylines(
                frame, [triangle], True, color, 2, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                label,
                (x_coord + 18, sensor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (215, 220, 227),
                1,
                cv2.LINE_AA)
            x_coord += 46 + len(label) * 7
        for label, color, active in radar_entries:
            center = (x_coord + 6, sensor_y - 4)
            diamond = np.asarray((
                (center[0], center[1] - 6),
                (center[0] + 6, center[1]),
                (center[0], center[1] + 6),
                (center[0] - 6, center[1]),
            ), dtype=np.int32)
            if active:
                cv2.fillPoly(frame, [diamond], color, lineType=cv2.LINE_AA)
            cv2.polylines(
                frame, [diamond], True, color, 2, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                label,
                (x_coord + 18, sensor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (215, 220, 227),
                1,
                cv2.LINE_AA)
            x_coord += 46 + len(label) * 7

    def render(
            self,
            carla_world,
            hero_actor,
            route_trace=None,
            route_path=None,
            destination_transform=None):
        if not self.ready or hero_actor is None:
            self.clear_selected_sensor_actors()
            return

        now_ms = pygame.time.get_ticks()
        if (
                self._last_refresh_ms is not None
                and now_ms - self._last_refresh_ms < self._refresh_period_ms):
            return
        self._last_refresh_ms = now_ms
        render_started_at = time.perf_counter()

        try:
            hero_transform = hero_actor.get_transform()
        except RuntimeError:
            self.clear_selected_sensor_actors()
            return
        self._center_x = float(hero_transform.location.x)
        self._center_y = float(hero_transform.location.y)
        self.last_network_degradation_strength = network_degradation_strength(
            hero_transform.location,
            self._network_degradation_zones)
        radar_priority_active = self._radar_priority_is_active(
            self.last_network_degradation_strength)

        scene = self._collect_live_scene(
            carla_world,
            hero_actor,
            hero_transform,
            radar_priority_active)

        plot_image = self._draw_static_map()
        self._draw_network_degradation_zones(plot_image)
        self._draw_cooperative_visibility_layer(
            plot_image,
            scene['sensor_markers'],
            scene['occluders'],
            int(hero_actor.id))
        self._draw_route(plot_image, route_trace, route_path)
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

        vehicle_count, pedestrian_count, sensor_counts = self._draw_live_actors(
            plot_image,
            hero_actor,
            hero_transform,
            radar_priority_active,
            scene)
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
            pedestrian_count,
            sensor_counts,
            self.last_network_degradation_strength,
            radar_priority_active)
        # This is local map construction/update latency: CARLA actor queries,
        # geometry drawing, and frame composition.  It intentionally excludes
        # the OpenCV window-present call and is not spatial-map-server E2E time.
        self.last_update_latency_ms = max(
            0.0, (time.perf_counter() - render_started_at) * 1000.0)

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
        return self.last_update_latency_ms

    def close(self):
        self._last_refresh_ms = None
        self.last_update_latency_ms = None
        self.last_network_degradation_strength = 0.0
        self.clear_selected_sensor_actors()
        self._radar_priority_latched = False
        self._radar_priority_exit_refreshes = 0
        self._stream_activation_requested = False
        self._owned_stream_count = 0
        self._listening_stream_count = 0
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
        self.route_config_path = args.route_config
        self.route_guidance_config = None
        self.route_guidance_name = None
        self.route_guidance_active = False
        self.show_route_guidance = False
        self._ego_spawn_was_explicit = bool(args.ego_spawn_was_explicit)
        if self.route_config_path is not None:
            self.route_guidance_config = load_route_config(self.route_config_path)
            if not maps_match(self.route_guidance_config['map'], self.map.name):
                raise ValueError(
                    "Route config map {!r} does not match loaded CARLA map {!r}".format(
                        self.route_guidance_config['map'], self.map.name))
            self.route_guidance_name = self.route_guidance_config['name']

        spawn_x = args.ego_spawn_x
        spawn_y = args.ego_spawn_y
        if self.route_guidance_config is not None and not self._ego_spawn_was_explicit:
            route_start = self.route_guidance_config['start']
            spawn_x = route_start['location']['x']
            spawn_y = route_start['location']['y']
        self.ego_spawn_transform = resolve_ego_spawn_transform(
            self.map,
            spawn_x,
            spawn_y)
        if self.route_guidance_config is not None and not self._ego_spawn_was_explicit:
            route_rotation = self.route_guidance_config['start']['rotation']
            self.ego_spawn_transform.rotation = carla.Rotation(
                pitch=route_rotation['pitch'],
                yaw=route_rotation['yaw'],
                roll=route_rotation['roll'])
            logging.info(
                "Using route config '%s' start as the ego startup/Y-respawn pose",
                self.route_guidance_name)
        elif self.route_guidance_config is not None:
            logging.info(
                'Explicit --ego-spawn-x/--ego-spawn-y override route start; '
                'guidance will include a connector to the saved route')
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
        self._route_sampling_resolution = (
            float(args.route_sampling_resolution)
            if args.route_sampling_resolution is not None
            else float(
                self.route_guidance_config['route_sampling_resolution_m']
                if self.route_guidance_config is not None else 2.0))
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
        self._route_progress_index = 0
        self._route_progress_key = None
        self.last_spawn_transform = None
        self.show_actor_bboxes = False
        self.ego_commanded_direction_sign = 1.0
        self.rogue_pedestrian_role_prefix = args.rogue_pedestrian_role_prefix
        self.rogue_pedestrian_warning_radius = (
            args.rogue_pedestrian_warning_radius)
        self.rogue_pedestrian_brake_radius = args.rogue_pedestrian_brake_radius
        self.show_topdown_map = False
        self.topdown_zoom_radius = args.topdown_zoom_radius
        self._network_degradation_local_override = bool(
            getattr(args, 'network_degradation_zones_explicit', False)
            or getattr(
                args, 'network_degradation_disabled_explicit', False))
        self.network_degradation_zones = tuple(
            args.network_degradation_zones)
        self.network_profile_session_token = None
        self.network_profile_stream_sensors = False
        self.network_profile_manifest_actor_id = None
        self.network_profile_zone_actor_ids = ()
        self._network_profile_next_refresh_at = 0.0
        self._network_profile_missing_refreshes = 0
        self._network_profile_last_diagnostic = None
        self._network_profile_last_signature = None
        self.infrastructure_sensor_traffic_light_ids = tuple(
            args.infrastructure_sensor_traffic_light_ids)
        self.spatial_map_active_sensor_pairs = int(
            args.spatial_map_active_sensor_pairs)
        self.spatial_map_degraded_radar_pairs = int(
            args.spatial_map_degraded_radar_pairs)
        self.spatial_map_sensor_forward_range = float(
            args.spatial_map_sensor_forward_range)
        self.spatial_map_sensor_forward_half_angle = float(
            args.spatial_map_sensor_forward_half_angle)
        self.cooperative_camera_horizontal_fov = float(
            args.cooperative_camera_horizontal_fov)
        self.cooperative_camera_range = float(
            args.cooperative_camera_range)
        self.cooperative_radar_horizontal_fov = float(
            args.cooperative_radar_horizontal_fov)
        self.cooperative_radar_range = float(
            args.cooperative_radar_range)
        self._cooperative_visibility_snapshot = {
            'sampled_at': 0.0,
            'actors': {},
            'cooperatively_detected_actor_ids': (),
            'ego_visible_actor_ids': (),
        }
        # Accepted only so older launch commands remain valid. The spatial-map
        # ego pair is always virtual and therefore needs no disable switch.
        self.disable_ego_sensor_pair = bool(args.disable_ego_sensor_pair)
        self.topdown_renderer = None
        self.selected_sensor_stream_controller = (
            SelectedSensorStreamController())
        self.show_live_metrics = False
        self.live_metrics_model = LiveMetricsModel(
            args.metrics_placeholder_seed,
            network_map_latency_penalty_ms=(
                args.network_degradation_map_latency_penalty_ms),
            network_map_accuracy_penalty_cm=(
                args.network_degradation_map_accuracy_penalty_cm),
            network_sense_to_act_penalty_ms=(
                args.network_degradation_sense_to_act_penalty_ms))
        self.live_metrics_renderer = None
        self._refresh_network_degradation_profile(force=True)
        logging.info(
            'Live metrics on U: measured local detections/map/control timing; '
            'collision outcome proxy; DEMO accuracy=clipped Gaussian '
            'mean %.1f sigma %.2f range %.1f..%.1f cm and reasoning=clipped '
            'Gaussian mean %.1f sigma %.1f range %.1f..%.1f ms '
            '(seed=%d, EWMA alpha=%.2f)',
            METRICS_ACCURACY_MEAN_CM,
            METRICS_ACCURACY_SIGMA_CM,
            METRICS_ACCURACY_MIN_CM,
            METRICS_ACCURACY_MAX_CM,
            METRICS_REASONING_MEAN_MS,
            METRICS_REASONING_SIGMA_MS,
            METRICS_REASONING_MIN_MS,
            METRICS_REASONING_MAX_MS,
            args.metrics_placeholder_seed,
            METRICS_PLACEHOLDER_EWMA_ALPHA)
        if self.network_degradation_zones:
            logging.info(
                'Network degradation model: zones=%s; maximum '
                'additive penalties map_latency=%.1f ms, accuracy_error=%.1f '
                'cm, sense_to_act=%.1f ms; radial smoothstep, no induced '
                'network delay; zone source=%s',
                ', '.join(
                    '({:.3f}, {:.3f}, r={:.1f} m)'.format(*zone)
                    for zone in self.network_degradation_zones),
                args.network_degradation_map_latency_penalty_ms,
                args.network_degradation_map_accuracy_penalty_cm,
                args.network_degradation_sense_to_act_penalty_ms,
                ('explicit local override'
                 if self._network_degradation_local_override else
                 ('shared profile'
                  if self.network_profile_session_token is not None else
                  'built-in fallback')))
        else:
            logging.info(
                'Network degradation model disabled (zone source=%s)',
                ('explicit local override'
                 if self._network_degradation_local_override else
                 'shared profile'))
        logging.info(
            'Spatial-map sensors are virtual-only: co-located RGB/radar pairs '
            'follow the ego vehicle, every live vehicle/walker inside the '
            'bounded inventory ROI (including strict %s hazard roles), and '
            'configured traffic-light '
            'IDs=%s at nominal height %.1f m. The nearest %d complete pair(s) '
            'are shown active in normal coverage; in a degraded zone every '
            'camera is inactive and the nearest %d radar(s) are highlighted. '
            'No map-node camera/radar CARLA actor is spawned, discovered, or '
            'streamed.',
            self.rogue_pedestrian_role_prefix,
            (','.join(
                str(identifier)
                for identifier in self.infrastructure_sensor_traffic_light_ids)
             if self.infrastructure_sensor_traffic_light_ids else 'none'),
            VIRTUAL_TRAFFIC_LIGHT_SENSOR_HEIGHT_M,
            self.spatial_map_active_sensor_pairs,
            self.spatial_map_degraded_radar_pairs)
        logging.info(
            'Spatial-map sensor ROI: %.1f m ahead within +/-%.1f degrees; '
            'side, rear, and farther virtual sites are not shown or selected',
            self.spatial_map_sensor_forward_range,
            self.spatial_map_sensor_forward_half_angle)
        logging.info(
            'Cooperative visibility model: virtual 2-D ground-plane target '
            'footprint probes and occlusion rays; '
            'camera FoV=%.1f deg range=%.1f m, radar FoV=%.1f deg '
            'range=%.1f m; [/- and ]/= adjust both normal and degraded '
            'display-only site budgets; no sensor actor or stream is created',
            self.cooperative_camera_horizontal_fov,
            self.cooperative_camera_range,
            self.cooperative_radar_horizontal_fov,
            self.cooperative_radar_range)
        if self.disable_ego_sensor_pair:
            logging.info(
                'Deprecated --disable-ego-sensor-pair is a no-op; the ego '
                'spatial-map pair is already virtual')
        logging.info(
            'Rogue-pedestrian AR alert on U: role=%s_<index> '
            'warning_radius=%.2f m brake_radius=%.2f m; red box and '
            'PEDESTRIAN label require a fresh detection by any active ego/'
            'peer sensor and clear after the ego passes the actor',
            self.rogue_pedestrian_role_prefix,
            self.rogue_pedestrian_warning_radius,
            self.rogue_pedestrian_brake_radius)
        if args.destination_x is not None and args.destination_y is not None:
            destination_z = args.destination_z if args.destination_z is not None else 0.0
            self.route_destination_override = carla.Location(
                x=args.destination_x,
                y=args.destination_y,
                z=destination_z)
        else:
            self.route_destination_override = None
        try:
            if not self.respawn_at_ego_start(notify=False):
                target = self.ego_spawn_transform.location
                raise RuntimeError(
                    'Configured ego spawn at x={:.2f}, y={:.2f} is occupied'.format(
                        target.x,
                        target.y))
            if self.route_guidance_config is not None:
                if not self._activate_configured_route_guidance():
                    raise RuntimeError(
                        "Unable to build loaded ego route {!r}".format(
                            self.route_guidance_name))
                # Keep v7's startup visual state coherent: U turns the route,
                # boxes, and top-down map on as one group the first time.
                self.show_route_guidance = False
                logging.info(
                    "Loaded route '%s' from %s: %d path points; press U to "
                    "show manual guidance",
                    self.route_guidance_name,
                    self.route_config_path,
                    len(self.route_path))
            self.world.on_tick(hud.on_world_tick)
        except Exception:
            # A Python constructor that raises is never assigned to game_loop's
            # `world` variable. Tear down the already spawned ego and sensors
            # here so a malformed/unreachable route cannot leak CARLA actors.
            try:
                self.destroy(close_visualizers=True)
            except Exception as cleanup_error:
                logging.error(
                    'Failed to clean up after World initialization error: %s',
                    cleanup_error)
            raise
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

    @staticmethod
    def _network_zone_log_text(zones):
        if not zones:
            return 'none'
        return ', '.join(
            '({:.3f}, {:.3f}, r={:.1f} m)'.format(*zone)
            for zone in zones)

    def _update_topdown_stream_status(self):
        """Publish the invariant that virtual spatial-map nodes cannot stream."""
        renderer = self.topdown_renderer
        if renderer is None:
            return
        controller = self.selected_sensor_stream_controller
        renderer.set_stream_activation_requested(
            False,
            len(controller.owned_actor_ids),
            len(controller.listening_actor_ids))

    def _update_network_profile_consumers(self):
        renderer = self.topdown_renderer
        if renderer is not None:
            renderer.set_network_degradation_zones(
                self.network_degradation_zones)
        self._update_topdown_stream_status()

    def _accept_network_degradation_profile(self, profile):
        """Commit one fully validated spawn_blocker world profile."""
        published_zones = normalize_network_degradation_zones(
            profile.zone_tuples)
        session_token = str(profile.session_token)
        manifest_actor_id = int(profile.manifest_actor_id)
        zone_actor_ids = tuple(int(actor_id)
                               for actor_id in profile.zone_actor_ids)
        stream_sensors_requested = bool(profile.start_active_sensors)
        signature = (
            session_token,
            manifest_actor_id,
            zone_actor_ids,
            published_zones,
            stream_sensors_requested,
            self._network_degradation_local_override,
        )

        self.network_profile_session_token = session_token
        self.network_profile_manifest_actor_id = manifest_actor_id
        self.network_profile_zone_actor_ids = zone_actor_ids
        # Profiles still publish zone geometry, but a stream opt-in cannot act
        # on display-only virtual markers.
        self.network_profile_stream_sensors = False
        self._network_profile_missing_refreshes = 0
        self._network_profile_last_diagnostic = None
        if not self._network_degradation_local_override:
            # An empty but committed profile intentionally means no zones; it
            # is distinct from an absent profile, which uses the built-in one.
            self.network_degradation_zones = published_zones
        self.selected_sensor_stream_controller.stop_all(
            'spatial-map sensor sites are virtual')
        self._update_network_profile_consumers()

        if signature != self._network_profile_last_signature:
            logging.info(
                'Accepted network profile token=%s manifest_actor=%d '
                'zone_actors=%s published_zones=%s effective_zones=%s '
                'requested_sensor_collection=%s effective=disabled-virtual%s',
                session_token,
                manifest_actor_id,
                (','.join(str(actor_id) for actor_id in zone_actor_ids)
                 if zone_actor_ids else 'none'),
                self._network_zone_log_text(published_zones),
                self._network_zone_log_text(self.network_degradation_zones),
                'enabled' if stream_sensors_requested else 'disabled',
                (' (local zone override retained)'
                 if self._network_degradation_local_override else ''))
        self._network_profile_last_signature = signature

    def _handle_missing_network_degradation_profile(
            self,
            diagnostic,
            profile_absent=False):
        """Fail streams off; fallback only after confirmed publisher absence."""
        diagnostic = str(diagnostic)
        had_profile = self.network_profile_session_token is not None
        self.network_profile_stream_sensors = False
        self.selected_sensor_stream_controller.stop_all(
            'shared network profile unavailable')
        self._update_network_profile_consumers()

        if not profile_absent:
            # Malformed, partial, or unreadable metadata is not proof that the
            # publisher is absent. Preserve the last committed (or startup
            # fallback) zones. Virtual marker streaming remains impossible.
            if diagnostic != self._network_profile_last_diagnostic:
                logging.warning(
                    'Network profile unavailable (%s); virtual map-node '
                    'streaming remains disabled and effective zones remain '
                    'unchanged at %s',
                    diagnostic,
                    self._network_zone_log_text(
                        self.network_degradation_zones))
            self._network_profile_last_diagnostic = diagnostic
            return

        if had_profile:
            self._network_profile_missing_refreshes += 1
            if diagnostic != self._network_profile_last_diagnostic:
                logging.warning(
                    'Network profile temporarily unavailable (%s); virtual '
                    'map-node streaming remains disabled and the previous '
                    'zones are retained for up to %d refreshes',
                    diagnostic,
                    NETWORK_PROFILE_MISSING_HYSTERESIS_REFRESHES)
            self._network_profile_last_diagnostic = diagnostic
            if (
                    self._network_profile_missing_refreshes
                    < NETWORK_PROFILE_MISSING_HYSTERESIS_REFRESHES):
                return

        self.network_profile_session_token = None
        self.network_profile_manifest_actor_id = None
        self.network_profile_zone_actor_ids = ()
        self._network_profile_missing_refreshes = 0
        if not self._network_degradation_local_override:
            self.network_degradation_zones = (
                DEFAULT_NETWORK_DEGRADATION_ZONE,)
        self._update_network_profile_consumers()
        absent_signature = (
            'absent',
            self.network_degradation_zones,
            self._network_degradation_local_override,
        )
        if absent_signature != self._network_profile_last_signature:
            logging.info(
                'No valid shared network profile (%s): effective_zones=%s; '
                'virtual map-node streaming is disabled%s',
                diagnostic,
                self._network_zone_log_text(self.network_degradation_zones),
                (' (local zone override retained)'
                 if self._network_degradation_local_override else
                 ' (built-in fallback)'))
        self._network_profile_last_signature = absent_signature
        self._network_profile_last_diagnostic = diagnostic

    def _refresh_network_degradation_profile(self, force=False):
        """Poll read-only world metadata without ticking or changing settings."""
        now = time.monotonic()
        if (
                not force
                and now < self._network_profile_next_refresh_at):
            return
        self._network_profile_next_refresh_at = (
            now + NETWORK_PROFILE_REFRESH_SECONDS)

        if discover_network_degradation_profile is None:
            self._handle_missing_network_degradation_profile(
                'profile helper unavailable',
                profile_absent=True)
            return
        try:
            profile = discover_network_degradation_profile(
                self.world,
                strict=True)
        except NetworkProfileError as exc:
            self._handle_missing_network_degradation_profile(
                'invalid transaction: {}'.format(exc))
            return
        except Exception as exc:
            self._handle_missing_network_degradation_profile(
                'discovery failed: {}'.format(exc))
            return

        if profile is None:
            self._handle_missing_network_degradation_profile(
                'publisher absent',
                profile_absent=True)
            return
        try:
            self._accept_network_degradation_profile(profile)
        except (AttributeError, TypeError, ValueError) as exc:
            self._handle_missing_network_degradation_profile(
                'invalid committed values: {}'.format(exc))

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
        # Set up only sensors required by the driving client. The additional
        # RGB/radar site shown on the spatial map is derived virtually from the
        # ego pose in TopDownMapRenderer and creates no CARLA child actors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma, self)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.live_metrics_model.reset_live_measurements()
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
        self._route_progress_index = 0
        self._route_progress_key = None
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
            if enabled:
                # Traffic Manager follows the lane forward; do not carry a
                # stale manual-reverse warning direction into autonomy.
                self.ego_commanded_direction_sign = 1.0
            self.player.set_autopilot(enabled, self.traffic_manager.get_port())

    def _get_route_planner(self):
        if GlobalRoutePlanner is None:
            return None
        if self._route_planner is None:
            self._route_planner = GlobalRoutePlanner(self.map, self._route_sampling_resolution)
        return self._route_planner

    @staticmethod
    def _route_config_location(value):
        return carla.Location(
            x=float(value['x']),
            y=float(value['y']),
            z=float(value['z']))

    @classmethod
    def _route_config_transform(cls, value):
        rotation = value['rotation']
        return carla.Transform(
            cls._route_config_location(value['location']),
            carla.Rotation(
                pitch=float(rotation['pitch']),
                yaw=float(rotation['yaw']),
                roll=float(rotation['roll'])))

    @staticmethod
    def _dedupe_route_path(locations, minimum_distance=0.10):
        result = []
        for location in locations:
            copied = copy_location(location)
            if result and result[-1].distance(copied) < minimum_distance:
                continue
            result.append(copied)
        return result

    def _trace_route_through_locations(self, control_locations):
        """Trace and join every ordered route-control segment once at startup."""
        planner = self._get_route_planner()
        if planner is None:
            return [], []

        route_trace = []
        for segment_number, (segment_start, segment_end) in enumerate(
                zip(control_locations, control_locations[1:]), start=1):
            try:
                segment = list(planner.trace_route(segment_start, segment_end))
            except Exception as exc:
                logging.error(
                    'Route segment %d could not be planned: %s',
                    segment_number,
                    exc)
                return [], []
            if not segment:
                logging.error(
                    'Route segment %d is unreachable near (%.2f, %.2f)',
                    segment_number,
                    segment_end.x,
                    segment_end.y)
                return [], []
            if (
                    route_trace and
                    route_trace[-1][0].transform.location.distance(
                        segment[0][0].transform.location) < 0.25):
                segment = segment[1:]
            route_trace.extend(segment)

        route_path = [
            copy_location(waypoint.transform.location)
            for waypoint, _ in route_trace]
        return route_trace, self._dedupe_route_path(route_path)

    def _waypoint_trace_for_path(self, route_path):
        """Build the lightweight trace needed by the top-down renderer."""
        route_trace = []
        previous_waypoint = None
        for location in route_path:
            try:
                waypoint = self.map.get_waypoint(
                    location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving)
            except RuntimeError:
                waypoint = None
            if waypoint is None:
                continue
            if (
                    previous_waypoint is not None and
                    previous_waypoint.transform.location.distance(
                        waypoint.transform.location) < 0.10):
                continue
            route_trace.append((waypoint, None))
            previous_waypoint = waypoint
        return route_trace

    def _planned_path_from_route_config(self):
        values = self.route_guidance_config.get('planned_path', [])
        if len(values) < 2:
            return []
        route_path = self._dedupe_route_path([
            self._route_config_location(value) for value in values])
        if len(route_path) < 2:
            return []
        configured_start = self._route_config_location(
            self.route_guidance_config['start']['location'])
        configured_end = self._route_config_location(
            self.route_guidance_config['end']['location'])
        if (
                route_path[0].distance(configured_start)
                > ROUTE_CONFIG_ENDPOINT_TOLERANCE_M or
                route_path[-1].distance(configured_end)
                > ROUTE_CONFIG_ENDPOINT_TOLERANCE_M):
            logging.warning(
                'Ignoring planned_path whose endpoints do not match the '
                'route controls within %.1f m',
                ROUTE_CONFIG_ENDPOINT_TOLERANCE_M)
            return []
        search_index = 0
        for order, value in enumerate(
                self.route_guidance_config['intermediate_waypoints'], start=1):
            control = self._route_config_location(value)
            remaining = route_path[search_index:]
            if not remaining:
                return []
            relative_index = min(
                range(len(remaining)),
                key=lambda index: remaining[index].distance(control))
            control_error = remaining[relative_index].distance(control)
            if control_error > ROUTE_CONFIG_CONTROL_TOLERANCE_M:
                logging.warning(
                    'Ignoring planned_path that misses intermediate control '
                    '%d by %.1f m',
                    order,
                    control_error)
                return []
            search_index += relative_index
        return route_path

    def _activate_configured_route_guidance(self, connect_to_saved_start=None):
        """Install a loaded route as passive, human-driver visual guidance."""
        if self.route_guidance_config is None or self.player is None:
            return False

        route_start = self._route_config_location(
            self.route_guidance_config['start']['location'])
        route_end_transform = self._route_config_transform(
            self.route_guidance_config['end'])
        via_locations = [
            self._route_config_location(value)
            for value in self.route_guidance_config['intermediate_waypoints']]
        current_location = copy_location(self.player.get_location())
        planned_path = self._planned_path_from_route_config()

        # A CLI spawn remains authoritative.  Connect it to the saved route;
        # with the normal route-start spawn, consume the dense exported path
        # directly and avoid doing route-planner work in the frame loop.
        if connect_to_saved_start is None:
            connect_to_saved_start = self._ego_spawn_was_explicit
        needs_connector = (
            bool(connect_to_saved_start)
            and current_location.distance(route_start) > 3.0)
        route_trace = []
        route_path = []
        if planned_path and not needs_connector:
            route_path = planned_path
            route_trace = self._waypoint_trace_for_path(route_path)
        elif planned_path:
            connector_trace, connector_path = self._trace_route_through_locations(
                [current_location, route_start])
            if not connector_path:
                return False
            route_trace = connector_trace + self._waypoint_trace_for_path(planned_path)
            route_path = self._dedupe_route_path(connector_path + planned_path)
        else:
            controls = [current_location]
            if needs_connector:
                controls.append(route_start)
            controls.extend(via_locations)
            controls.append(route_end_transform.location)
            controls = self._dedupe_route_path(controls, minimum_distance=0.5)
            route_trace, route_path = self._trace_route_through_locations(controls)
            if not route_path:
                return False

        self.route_loop_active = False
        self.route_loop_autonomous = False
        self.route_guidance_active = True
        self.route_origin_transform = copy_transform(self.ego_spawn_transform)
        self.route_destination_transform = route_end_transform
        self.route_trace = route_trace
        self.route_path = route_path
        self.traffic_manager_route_path = []
        return len(self.route_path) >= 2

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

        self.route_guidance_active = False
        self.route_loop_active = True
        self.show_route_guidance = True
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
        if self.route_guidance_config is not None:
            if not self._activate_configured_route_guidance(
                    connect_to_saved_start=False):
                self.hud.notification(
                    'Route loop Off; unable to restore loaded route guidance',
                    seconds=4.0)

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
        any_visible = (
            self.show_actor_bboxes or
            self.show_topdown_map or
            self.show_live_metrics or
            ((self.route_guidance_active or self.route_loop_active)
             and self.show_route_guidance))
        enable = not any_visible
        self.show_actor_bboxes = enable
        self.show_route_guidance = enable and (
            self.route_guidance_active or self.route_loop_active)
        if enable:
            # Re-read the committed blocker manifest before constructing the
            # windows so U immediately reflects the current shared zones.
            self._refresh_network_degradation_profile(force=True)
            if self.topdown_renderer is not None and not self.topdown_renderer.ready:
                self.topdown_renderer.close()
                self.topdown_renderer = None
            if self.topdown_renderer is None and cv2 is not None:
                try:
                    self.topdown_renderer = TopDownMapRenderer(
                        self.world,
                        self.map,
                        self.topdown_zoom_radius,
                        rogue_pedestrian_role_prefix=(
                            self.rogue_pedestrian_role_prefix),
                        network_degradation_zones=(
                            self.network_degradation_zones),
                        infrastructure_sensor_traffic_light_ids=(
                            self.infrastructure_sensor_traffic_light_ids),
                        active_sensor_pairs=(
                            self.spatial_map_active_sensor_pairs),
                        degraded_active_radars=(
                            self.spatial_map_degraded_radar_pairs),
                        sensor_forward_range_m=(
                            self.spatial_map_sensor_forward_range),
                        sensor_forward_half_angle_degrees=(
                            self.spatial_map_sensor_forward_half_angle),
                        cooperative_camera_horizontal_fov_degrees=(
                            self.cooperative_camera_horizontal_fov),
                        cooperative_camera_range_m=(
                            self.cooperative_camera_range),
                        cooperative_radar_horizontal_fov_degrees=(
                            self.cooperative_radar_horizontal_fov),
                        cooperative_radar_range_m=(
                            self.cooperative_radar_range))
                except Exception as exc:
                    logging.warning('Unable to initialize top-down map: %s', exc)
                    self.topdown_renderer = None
            if self.topdown_renderer is not None:
                self._update_topdown_stream_status()
            self.show_topdown_map = (
                self.topdown_renderer is not None
                and self.topdown_renderer.ready)

            if (
                    self.live_metrics_renderer is not None
                    and not self.live_metrics_renderer.ready):
                self.live_metrics_renderer.close()
                self.live_metrics_renderer = None
            if self.live_metrics_renderer is None and cv2 is not None:
                self.live_metrics_renderer = LiveMetricsRenderer()
            self.show_live_metrics = (
                self.live_metrics_renderer is not None
                and self.live_metrics_renderer.ready)

            enabled_labels = ['actor boxes/rogue alerts']
            if self.show_topdown_map:
                enabled_labels.append('top-down map/sensors/network zone')
            if self.show_route_guidance:
                enabled_labels.append('route arrows')
            if self.show_live_metrics:
                enabled_labels.append('live metrics')
            unavailable = []
            if not self.show_topdown_map:
                unavailable.append('top-down map/sensors/network zone')
            if not self.show_live_metrics:
                unavailable.append('live metrics')
            message = '{} On'.format(', '.join(enabled_labels))
            if unavailable:
                reason = 'OpenCV missing' if cv2 is None else 'unavailable'
                message += ' ({}: {})'.format(
                    ', '.join(unavailable), reason)
            self.hud.notification(message, seconds=3.0)
        else:
            self.show_topdown_map = False
            self.show_route_guidance = False
            self.show_live_metrics = False
            self._clear_cooperative_visibility()
            self.selected_sensor_stream_controller.stop_all(
                'U visualizations disabled')
            if self.topdown_renderer is not None:
                self.topdown_renderer.close()
            if self.live_metrics_renderer is not None:
                self.live_metrics_renderer.close()
            self.hud.notification(
                'Route arrows, actor boxes/rogue alerts, top-down map/sensors/'
                'network zone, and live metrics Off')

    def _clear_cooperative_visibility(self):
        self._cooperative_visibility_snapshot = {
            'sampled_at': 0.0,
            'actors': {},
            'cooperatively_detected_actor_ids': (),
            'ego_visible_actor_ids': (),
        }
        if self.topdown_renderer is not None:
            self.topdown_renderer.clear_cooperative_visibility()

    def _sync_cooperative_visibility(self):
        renderer = self.topdown_renderer
        if (
                renderer is None
                or not renderer.ready
                or not self.show_topdown_map):
            self._clear_cooperative_visibility()
            return
        snapshot = renderer.last_cooperative_visibility
        self._cooperative_visibility_snapshot = {
            'sampled_at': float(snapshot.get('sampled_at', 0.0)),
            'actors': dict(snapshot.get('actors', {})),
            'cooperatively_detected_actor_ids': tuple(
                snapshot.get('cooperatively_detected_actor_ids', ())),
            'ego_visible_actor_ids': tuple(
                snapshot.get('ego_visible_actor_ids', ())),
        }

    def cooperative_visibility_state(self, actor_id):
        """Return fresh ego/peer visibility authority for one actor ID."""
        if not self.show_topdown_map:
            return 'UNRESOLVED_OCCLUDED'
        snapshot = self._cooperative_visibility_snapshot
        sampled_at = float(snapshot.get('sampled_at', 0.0))
        if (
                sampled_at <= 0.0
                or time.perf_counter() - sampled_at
                > COOPERATIVE_VISIBILITY_MAX_AGE_SECONDS):
            return 'UNRESOLVED_OCCLUDED'
        try:
            result = snapshot.get('actors', {}).get(int(actor_id), {})
        except (TypeError, ValueError):
            return 'UNRESOLVED_OCCLUDED'
        if bool(result.get('visible_to_ego', False)):
            return 'EGO_VISIBLE'
        if (
                bool(result.get('visible_to_peer', False))
                or bool(result.get('cooperatively_detected', False))):
            return 'COOPERATIVELY_REVEALED'
        return 'UNRESOLVED_OCCLUDED'

    def adjust_cooperative_sensor_pairs(self, delta):
        """Change the live virtual-site budget and revoke stale warnings."""
        try:
            delta = int(delta)
        except (TypeError, ValueError):
            return False
        if delta == 0:
            return False
        current = int(self.spatial_map_active_sensor_pairs)
        updated = max(
            0,
            min(MAX_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS, current + delta))
        if updated == current:
            self.hud.notification(
                'Cooperative virtual sensor sites remain {}'.format(current))
            return False
        self.spatial_map_active_sensor_pairs = updated
        # Runtime selection represents one operator-controlled site budget.
        # In degraded coverage it maps to the same number of radar-only sites.
        self.spatial_map_degraded_radar_pairs = updated
        self._clear_cooperative_visibility()
        if self.topdown_renderer is not None:
            self.topdown_renderer.set_active_sensor_pair_limits(
                updated, updated)
        self.hud.notification(
            'Cooperative virtual sensor sites: {} (normal camera+radar; '
            'degraded radar-only)'.format(updated),
            seconds=2.5)
        return True

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
            if self.show_route_guidance:
                draw_route_waypoints(
                    self.world,
                    self.route_trace,
                    origin_transform=self.route_origin_transform,
                    destination_transform=self.route_destination_transform)
        if self.camera_manager is not None:
            self.camera_manager.sync_head_pose_to_vehicle()
        self.hud.tick(self, clock)

    def render(self, display):
        self._refresh_network_degradation_profile()
        if self.show_topdown_map and self.topdown_renderer is not None and self.player is not None:
            show_route = self.show_route_guidance and (
                self.route_guidance_active or self.route_loop_active)
            # Spatial-map sensor sites are display-only; stream status remains
            # hard-disabled even if a legacy shared profile requests capture.
            self._update_topdown_stream_status()
            map_latency_ms = self.topdown_renderer.render(
                self.world,
                self.player,
                route_trace=self.route_trace if show_route else None,
                route_path=self.route_path if show_route else None,
                destination_transform=(
                    self.route_destination_transform if show_route else None))
            if not self.topdown_renderer.ready:
                self.show_topdown_map = False
                self._clear_cooperative_visibility()
                self.selected_sensor_stream_controller.stop_all(
                    'top-down renderer unavailable')
                self.hud.notification(
                    'Top-down map unavailable; sensor display disabled',
                    seconds=3.0)
            else:
                self.selected_sensor_stream_controller.stop_all(
                    'spatial-map sensor sites are virtual')
                self._update_topdown_stream_status()
                if map_latency_ms is not None:
                    self.live_metrics_model.note_spatial_map_latency(
                        map_latency_ms)
                # Publish the current map decision before composing the ego
                # frame, so a sensor-authorized box cannot appear one refresh
                # early.
                self._sync_cooperative_visibility()
        else:
            self._clear_cooperative_visibility()
            self.selected_sensor_stream_controller.stop_all(
                'top-down sensor display inactive')
        self.camera_manager.render(display)
        self.hud.render(display)
        if self.show_live_metrics and self.live_metrics_renderer is not None:
            degradation_strength = 0.0
            if self.player is not None:
                try:
                    degradation_strength = network_degradation_strength(
                        self.player.get_location(),
                        self.network_degradation_zones)
                except RuntimeError:
                    degradation_strength = 0.0
            collision_count = (
                len(self.collision_sensor.history)
                if self.collision_sensor is not None else 0)
            radar_priority_active = degradation_strength > 0.0
            if (
                    self.topdown_renderer is not None
                    and self.topdown_renderer.ready):
                # Keep the metrics modality label synchronized with the map's
                # deterministic boundary hysteresis.
                radar_priority_active = (
                    self.topdown_renderer.radar_priority_active)
            metrics = self.live_metrics_model.snapshot(
                collision_count=collision_count,
                network_degradation_strength_value=degradation_strength,
                network_sensor_policy=(
                    'RADAR PRIORITY'
                    if radar_priority_active else
                    'BALANCED'))
            if not self.live_metrics_renderer.render(metrics):
                self.show_live_metrics = False

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self, close_visualizers=False):
        if self.radar_sensor is not None:
            self.toggle_radar()
        # Defensive cleanup for legacy controller state. The virtual map-node
        # path never populates this controller.
        self.selected_sensor_stream_controller.stop_all(
            'ego respawn or client shutdown')
        self._update_topdown_stream_status()
        if self.topdown_renderer is not None:
            self.topdown_renderer.clear_selected_sensor_actors()
        if close_visualizers and self.topdown_renderer is not None:
            self.topdown_renderer.close()
        if close_visualizers and self.live_metrics_renderer is not None:
            self.live_metrics_renderer.close()
        sensors = [
            getattr(self.camera_manager, 'sensor', None),
            getattr(self.collision_sensor, 'sensor', None),
            getattr(self.lane_invasion_sensor, 'sensor', None),
            getattr(self.gnss_sensor, 'sensor', None),
            getattr(self.imu_sensor, 'sensor', None)]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.camera_manager = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        if self.player is not None:
            self.player.destroy()
            self.player = None


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        world.ego_commanded_direction_sign = 1.0
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
        world.ego_commanded_direction_sign = 1.0
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
                elif event.key in (K_LEFTBRACKET, K_MINUS):
                    step = (
                        COOPERATIVE_SENSOR_COUNT_FAST_STEP
                        if pygame.key.get_mods() & KMOD_SHIFT else 1)
                    world.adjust_cooperative_sensor_pairs(-step)
                elif event.key in (K_RIGHTBRACKET, K_EQUALS):
                    step = (
                        COOPERATIVE_SENSOR_COUNT_FAST_STEP
                        if pygame.key.get_mods() & KMOD_SHIFT else 1)
                    world.adjust_cooperative_sensor_pairs(step)
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

        control_submitted = False
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(keys, clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Publish the driver's commanded travel direction for the
                # pass-by warning.  Vehicle.get_control() always exposes a
                # VehicleControl, so it cannot report a stopped Ackermann
                # direction change on its own.
                reverse_commanded = (
                    not world.constant_velocity_enabled and
                    (self._ackermann_reverse < 0
                     if self._ackermann_enabled
                     else self._control.reverse))
                world.ego_commanded_direction_sign = (
                    -1.0 if reverse_commanded else 1.0)
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
                control_submitted = True

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(keys, clock.get_time(), world)
                world.player.apply_control(self._control)
                control_submitted = True

        if control_submitted:
            world.live_metrics_model.note_control_submitted(
                world.camera_manager.latest_camera_received_at,
                time.perf_counter())

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
        # Atomically pair each decoded camera surface with the pose recorded on
        # that exact sensor frame.  Using sensor.get_transform() later causes
        # visible AR jitter while the ego or the head-look camera is moving.
        self._latest_camera_frame = None
        self._parent = parent_actor
        self.hud = hud
        self._world_wrapper = world_wrapper
        self.recording = False
        self._nearest_rogue_pedestrian_distance = None
        self._rogue_travel_direction_sign = 1.0
        self._rogue_label_font = pygame.font.Font(
            pygame.font.get_default_font(), 18)
        self._rogue_warning_title_font = pygame.font.Font(
            pygame.font.get_default_font(), 22)
        self._rogue_warning_action_font = pygame.font.Font(
            pygame.font.get_default_font(), 25)
        self._rogue_label_surface = self._rogue_label_font.render(
            ROGUE_PEDESTRIAN_LABEL, True, (255, 255, 255))
        self._rogue_warning_surfaces = {
            ROGUE_PEDESTRIAN_SLOW_ACTION:
                self._build_rogue_warning_surface(
                    ROGUE_PEDESTRIAN_SLOW_ACTION,
                    ROGUE_PEDESTRIAN_CAUTION_COLOR),
            ROGUE_PEDESTRIAN_BRAKE_ACTION:
                self._build_rogue_warning_surface(
                    ROGUE_PEDESTRIAN_BRAKE_ACTION,
                    ROGUE_PEDESTRIAN_WARNING_COLOR),
        }
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
                self._latest_camera_frame = None
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

    def _build_rogue_warning_surface(self, action_text, action_color):
        """Build one reusable, translucent top-right warning card."""
        card = pygame.Surface(
            (ROGUE_WARNING_CARD_WIDTH, ROGUE_WARNING_CARD_HEIGHT),
            pygame.SRCALPHA)
        bounds = card.get_rect()
        pygame.draw.rect(
            card, (7, 9, 13, 230), bounds, border_radius=12)
        pygame.draw.rect(
            card,
            ROGUE_PEDESTRIAN_WARNING_COLOR,
            bounds,
            2,
            border_radius=12)

        triangle = ((18, 91), (48, 22), (78, 91))
        pygame.draw.polygon(
            card, ROGUE_PEDESTRIAN_WARNING_COLOR, triangle, 3)
        exclamation = self._rogue_warning_action_font.render(
            '!', True, (255, 255, 255))
        card.blit(
            exclamation,
            (48 - exclamation.get_width() // 2,
             50 - exclamation.get_height() // 2))

        title_first, title_second = ROGUE_PEDESTRIAN_WARNING_TITLE.split(
            ' ', 1)
        approaching = self._rogue_warning_title_font.render(
            title_first, True, (245, 245, 245))
        pedestrian = self._rogue_warning_title_font.render(
            title_second, True, (245, 245, 245))
        action = self._rogue_warning_action_font.render(
            action_text, True, action_color)
        card.blit(approaching, (96, 12))
        card.blit(pedestrian, (96, 39))
        card.blit(action, (96, 72))
        return card

    def _draw_rogue_pedestrian_label(self, surface, bounding_box):
        """Draw the cached PEDESTRIAN class label adjacent to one red box."""
        x1, y1, _, _ = bounding_box
        padding_x = 6
        padding_y = 3
        label_width = self._rogue_label_surface.get_width() + 2 * padding_x
        label_height = self._rogue_label_surface.get_height() + 2 * padding_y
        label_x = max(0, min(int(x1), surface.get_width() - label_width))
        above_y = int(y1) - label_height - 2
        label_y = above_y if above_y >= 0 else min(
            surface.get_height() - label_height,
            int(y1) + 2)
        label_y = max(0, label_y)
        label_rect = pygame.Rect(
            label_x, label_y, label_width, label_height)
        pygame.draw.rect(
            surface, (160, 18, 18), label_rect, border_radius=3)
        pygame.draw.rect(
            surface,
            ROGUE_PEDESTRIAN_BOX_COLOR,
            label_rect,
            1,
            border_radius=3)
        surface.blit(
            self._rogue_label_surface,
            (label_x + padding_x, label_y + padding_y))

    def _draw_rogue_pedestrian_warning(self, surface, distance):
        """Draw the cached proximity-severity warning over every AR element."""
        action = rogue_pedestrian_warning_action(
            distance,
            self._world_wrapper.rogue_pedestrian_brake_radius)
        card = self._rogue_warning_surfaces[action]
        margin = 16
        x_coord = max(0, surface.get_width() - card.get_width() - margin)
        y_coord = min(
            margin,
            max(0, surface.get_height() - card.get_height()))
        surface.blit(card, (x_coord, y_coord))

    def render(self, display):
        frame = self._latest_camera_frame
        source_surface = self.surface
        camera_transform = None
        self._nearest_rogue_pedestrian_distance = None
        if frame is not None:
            source_surface, camera_transform, _ = frame
        if source_surface is not None:
            rendered_surface = source_surface
            should_draw_route = self._should_draw_route_overlay()
            if self._world_wrapper.show_actor_bboxes or should_draw_route:
                rendered_surface = source_surface.copy()
                if should_draw_route:
                    self._draw_route_overlay(
                        rendered_surface,
                        camera_transform=camera_transform)
                # Actor boxes, labels, and the warning card stay above the
                # road-painted guidance so the hazard UI remains readable.
                if self._world_wrapper.show_actor_bboxes:
                    self._draw_actor_bboxes(
                        rendered_surface,
                        camera_transform=camera_transform)
                if self._nearest_rogue_pedestrian_distance is not None:
                    self._draw_rogue_pedestrian_warning(
                        rendered_surface,
                        self._nearest_rogue_pedestrian_distance)
            display.blit(rendered_surface, (0, 0))

    @property
    def latest_camera_received_at(self):
        frame = self._latest_camera_frame
        return frame[2] if frame is not None else None

    def _is_camera_sensor(self):
        return (
            self.sensor is not None and
            self.index is not None and
            self.sensors[self.index][0].startswith('sensor.camera'))

    def _should_draw_route_overlay(self):
        return (
            self._is_camera_sensor() and
            # Sensor 0 is the undistorted RGB feed. The optional distorted RGB
            # lens needs a nonlinear projection model, so do not place pinhole
            # route geometry on that modality.
            self.index == 0 and
            self._world_wrapper.show_route_guidance and
            (self._world_wrapper.route_guidance_active or
             self._world_wrapper.route_loop_active) and
            bool(self._world_wrapper.route_path or
                 self._world_wrapper.route_trace))

    def _route_locations_ahead(self, max_distance=65.0):
        route_source = self._world_wrapper.route_path
        if route_source:
            route_locations = route_source
        else:
            route_source = self._world_wrapper.route_trace
            route_locations = [waypoint.transform.location for waypoint, _ in self._world_wrapper.route_trace]
        if len(route_locations) < 2:
            return []
        route_key = (id(route_source), len(route_locations))
        route_changed = route_key != self._world_wrapper._route_progress_key
        if route_changed:
            # A new/rebuilt route gets one global acquisition. Progress lives
            # on World, so TAB/Backspace camera replacement keeps its cursor.
            # Steady-state frames use a bounded, monotonic search so long
            # routes stay cheap and self-crossings do not jump backward.
            search_start = 0
            search_stop = len(route_locations) - 1
            progress_index = 0
        else:
            progress_index = self._world_wrapper._route_progress_index
            search_start = max(0, progress_index - 3)
            search_stop = min(len(route_locations) - 1, progress_index + 300)

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

        best_segment_index = search_start
        best_start_location = copy_location(route_locations[search_start])
        best_score = float('inf')
        saw_forward_segment = False
        for index in range(search_start, search_stop):
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
                best_score = float('inf')
            tie_break_weight = 1e-6 if route_changed else 0.01
            score = distance + tie_break_weight * max(
                0, index - progress_index)
            if score < best_score:
                best_segment_index = index
                best_start_location = projected_location
                best_score = score

        if not route_changed and best_segment_index < progress_index:
            best_segment_index = progress_index
            best_start_location = _project_to_segment(
                route_locations[best_segment_index],
                route_locations[best_segment_index + 1],
                front_location)
        self._world_wrapper._route_progress_key = route_key
        self._world_wrapper._route_progress_index = best_segment_index

        selected_locations = [best_start_location]
        travelled = 0.0
        previous_location = best_start_location

        for index in range(best_segment_index + 1, len(route_locations)):
            location = route_locations[index]
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

    def _route_segments_for_camera(
            self,
            surface,
            camera_transform=None,
            route_locations=None):
        width = surface.get_width()
        height = surface.get_height()
        field_of_view = float(self.sensor.attributes.get('fov', 90.0))
        calibration = get_camera_K(width, height, field_of_view)
        if camera_transform is None:
            camera_transform = self.sensor.get_transform()
        if route_locations is None:
            route_locations = self._route_locations_ahead(
                max_distance=ROUTE_GUIDANCE_LOOKAHEAD_M)
        locations = [
            location + carla.Location(z=0.08)
            for location in route_locations]
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
    def _sample_route_arrows(route_locations):
        """Return world-space arrow polygons sampled along a route polyline."""
        if len(route_locations) < 2:
            return []
        arrows = []
        travelled = 0.0
        next_arrow_distance = ROUTE_ARROW_START_M
        half_length = ROUTE_ARROW_LENGTH_M * 0.5
        half_width = ROUTE_ARROW_WIDTH_M * 0.5
        shaft_half_width = half_width * 0.36
        shoulder = ROUTE_ARROW_LENGTH_M * 0.08

        for start, end in zip(route_locations, route_locations[1:]):
            delta_x = float(end.x - start.x)
            delta_y = float(end.y - start.y)
            segment_length = math.hypot(delta_x, delta_y)
            if segment_length < 1e-3:
                continue
            tangent_x = delta_x / segment_length
            tangent_y = delta_y / segment_length
            right_x = -tangent_y
            right_y = tangent_x
            while next_arrow_distance <= travelled + segment_length:
                ratio = (next_arrow_distance - travelled) / segment_length
                center_x = float(start.x + delta_x * ratio)
                center_y = float(start.y + delta_y * ratio)
                center_z = float(start.z + (end.z - start.z) * ratio + 0.09)
                local_points = (
                    (-half_length, -shaft_half_width),
                    (shoulder, -shaft_half_width),
                    (shoulder, -half_width),
                    (half_length, 0.0),
                    (shoulder, half_width),
                    (shoulder, shaft_half_width),
                    (-half_length, shaft_half_width),
                )
                arrows.append([
                    carla.Location(
                        x=center_x + tangent_x * forward + right_x * lateral,
                        y=center_y + tangent_y * forward + right_y * lateral,
                        z=center_z)
                    for forward, lateral in local_points
                ])
                next_arrow_distance += ROUTE_ARROW_SPACING_M
            travelled += segment_length
        return arrows

    @staticmethod
    def _project_world_polygon(
            locations,
            camera_transform,
            calibration,
            width,
            height):
        points_world = np.asarray(
            [[point.x, point.y, point.z] for point in locations],
            dtype=np.float32)
        points_camera = world_to_camera(points_world, camera_transform)
        depth = points_camera[:, 0]
        if np.any(depth <= 0.25):
            return None
        u_values = (
            calibration[0, 2] +
            (points_camera[:, 1] / depth) * calibration[0, 0])
        v_values = (
            calibration[1, 2] -
            (points_camera[:, 2] / depth) * calibration[1, 1])
        if not np.all(np.isfinite(u_values)) or not np.all(np.isfinite(v_values)):
            return None
        margin = 240
        if (
                np.all(u_values < -margin) or
                np.all(u_values > width + margin) or
                np.all(v_values < -margin) or
                np.all(v_values > height + margin)):
            return None
        return [
            (int(round(u_value)), int(round(v_value)))
            for u_value, v_value in zip(u_values, v_values)]

    def _draw_route_overlay(self, surface, camera_transform=None):
        if camera_transform is None:
            camera_transform = self.sensor.get_transform()
        route_locations = self._route_locations_ahead(
            max_distance=ROUTE_GUIDANCE_LOOKAHEAD_M)
        route_segments = self._route_segments_for_camera(
            surface,
            camera_transform=camera_transform,
            route_locations=route_locations)
        if not route_segments:
            return

        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        width = surface.get_width()
        height = surface.get_height()
        line_width = max(3, int(width / 360))
        route_outline = (5, 48, 58, 180)
        route_color = (32, 205, 238, 165)
        arrow_outline = (4, 55, 68, 225)
        arrow_fill = (42, 224, 255, 205)

        for segment in route_segments:
            pygame.draw.lines(
                overlay, route_outline, False, segment, line_width + 3)
            pygame.draw.lines(
                overlay, route_color, False, segment, line_width)

        field_of_view = float(self.sensor.attributes.get('fov', 90.0))
        calibration = get_camera_K(width, height, field_of_view)
        for polygon_world in self._sample_route_arrows(route_locations):
            polygon_screen = self._project_world_polygon(
                polygon_world,
                camera_transform,
                calibration,
                width,
                height)
            if polygon_screen is None:
                continue
            pygame.draw.polygon(overlay, arrow_fill, polygon_screen)
            pygame.draw.polygon(overlay, arrow_outline, polygon_screen, 2)

        surface.blit(overlay, (0, 0))

    def _draw_actor_bboxes(self, surface, camera_transform=None):
        self._nearest_rogue_pedestrian_distance = None
        if not self._is_camera_sensor():
            self._world_wrapper.live_metrics_model.note_events_detected(0)
            return 0

        width = surface.get_width()
        height = surface.get_height()
        field_of_view = float(self.sensor.attributes.get('fov', 90.0))
        calibration = get_camera_K(width, height, field_of_view)
        if camera_transform is None:
            camera_transform = self.sensor.get_transform()
        try:
            ego_transform = self._parent.get_transform()
            ego_velocity = self._parent.get_velocity()
            world_actors = self._parent.get_world().get_actors()
        except RuntimeError:
            self._world_wrapper.live_metrics_model.note_events_detected(0)
            return 0
        ego_location = ego_transform.location
        ego_planar_radius = actor_planar_bounding_radius(self._parent)
        ego_travel_forward = ego_transform.get_forward_vector()
        longitudinal_speed = (
            float(ego_velocity.x) * float(ego_travel_forward.x) +
            float(ego_velocity.y) * float(ego_travel_forward.y))
        travel_direction_sign = getattr(
            self, '_rogue_travel_direction_sign', 1.0)
        if math.isfinite(longitudinal_speed):
            if longitudinal_speed < -ROGUE_PEDESTRIAN_REVERSE_SPEED_THRESHOLD_MPS:
                travel_direction_sign = -1.0
            elif longitudinal_speed > ROGUE_PEDESTRIAN_REVERSE_SPEED_THRESHOLD_MPS:
                travel_direction_sign = 1.0

        if (
                math.isfinite(longitudinal_speed) and
                abs(longitudinal_speed) <=
                ROGUE_PEDESTRIAN_REVERSE_SPEED_THRESHOLD_MPS):
            try:
                commanded_sign = float(
                    self._world_wrapper.ego_commanded_direction_sign)
            except (AttributeError, TypeError, ValueError):
                commanded_sign = travel_direction_sign
            if math.isfinite(commanded_sign):
                if commanded_sign < 0.0:
                    travel_direction_sign = -1.0
                elif commanded_sign > 0.0:
                    travel_direction_sign = 1.0
        self._rogue_travel_direction_sign = travel_direction_sign

        if travel_direction_sign < 0.0:
            ego_travel_forward = carla.Vector3D(
                x=-float(ego_travel_forward.x),
                y=-float(ego_travel_forward.y),
                z=-float(ego_travel_forward.z))
        actor_specs = (
            ('vehicle.*', (64, 160, 255)),
            ('walker.pedestrian.*', (0, 255, 0)),
        )
        events_detected = 0

        for pattern, color in actor_specs:
            for actor in world_actors.filter(pattern):
                if actor.id == self._parent.id:
                    continue
                is_rogue_pedestrian = actor_is_rogue_pedestrian(
                    actor,
                    self._world_wrapper.rogue_pedestrian_role_prefix)
                cooperative_state = None
                if is_rogue_pedestrian:
                    cooperative_state = (
                        self._world_wrapper.cooperative_visibility_state(
                            actor.id))
                    if cooperative_state == 'UNRESOLVED_OCCLUDED':
                        # Do not leak the ground-truth projection through the
                        # occluder before the spatial map has an observation.
                        continue
                try:
                    actor_transform = actor.get_transform()
                except RuntimeError:
                    continue
                actor_location = actor_transform.location
                planar_distance = math.hypot(
                    float(actor_location.x - ego_location.x),
                    float(actor_location.y - ego_location.y))
                render_distance_limit = 90.0
                if (
                        is_rogue_pedestrian
                        and rogue_pedestrian_warning_is_sensor_authorized(
                            cooperative_state)):
                    render_distance_limit = max(
                        render_distance_limit,
                        self._world_wrapper.rogue_pedestrian_warning_radius)
                if planar_distance > render_distance_limit:
                    continue

                alert_distance = None
                if (
                        is_rogue_pedestrian
                        and rogue_pedestrian_warning_is_sensor_authorized(
                            cooperative_state)):
                    pass_clearance = max(
                        ROGUE_PEDESTRIAN_PASS_MARGIN_M,
                        ego_planar_radius +
                        actor_planar_bounding_radius(actor) +
                        ROGUE_PEDESTRIAN_PASS_EXTRA_CLEARANCE_M)
                    alert_distance = rogue_pedestrian_alert_distance(
                        ego_transform,
                        actor_transform,
                        self._world_wrapper.rogue_pedestrian_warning_radius,
                        pass_margin=pass_clearance,
                        travel_forward=ego_travel_forward)
                    if (
                            alert_distance is not None and
                            (self._nearest_rogue_pedestrian_distance is None or
                             alert_distance <
                             self._nearest_rogue_pedestrian_distance)):
                        # The warning card is a proximity safety signal and
                        # remains active even if the driver looks away or the
                        # close-range box temporarily fills the viewport.
                        self._nearest_rogue_pedestrian_distance = alert_distance
                try:
                    bounding_box = project_bbox_corners_to_2d(
                        actor_transform,
                        actor.bounding_box,
                        camera_transform,
                        calibration,
                        width,
                        height)
                except (AttributeError, RuntimeError, ValueError):
                    continue
                if bounding_box is None:
                    continue

                draw_color = color
                line_width = 2
                if alert_distance is not None:
                    draw_color = ROGUE_PEDESTRIAN_BOX_COLOR
                    line_width = 3

                x1, y1, x2, y2 = bounding_box
                pygame.draw.rect(
                    surface,
                    draw_color,
                    pygame.Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                    line_width)
                if alert_distance is not None:
                    self._draw_rogue_pedestrian_label(
                        surface, bounding_box)
                events_detected += 1
        self._world_wrapper.live_metrics_model.note_events_detected(
            events_detected,
            sampled_at=self.latest_camera_received_at)
        return events_detected

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        camera_received_at = time.perf_counter()
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
        if self.sensors[self.index][0].startswith('sensor.camera'):
            # Tuple assignment is atomic under CPython, so render() never pairs
            # one camera image with another frame's sensor pose.
            self._latest_camera_frame = (
                self.surface,
                copy_transform(image.transform),
                camera_received_at)
        else:
            self._latest_camera_frame = None
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
        pygame.display.set_caption('CARLA Manual Control AR v13')
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
        '--spatial-map-sensor-forward-range',
        '--spatial-map-sensor-forward-distance',
        dest='spatial_map_sensor_forward_range',
        metavar='METERS',
        default=DEFAULT_SPATIAL_MAP_SENSOR_FORWARD_RANGE_M,
        type=positive_finite_float,
        help=(
            'maximum planar distance for virtual sensor-site display in '
            'front of the ego vehicle (default: %(default)s m)'))
    argparser.add_argument(
        '--spatial-map-sensor-forward-half-angle',
        dest='spatial_map_sensor_forward_half_angle',
        metavar='DEGREES',
        default=DEFAULT_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG,
        type=positive_finite_float,
        help=(
            'half-angle of the forward virtual-site cone; 45 means +/-45 '
            'degrees and excludes side/rear sites '
            '(default: %(default)s degrees)'))
    argparser.add_argument(
        '--cooperative-sensor-pair-horizontal-fov',
        '--sensor-pair-horizontal-fov',
        dest='cooperative_sensor_pair_horizontal_fov',
        metavar='DEGREES',
        default=None,
        type=positive_finite_float,
        help=(
            'pair-wide horizontal field of view override applied equally to '
            'every active virtual RGB camera and radar; cannot be combined '
            'with either modality-specific FoV option (default: use camera '
            '{:.1f} and radar {:.1f} degrees)'.format(
                DEFAULT_COOPERATIVE_CAMERA_HORIZONTAL_FOV_DEG,
                DEFAULT_COOPERATIVE_RADAR_HORIZONTAL_FOV_DEG)))
    argparser.add_argument(
        '--cooperative-camera-horizontal-fov',
        metavar='DEGREES',
        default=None,
        type=positive_finite_float,
        help=(
            'horizontal field of view used by each active virtual RGB camera '
            'for 2-D cooperative visibility (default: {:.1f} degrees)'.format(
                DEFAULT_COOPERATIVE_CAMERA_HORIZONTAL_FOV_DEG)))
    argparser.add_argument(
        '--cooperative-camera-range',
        metavar='METERS',
        default=DEFAULT_COOPERATIVE_CAMERA_RANGE_M,
        type=positive_finite_float,
        help=(
            '2-D visibility range of each active virtual RGB camera '
            '(default: %(default)s m)'))
    argparser.add_argument(
        '--cooperative-radar-horizontal-fov',
        metavar='DEGREES',
        default=None,
        type=positive_finite_float,
        help=(
            'horizontal field of view used by each active virtual radar '
            '(default: {:.1f} degrees)'.format(
                DEFAULT_COOPERATIVE_RADAR_HORIZONTAL_FOV_DEG)))
    argparser.add_argument(
        '--cooperative-radar-range',
        metavar='METERS',
        default=DEFAULT_COOPERATIVE_RADAR_RANGE_M,
        type=positive_finite_float,
        help=(
            '2-D visibility range of each active virtual radar '
            '(default: %(default)s m)'))
    argparser.add_argument(
        '--disable-ego-sensor-pair',
        dest='disable_ego_sensor_pair',
        action='store_true',
        help=(
            'deprecated compatibility no-op; the displayed ego RGB/radar pair '
            'is always virtual and no map-only sensor actors are attached'))
    argparser.add_argument(
        '--disable-managed-sensor-pairs',
        dest='legacy_disable_managed_sensor_pairs',
        action='store_true',
        help=argparse.SUPPRESS)
    argparser.add_argument(
        '--spatial-map-active-sensor-pairs',
        '--managed-sensor-active-pairs',
        dest='spatial_map_active_sensor_pairs',
        metavar='COUNT',
        default=DEFAULT_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS,
        type=nonnegative_int,
        help=(
            'nearest virtual co-located camera/radar sites shown active '
            'outside degraded zones; this is a display-only selection '
            '(default: %(default)s)'))
    argparser.add_argument(
        '--spatial-map-degraded-radar-pairs',
        '--managed-sensor-degraded-radar-pairs',
        dest='spatial_map_degraded_radar_pairs',
        metavar='COUNT',
        default=DEFAULT_SPATIAL_MAP_DEGRADED_RADAR_PAIRS,
        type=nonnegative_int,
        help=(
            'nearest virtual radar sites shown active inside a degraded zone; '
            'every camera remains visually inactive and no data is collected '
            '(default: %(default)s)'))
    # Retired scene-wide attachment flags remain accepted as hidden no-ops so
    # existing launch scripts keep working. Scene-wide attachment is always off.
    argparser.add_argument(
        '--managed-sensor-pairs-per-category',
        dest='legacy_managed_sensor_pairs_per_category',
        metavar='COUNT',
        default=None,
        type=nonnegative_int,
        help=argparse.SUPPRESS)
    argparser.add_argument(
        '--managed-sensor-degraded-camera-pairs',
        dest='legacy_managed_sensor_degraded_camera_pairs',
        metavar='COUNT',
        default=None,
        type=nonnegative_int,
        help=argparse.SUPPRESS)
    argparser.add_argument(
        '--managed-sensor-refresh-seconds',
        dest='legacy_managed_sensor_refresh_seconds',
        metavar='SECONDS',
        default=None,
        type=positive_finite_float,
        help=argparse.SUPPRESS)
    argparser.add_argument(
        '--metrics-placeholder-seed', metavar='SEED',
        default=DEFAULT_METRICS_PLACEHOLDER_SEED, type=int,
        help=(
            'dedicated reproducibility seed for the DEMO-only spatial-map '
            'accuracy and AI-reasoning metrics (default: %(default)s)'))
    argparser.add_argument(
        '--network-degradation-zone',
        dest='network_degradation_zones',
        metavar=('X', 'Y', 'RADIUS'),
        nargs=3,
        action='append',
        default=None,
        type=finite_float,
        help=(
            'repeatable cellular-degradation circle in CARLA meters; one or '
            'more explicit zones locally override shared spawn_blocker '
            'profile locations and the Town10 fallback '
            '({:.3f}, {:.3f}, r={:.1f} m)'.format(
                *DEFAULT_NETWORK_DEGRADATION_ZONE)))
    argparser.add_argument(
        '--disable-network-degradation',
        '--disable-network-degradation-zone',
        dest='disable_network_degradation',
        action='store_true',
        help=(
            'locally force no network zones or metric penalties, overriding '
            'shared/default zone locations; spatial-map sensor sites remain '
            'virtual and non-streaming'))
    argparser.add_argument(
        '--network-degradation-map-latency-penalty-ms',
        metavar='MS',
        default=DEFAULT_NETWORK_MAP_LATENCY_PENALTY_MS,
        type=nonnegative_finite_float,
        help=(
            'maximum display-only spatial-map latency addition at a zone '
            'centre (default: %(default)s ms)'))
    argparser.add_argument(
        '--network-degradation-map-accuracy-penalty-cm',
        metavar='CM',
        default=DEFAULT_NETWORK_MAP_ACCURACY_PENALTY_CM,
        type=nonnegative_finite_float,
        help=(
            'maximum display-only spatial-map error addition at a zone '
            'centre (default: %(default)s cm)'))
    argparser.add_argument(
        '--network-degradation-sense-to-act-penalty-ms',
        metavar='MS',
        default=DEFAULT_NETWORK_SENSE_TO_ACT_PENALTY_MS,
        type=nonnegative_finite_float,
        help=(
            'maximum display-only sense-to-act latency addition at a zone '
            'centre (default: %(default)s ms)'))
    argparser.add_argument(
        '--infrastructure-sensor-traffic-light-ids',
        metavar='IDS',
        default=DEFAULT_INFRASTRUCTURE_SENSOR_TRAFFIC_LIGHT_IDS,
        type=infrastructure_sensor_traffic_light_ids,
        help=(
            'comma-separated traffic-light IDs receiving a virtual camera/'
            'radar pair using the live root pose or catalog fallback; use "none" '
            '(default: 14,24,11)'))
    argparser.add_argument(
        '--rogue-pedestrian-warning-radius',
        '--rogue-pedestrian-alert-radius',
        dest='rogue_pedestrian_warning_radius',
        metavar='METERS',
        default=DEFAULT_ROGUE_PEDESTRIAN_WARNING_RADIUS_M,
        type=positive_finite_float,
        help=(
            'planar ego radius that changes spawn_blocker_v5 pedestrian '
            'boxes to red and shows the warning card after any active ego/'
            'peer sensor detects the pedestrian while U visuals are on '
            '(default: %(default)s)'))
    argparser.add_argument(
        '--rogue-pedestrian-brake-radius',
        metavar='METERS',
        default=DEFAULT_ROGUE_PEDESTRIAN_BRAKE_RADIUS_M,
        type=positive_finite_float,
        help=(
            'inner planar radius that changes the warning action from SLOW '
            'DOWN to APPLY BRAKES; must not exceed the warning radius '
            '(default: %(default)s)'))
    argparser.add_argument(
        '--rogue-pedestrian-role-prefix',
        metavar='PREFIX',
        default=DEFAULT_ROGUE_PEDESTRIAN_ROLE_PREFIX,
        type=rogue_pedestrian_role_prefix,
        help=(
            'role-name stem identifying blocker walkers as PREFIX_<index> '
            '(default: %(default)s)'))
    argparser.add_argument(
        '--route-config', metavar='PATH', default=None, type=os.path.abspath,
        help=(
            'coordinate-based ego route JSON exported by '
            'physical_ai_scenario_controller_ui_v2.py; when explicit ego '
            'spawn coordinates are absent, the saved start is also used for '
            'startup and Y respawn'))
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
        '--route-sampling-resolution', default=None, type=float,
        help=(
            'waypoint spacing used by the route planner in meters; defaults '
            'to the loaded route config value, or 2.0 without a route config'))
    args = argparser.parse_args()

    try:
        (
            args.cooperative_camera_horizontal_fov,
            args.cooperative_radar_horizontal_fov,
        ) = resolve_cooperative_sensor_fovs(
            args.cooperative_sensor_pair_horizontal_fov,
            args.cooperative_camera_horizontal_fov,
            args.cooperative_radar_horizontal_fov)
    except (TypeError, ValueError) as exc:
        argparser.error(str(exc))

    # Preserve whether the operator supplied a local zone policy before the
    # parser replaces None with the built-in fallback. A shared blocker
    # profile can otherwise update zone locations dynamically.
    args.network_degradation_zones_explicit = (
        args.network_degradation_zones is not None)
    args.network_degradation_disabled_explicit = bool(
        args.disable_network_degradation)
    if args.disable_network_degradation and args.network_degradation_zones:
        argparser.error(
            '--disable-network-degradation cannot be combined with '
            '--network-degradation-zone')
    raw_network_zones = (
        ()
        if args.disable_network_degradation else
        (args.network_degradation_zones
         if args.network_degradation_zones is not None else
         (DEFAULT_NETWORK_DEGRADATION_ZONE,)))
    try:
        args.network_degradation_zones = normalize_network_degradation_zones(
            raw_network_zones)
    except (TypeError, ValueError) as exc:
        argparser.error(str(exc))

    for option_name, value in (
            ('--spatial-map-active-sensor-pairs',
             args.spatial_map_active_sensor_pairs),
            ('--spatial-map-degraded-radar-pairs',
             args.spatial_map_degraded_radar_pairs)):
        if value > MAX_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS:
            argparser.error(
                '{} must not exceed {}'.format(
                    option_name,
                    MAX_SPATIAL_MAP_ACTIVE_SENSOR_PAIRS))

    if (
            args.spatial_map_sensor_forward_range
            > MAX_SPATIAL_MAP_SENSOR_FORWARD_RANGE_M):
        argparser.error(
            '--spatial-map-sensor-forward-range must not exceed {:.1f} '
            'meters'.format(MAX_SPATIAL_MAP_SENSOR_FORWARD_RANGE_M))
    if (
            args.spatial_map_sensor_forward_half_angle
            > MAX_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG):
        argparser.error(
            '--spatial-map-sensor-forward-half-angle must not exceed {:.1f} '
            'degrees'.format(
                MAX_SPATIAL_MAP_SENSOR_FORWARD_HALF_ANGLE_DEG))

    for option_name, value in (
            ('--cooperative-camera-horizontal-fov',
             args.cooperative_camera_horizontal_fov),
            ('--cooperative-radar-horizontal-fov',
             args.cooperative_radar_horizontal_fov)):
        if value > MAX_COOPERATIVE_SENSOR_FOV_DEG:
            argparser.error(
                '{} must not exceed {:.1f} degrees'.format(
                    option_name, MAX_COOPERATIVE_SENSOR_FOV_DEG))
    for option_name, value in (
            ('--cooperative-camera-range',
             args.cooperative_camera_range),
            ('--cooperative-radar-range',
             args.cooperative_radar_range)):
        if value > MAX_COOPERATIVE_SENSOR_RANGE_M:
            argparser.error(
                '{} must not exceed {:.1f} meters'.format(
                    option_name, MAX_COOPERATIVE_SENSOR_RANGE_M))

    if (args.destination_x is None) != (args.destination_y is None):
        argparser.error('--destination-x and --destination-y must be provided together')
    if (args.ego_spawn_x is None) != (args.ego_spawn_y is None):
        argparser.error('--ego-spawn-x and --ego-spawn-y must be provided together')
    args.ego_spawn_was_explicit = args.ego_spawn_x is not None
    if args.ego_spawn_x is None:
        args.ego_spawn_x = DEFAULT_EGO_SPAWN_X
        args.ego_spawn_y = DEFAULT_EGO_SPAWN_Y
    if (
            args.route_sampling_resolution is not None and
            (not math.isfinite(args.route_sampling_resolution) or
             args.route_sampling_resolution <= 0.0)):
        argparser.error('--route-sampling-resolution must be a positive finite number')
    if (
            args.rogue_pedestrian_brake_radius >
            args.rogue_pedestrian_warning_radius):
        argparser.error(
            '--rogue-pedestrian-brake-radius must not exceed '
            '--rogue-pedestrian-warning-radius')

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    legacy_managed_options = []
    if args.legacy_disable_managed_sensor_pairs:
        legacy_managed_options.append('--disable-managed-sensor-pairs')
    if args.legacy_managed_sensor_pairs_per_category is not None:
        legacy_managed_options.append('--managed-sensor-pairs-per-category')
    if args.legacy_managed_sensor_degraded_camera_pairs is not None:
        legacy_managed_options.append('--managed-sensor-degraded-camera-pairs')
    if args.legacy_managed_sensor_refresh_seconds is not None:
        legacy_managed_options.append('--managed-sensor-refresh-seconds')
    if legacy_managed_options:
        logging.info(
            'Accepted retired scene-wide sensor option(s) as no-ops: %s. '
            'All spatial-map RGB/radar sites are virtual; no map-only sensor '
            'actors are attached.',
            ', '.join(legacy_managed_options))

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
