#!/usr/bin/env python3

"""
Spawn static/reactive vehicle and reactive pedestrian blockers for CARLA.

This client is designed to run beside manual_control_ar_v8.py and/or
pedestrian_head_camera_client_v8.py/v9.py. It remains a passive world client: it
never loads a map, changes world settings, changes the
Traffic Manager, or calls world.tick(). In asynchronous mode the CARLA server
advances the simulation itself; only an already-synchronous world needs a
separate clock owner.

With no blocker-location arguments, the script spawns the three original
captured vehicle blockers, six dedicated static Nissan Patrol blockers, and
two captured pedestrian blockers in Town10HD. Vehicle #3 is a
repeatable crosswalk hazard: it waits near its captured home at the resolved
Driving waypoint, watches
for the uniquely identified ``manual_pedestrian`` from
``pedestrian_head_camera_client_v8.py`` or ``v9.py`` to cross the nearby road,
and then
follows CARLA Driving waypoints through that lane. The captured crosswalk XY
is an anchor, not a mandatory point: the script intersects the pedestrian's
live motion ray with a bounded section of the road route and latches that
route-relative collision station. At startup
the captured shoulder pose is projected to its nearest
Driving waypoint and GlobalRoutePlanner builds the road route once. Steering,
vehicle-front progress, collision timing, and post-event runout all use route
arc distance; the vehicle never cuts a straight chord across the sidewalk.
Its speed is retimed from the pedestrian's current position, velocity, and
bounded acceleration, with a curve-derived speed cap. The vehicle remains
frozen at its road home until the live pedestrian ETA enters a configurable
launch window. If that ETA grows or disappears before the final committed
approach, the vehicle brakes normally and holds upstream of the contact plane
instead of racing through the crossing on a stale estimate. If the pedestrian
ETA is already shorter than the vehicle's physical minimum travel time, the
default best-effort mode still launches at the route-safe limit and logs that
contact cannot be guaranteed. After a miss it continues along the lane
beyond the crosswalk, is destroyed, and is freshly spawned at
the resolved road waypoint. A confirmed collision is handled differently: the
script releases scripted drive for a short physics-settling interval, brakes
normally without zeroing velocity, and leaves the vehicle visible at its
resulting collision pose before respawning. Collision-free misses retain the
lane-following drive-through. Vehicles #1, #2, and #4 through #9 remain static.
Repeating --vehicle-location or --pedestrian-location replaces that category's
baked-in locations and supports any number of blockers. --no-vehicle-blockers
and --no-pedestrian-blockers disable a category explicitly.

This script deliberately does not spawn physical RGB-camera or radar actors on
its vehicles, pedestrians, or traffic-light poles. Spatial-map clients derive
co-located camera/radar display markers virtually from the stable blocker role
names and live traffic-light roots. The virtual markers therefore add no CARLA
sensor actors, render targets, callbacks, or sensor-stream load. The existing
``sensor.other.collision`` actors are separate and remain subscribed because
they drive encounter state.

The former traffic-light sensor mount and route command-line options remain
accepted as deprecated no-ops so existing launch commands do not break. Route
files are no longer loaded and traffic-light roots are no longer discovered or
instrumented by this client.

This client also publishes two independently configurable circular cellular
network-degradation zones into the current CARLA world.  The first defaults to
the ego-vehicle/occluded-pedestrian encounter and the second defaults to the
ego-pedestrian/rogue-vehicle encounter.  Invisible, parentless GNSS metadata
actors carry the profile so every passive display client connected to the same
world sees one consistent configuration.  Zone actors are published first and
a versioned manifest is published last; teardown removes the manifest first so
clients never accept a partial profile.  Each zone can be disabled separately,
or ``--disable-network-degradation`` disables both. A complete pre-existing
profile with the exact requested settings is safely reused rather than treated
as an ownership conflict. A stable manifest-free residue that exactly matches
this launch's requested zones is recovered automatically after several passive
world snapshots. Use ``--replace-existing-network-profile`` only to replace an
incompatible or otherwise ambiguous profile known to be stale.

``--start-active-spatial-map-sensors`` is also retained only for command-line
compatibility. It is ignored and the shared profile always publishes spatial
sensor streaming as disabled because scenario sensor markers are virtual and
have no CARLA actor IDs to subscribe.

Pedestrians may also use a ground-projected location captured from CARLA's
free-floating spectator camera. If direct spawning fails while the requested
target remains clear, the script tries a nearby Sidewalk waypoint and then the
nearest sampled navigation locations before relocating the walker to the
requested transform. Occupied initial targets are skipped, and occupied
respawn targets are retried on later passive snapshots.
If the preferred pedestrian blueprint is unavailable, the script logs the
problem and selects the first available walker.pedestrian.* ID alphabetically.

The ego vehicle is discovered by role_name (default: hero, matching
manual_control_ar_v8.py). A waiting pedestrian activates only when the ego is
moving toward it and an acceleration-aware, perpendicular intercept lies inside
the configured reaction-plus-hard-braking stopping-distance envelope. The
constant-acceleration prediction uses the ego's current location, velocity, and
acceleration. By default it selects L2 where the pedestrian line is
perpendicular to the predicted trajectory of the ego vehicle's *front contact
reference*, not its actor origin. That reference includes the vehicle's actual
oriented bounding-box nose, the pedestrian radius, and a small configurable
lead margin, so the walker reaches L2 before the front bumper instead of being
hit late along the vehicle's side. ``--impact-target center`` restores the old
center-target timing for comparison. The configured ``--pedestrian-speed`` is
the maximum allowed physical speed.

After activation the straight pedestrian crossing line stays fixed. Every
update uses fresh ego velocity and acceleration to predict when the same front
contact reference will next intersect that line and retimes the pedestrian
without turning it into a moving-target chase. If steering changes the
predicted tangent beyond
``--active-perpendicular-tolerance``, the encounter is recycled instead of
claiming a synchronized perpendicular collision that is no longer feasible.
It is likewise recycled if fresh prediction requires more than the configured
maximum pedestrian speed or moves L2 behind the committed walker.
Short-lived CARLA debug lines show the predicted ego front-contact trajectory,
perpendicular pedestrian path, and current L2 by default; use
``--no-intercept-debug`` to hide them. Root velocity is used because this CARLA
0.10 build caps direct WalkerControl motion near normal walking speed;
WalkerControl remains active only for the running animation. At activation the
walker is explicitly faced along the crossing path. A simulation-time progress
watchdog detects stalled motion and, after normal controls have been reasserted,
can enter a bounded scripted fallback. Each proposed step is swept against
vertically compatible vehicle footprints and stops short of them so CARLA
handles the final physical contact. Set ``--stall-recovery-step 0`` to disable
that fallback and recycle a stalled walker instead.
This is a CARLA-only test model; steering, braking, road geometry, or unsuitable
staging can still prevent a collision.

After any vehicle hits a blocker, or after a qualified near miss at L2, the
pedestrian is stopped and remains visible for five seconds of simulation time.
Its sensor and walker are then destroyed and a fresh blocker is spawned at the
original requested transform. Respawning is deferred to a later passive world
snapshot so collision actors can clear. CARLA bounding volumes normally touch
before actor centers become identical, so a collided walker is held at the
physical contact location rather than being teleported into the vehicle; a
near-miss walker is held at L2.

Version 5 retains v4's dynamic interception and recurring respawn model while
adding the independently managed vehicle #3 crosswalk encounter described
above. The current implementation lane-snaps its home, plans the shortest
directed road route to the pedestrian's Driving lane, follows it with
lookahead Ackermann steering, rejects excessive route deviation, and derives a
curve-safe speed limit from the selected lateral-acceleration budget. The
vehicle uses CARLA's normal Ackermann controller by default so rigid-body
collision response remains active; an explicitly selectable constant-velocity
mode is available for deterministic scripted experiments but is not intended
for physical-impact rendering. Target contact takes priority over simultaneous
prop/road events. Exact and swept geometry guard delayed sensor callbacks, and
collision actors are retained through both simulation-clock and wall-clock
minimum settle/hold intervals. The feature never uses the Traffic Manager, never
advances the world clock, and does not teleport a live collision actor back to
its home pose.

Version 4 retains v3's dynamic interception and recurring respawn model while
adapting the reusable resilience mechanisms from scenesense_scenario_harness.py:
activation yaw alignment, tight endpoint completion, repeated directional
commands, and a deterministic fallback after verified motion stalls. It also
adds vertically gated projected-bounding-box checks, occupied-target protection,
partial-command rollback, and CARLA clock-rewind recovery. This revision adds
acceleration-aware perpendicular interception, active ETA retiming, near-miss
classification, debug geometry, a visible post-event hold state, and
bounding-box-aware front-impact targeting.

Examples:
    # Spawn the nine baked-in vehicles and two pedestrians.
    python3 spawn_blocker_v5.py

    # Legacy sensor-mount arguments remain accepted but are ignored.
    python3 spawn_blocker_v5.py \
        --traffic-light-sensor-yaw 15 \
        --traffic-light-sensor-pitch -45

    # Replace only the pedestrian defaults; keep the nine default vehicles.
    python3 spawn_blocker_v5.py \\
        --pedestrian-blueprint walker.pedestrian.0015 \\
        --pedestrian-location 70.0 61.0 0.2 -90 \\
        --ego-role-name hero

    # Spawn only one spectator-derived pedestrian.
    python3 spawn_blocker_v5.py \\
        --no-vehicle-blockers --from-spectator \\
        --ego-role-name manual_ar_ego

    # Keep the vehicles, disable blocker pedestrians, and tune the crosswalk run.
    python3 spawn_blocker_v5.py --no-pedestrian-blockers \\
        --reactive-vehicle-speed-max 12 --reactive-post-distance 10
"""

import argparse
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import carla

from network_degradation_profile_v1 import (
    NETWORK_PROFILE_BLUEPRINT_ID,
    NETWORK_PROFILE_MANIFEST_PREFIX,
    NETWORK_PROFILE_PEDESTRIAN_ZONE_INDEX,
    NETWORK_PROFILE_ROLE_PREFIX,
    NETWORK_PROFILE_SENSOR_TICK_SECONDS,
    NETWORK_PROFILE_VEHICLE_ZONE_INDEX,
    NetworkDegradationZone,
    NetworkProfileError,
    build_manifest_role,
    build_zone_role,
    discover_network_degradation_profile,
    find_profile_actor_conflicts,
    new_session_token,
    normalize_zone,
    parse_manifest_role,
    parse_zone_role,
)
CARLA_AGENT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "carla")
)
if CARLA_AGENT_PATH not in sys.path:
    sys.path.append(CARLA_AGENT_PATH)

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    GlobalRoutePlanner = None


LOG = logging.getLogger("spawn_mixed_blockers")

DEFAULT_VEHICLE_BLUEPRINT = "vehicle.fuso.mitsubishi"
DEFAULT_ADDITIONAL_PATROL_BLUEPRINT = "vehicle.nissan.patrol"
DEFAULT_ADDITIONAL_PATROL_INDEX = 4
DEFAULT_CAPTURED_PATROL_INDICES = (5, 6, 7, 8, 9)
DEFAULT_STATIC_PATROL_INDICES = (
    DEFAULT_ADDITIONAL_PATROL_INDEX,
) + DEFAULT_CAPTURED_PATROL_INDICES
DEFAULT_PEDESTRIAN_BLUEPRINT = "walker.pedestrian.0015"
DEFAULT_EGO_ROLE_NAME = "hero"
DEFAULT_EGO_PEDESTRIAN_ROLE_NAME = "manual_pedestrian"
DEFAULT_CLIENT_TIMEOUT_SECONDS = 10.0
DEFAULT_VEHICLE_Z_OFFSET_M = -0.07
DEFAULT_PEDESTRIAN_Z_OFFSET_M = 0.5
DEFAULT_NAVIGATION_SAMPLES = 250
DEFAULT_NAVIGATION_SEARCH_RADIUS_M = 30.0
DEFAULT_PLACEMENT_TOLERANCE_M = 0.75
DEFAULT_SPAWN_OCCUPANCY_CLEARANCE_M = 0.35
DEFAULT_SPAWN_OCCUPANCY_VERTICAL_CLEARANCE_M = 1.0
DEFAULT_SPECTATOR_GROUND_SEARCH_M = 100.0
DEFAULT_UPDATE_HZ = 20.0
DEFAULT_TICK_TIMEOUT_SECONDS = 2.0

# CARLA may not expose a newly spawned metadata sensor through get_actors()
# until a later server snapshot.  These bounded waits remain passive: this
# client never advances the clock or changes WorldSettings.  In asynchronous
# mode each wait normally returns on the next simulation frame; in synchronous
# mode the existing clock owner must provide the snapshots.
NETWORK_PROFILE_REGISTRY_RETRY_ATTEMPTS = 5
NETWORK_PROFILE_PASSIVE_WAIT_TIMEOUT_SECONDS = 0.5
NETWORK_PROFILE_STALE_CONFIRMATION_TICKS = 3
NETWORK_PROFILE_STALE_MIN_STABLE_SECONDS = 1.5
NETWORK_PROFILE_STALE_MAX_OBSERVATION_SECONDS = 4.0
NETWORK_PROFILE_STALE_MAX_SNAPSHOT_OBSERVATIONS = 200

DEFAULT_PEDESTRIAN_SPEED_MPS = 8.0
DEFAULT_MIN_PEDESTRIAN_SPEED_MPS = 0.5
DEFAULT_IMPACT_TARGET = "front"
# Schedule the walker this far ahead of first mathematical bounding-box
# contact. At the default 20 Hz update rate, 0.5 m compensates roughly one
# command/physics frame for a road vehicle moving near 10 m/s.
DEFAULT_FRONT_IMPACT_MARGIN_M = 0.5
# This shipping CARLA 0.10 build caps direct WalkerControl motion near 2 m/s.
# Active blockers therefore use set_target_velocity() for their physical speed
# and WalkerControl only for animation. Preserve this multiplier as an
# animation-only compatibility knob for existing command lines.
DEFAULT_WALKER_CONTROL_SPEED_SCALE = 1.0
DEFAULT_WALKER_ANIMATION_SPEED_CAP_MPS = 5.5
DEFAULT_MIN_EGO_SPEED_MPS = 2.0
DEFAULT_MIN_CLOSING_SPEED_MPS = 1.0
DEFAULT_TRIGGER_DISTANCE_M = 45.0
DEFAULT_MAX_APPROACH_ANGLE_DEG = 70.0
DEFAULT_MAX_LATERAL_OFFSET_M = 10.0
DEFAULT_REACTION_TIME_SECONDS = 0.7
DEFAULT_MAX_BRAKE_DECELERATION_MPS2 = 8.0
DEFAULT_BRAKING_MARGIN_M = 0.5
DEFAULT_MAX_INTERCEPT_TIME_SECONDS = 6.0
DEFAULT_MIN_INTERCEPT_TIME_SECONDS = 0.25
DEFAULT_PREDICTION_ACCELERATION_LIMIT_MPS2 = 10.0
DEFAULT_ACCELERATION_SMOOTHING = 0.35
DEFAULT_ACTIVE_PERPENDICULAR_TOLERANCE_DEG = 5.0
DEFAULT_ACTIVE_TIMEOUT_SECONDS = 20.0
DEFAULT_EXPIRE_DISTANCE_M = 4.0
DEFAULT_COLLISION_DISTANCE_M = 1.2
DEFAULT_MAX_ACTIVE_PEDESTRIANS = 1
DEFAULT_CROSSING_EXTRA_DISTANCE_M = 1.0
DEFAULT_INTERCEPT_ARRIVAL_TOLERANCE_M = 0.15
DEFAULT_NEAR_MISS_DISTANCE_M = 3.0
DEFAULT_HARD_BRAKE_DECELERATION_MPS2 = 3.5
DEFAULT_STOPPED_EGO_SPEED_MPS = 0.5
DEFAULT_POST_EVENT_HOLD_SECONDS = 5.0
DEFAULT_DEBUG_DRAW_INTERVAL_SECONDS = 0.2
DEFAULT_RESPAWN_DELAY_SECONDS = 0.5
DEFAULT_RESPAWN_RETRY_INTERVAL_SECONDS = 1.0
DEFAULT_RESPAWN_CLEARANCE_M = 3.0
DEFAULT_RESPAWN_VERTICAL_CLEARANCE_M = 2.5
DEFAULT_MOTION_STALL_TIMEOUT_SECONDS = 0.75
DEFAULT_MOTION_STALL_MIN_PROGRESS_M = 0.25
DEFAULT_STALL_RECOVERY_STEP_M = 0.35
DEFAULT_MAX_MOTION_COMMAND_FAILURES = 3
DEFAULT_MAX_SCRIPTED_RECOVERY_SECONDS = 15.0
DEFAULT_COLLISION_VERTICAL_CLEARANCE_M = 0.25
DEFAULT_COLLISION_CENTER_VERTICAL_TOLERANCE_M = 2.0

# Vehicle #3 / ego-pedestrian crosswalk encounter. The target XY was captured
# from pedestrian_head_camera_client_v8.py while its manual_pedestrian actor
# stood at the desired impact point. Z is intentionally omitted: attack timing
# and progress are defined in the CARLA world XY plane.
DEFAULT_REACTIVE_VEHICLE_INDEX = 3
DEFAULT_REACTIVE_ATTACK_LOCATION = (
    101.51990509033203,
    38.707523345947266,
)
# The road-following path is longer than the old shoulder-to-target chord and
# its tight right turn needs a curve-safe speed. Thirty metres lets both the
# v8 walk and Shift-run controls be detected early enough for a feasible run;
# ETA/cross-track checks still reject unrelated trajectories.
DEFAULT_REACTIVE_TRIGGER_DISTANCE_M = 30.0
DEFAULT_REACTIVE_REARM_DISTANCE_M = 6.0
DEFAULT_REACTIVE_MIN_PEDESTRIAN_SPEED_MPS = 0.20
DEFAULT_REACTIVE_MIN_CLOSING_SPEED_MPS = 0.15
DEFAULT_REACTIVE_MAX_CROSS_TRACK_M = 1.5
DEFAULT_REACTIVE_MIN_PEDESTRIAN_ETA_SECONDS = 0.50
DEFAULT_REACTIVE_MAX_PEDESTRIAN_ETA_SECONDS = 8.0
DEFAULT_REACTIVE_PEDESTRIAN_ACCELERATION_LIMIT_MPS2 = 6.0
DEFAULT_REACTIVE_VEHICLE_SPEED_MIN_MPS = 1.0
DEFAULT_REACTIVE_VEHICLE_SPEED_MAX_MPS = 10.0
DEFAULT_REACTIVE_VEHICLE_MAX_ACCELERATION_MPS2 = 5.0
DEFAULT_REACTIVE_VEHICLE_MAX_DECELERATION_MPS2 = 8.0
DEFAULT_REACTIVE_IMPACT_LEAD_SECONDS = 0.10
# The actor stays frozen at its road home until the pedestrian ETA is no more
# than this margin above the optimistic curve-safe vehicle travel time.  This
# avoids launching several seconds early for a slow or paused manual walker.
DEFAULT_REACTIVE_LAUNCH_MARGIN_SECONDS = 0.50
# Before the final timing commitment, a vehicle whose target ETA moves later
# brakes toward a front-bumper hold line this far upstream of first contact.
DEFAULT_REACTIVE_APPROACH_HOLD_DISTANCE_M = 1.50
DEFAULT_REACTIVE_APPROACH_HOLD_TIMEOUT_SECONDS = 30.0
DEFAULT_REACTIVE_COMMIT_DISTANCE_M = 5.0
DEFAULT_REACTIVE_COMMIT_SPEED_MPS = 5.0
DEFAULT_REACTIVE_RUNOUT_SPEED_MPS = 6.0
DEFAULT_REACTIVE_POST_DISTANCE_M = 10.0
DEFAULT_REACTIVE_ACTIVE_TIMEOUT_SECONDS = 12.0
DEFAULT_REACTIVE_RUNOUT_TIMEOUT_SECONDS = 6.0
DEFAULT_REACTIVE_TARGET_PASS_DISTANCE_M = 0.75
# Keep the impacted vehicle physical for several rendered frames before
# braking, then leave it visible at the resulting contact pose. Destroying it
# in the collision-processing tick suppresses CARLA's vehicle/walker response.
DEFAULT_REACTIVE_IMPACT_SETTLE_SECONDS = 0.25
DEFAULT_REACTIVE_CONTACT_HOLD_SECONDS = 5.0
DEFAULT_REACTIVE_CONTACT_STOP_SPEED_MPS = 0.25
DEFAULT_REACTIVE_RESPAWN_DELAY_SECONDS = 0.5
DEFAULT_REACTIVE_RESPAWN_RETRY_SECONDS = 1.0
DEFAULT_REACTIVE_RESPAWN_CLEARANCE_M = 3.0
DEFAULT_REACTIVE_ACTIVATION_Z_LIFT_M = 0.0
DEFAULT_REACTIVE_ROUTE_SAMPLING_RESOLUTION_M = 0.5
DEFAULT_REACTIVE_ROUTE_LOOKAHEAD_M = 5.0
# reactive_vehicle_steer() adds up to this much speed-dependent lookahead.
# Reserve and curvature-check the same distance beyond the entire runout so
# steering never reaches beyond the lane route or enters an uncapped bend.
REACTIVE_ROUTE_DYNAMIC_LOOKAHEAD_MAX_M = 3.0
DEFAULT_REACTIVE_ROUTE_HOME_PROJECTION_LIMIT_M = 6.0
DEFAULT_REACTIVE_ROUTE_ATTACK_OFFSET_LIMIT_M = 2.0
DEFAULT_REACTIVE_ROUTE_DEVIATION_LIMIT_M = 2.5
DEFAULT_REACTIVE_ROUTE_MAX_LENGTH_M = 250.0
DEFAULT_REACTIVE_ROUTE_MAX_LATERAL_ACCELERATION_MPS2 = 3.0
DEFAULT_REACTIVE_STEER_LIMIT_RADIANS = 0.65
# The nominal crosswalk point is only an anchor.  A live pedestrian may cross
# the same road several metres before or after it, so activation searches this
# much route arc on either side for the pedestrian motion ray intersection.
DEFAULT_REACTIVE_INTERCEPT_ROUTE_WINDOW_M = 8.0
DEFAULT_REACTIVE_MIN_CROSSING_ANGLE_DEGREES = 20.0
DEFAULT_REACTIVE_BEST_EFFORT_LAUNCH = True
# Collision callbacks are asynchronous.  Retain enough pose history to detect
# a fast walker sweeping through the vehicle between two 20 Hz updates, and
# tolerate a few post-impact geometry/control failures while that callback is
# delivered instead of immediately destroying the vehicle.
DEFAULT_REACTIVE_CONTACT_SWEEP_MAX_INTERVAL_SECONDS = 0.25
DEFAULT_REACTIVE_TRANSIENT_FAILURE_LIMIT = 3

# Ground targets captured from the Town10HD_Opt spectator. Vehicle #3 was
# recaptured at CARLA frame 444991. In the normal configuration it is moved to
# its Driving-lane home, leaving the captured broad-shoulder pose available for
# static Patrol #4, aligned to road 21's yaw. Do not move #4 outward across the
# curb: Town10HD's sidewalk is elevated there and rejects the inherited
# road-level spawn Z. When #3 is configured to remain static (or another
# vehicle index is selected as the rogue), resolve_vehicle_targets moves #4
# farther along the same shoulder so their bounding boxes remain separate.
# Static Patrols #5 through #9 use spectator captures 1 through 5. Captures 1
# and 2 were only 0.596 m apart, so their full-size Patrol footprints cannot
# coexist. #5 preserves capture 1 exactly. #6 preserves capture 2's yaw and
# roadside lateral offset but moves 6.0 m backward along road 12's tangent;
# this produces zero road-normal shift and less than 0.008 m of world-Y change.
# A live two-vehicle CARLA probe verified the adjusted pose while #5 occupied
# capture 1. #7 through #9 use captures 3 through 5 exactly. Stored Z values
# are ground projections.
# Category-specific Z offsets are applied once when CARLA transforms are built.
DEFAULT_VEHICLE_LOCATIONS: Tuple[Tuple[float, float, float, float], ...] = (
    (
        56.35101318359375,
        62.8189811706543,
        -0.007947438396513462,
        173.5841064453125,
    ),
    (
        47.52729797363281,
        40.43889617919922,
        0.03450716286897659,
        -91.84769439697266,
    ),
    (
        82.12229919433594,
        31.94034194946289,
        0.03722051903605461,
        4.73963737487793,
    ),
    (
        82.12229919433594,
        31.94034194946289,
        0.03722051903605461,
        0.15919755399227142,
    ),
    (
        -3.67894268,
        73.425430298,
        0.054500684,
        1.762827516,
    ),
    (
        -10.206179957673232,
        73.1395284404447,
        0.021087050437927246,
        7.70363712310791,
    ),
    (
        21.83455467224121,
        73.26212310791016,
        0.04297012835741043,
        4.923678874969482,
    ),
    (
        57.394447326660156,
        31.972843170166016,
        0.022511951625347137,
        5.017217636108398,
    ),
    (
        38.66566848754883,
        9.568473815917969,
        0.029275402426719666,
        178.70921325683594,
    ),
)

# Lane-centered road-21 shoulder pose used only when vehicle #3 is not the
# active lane-snapped rogue. It keeps the otherwise coincident #3/#4 static
# actors physically separate without placing #4 on the sidewalk.
DEFAULT_ADDITIONAL_PATROL_CLEARANCE_LOCATION: Tuple[
    float,
    float,
    float,
    float,
] = (
    73.5000228881836,
    31.388269424438477,
    0.0,
    0.15919755399227142,
)

DEFAULT_PEDESTRIAN_LOCATIONS: Tuple[Tuple[float, float, float, float], ...] = (
    (
        19.791866302490234,
        32.016666412353516,
        0.00566829415038228,
        -89.58074188232422,
    ),
    (
        8.827261924743652,
        62.21647644042969,
        0.02347649447619915,
        90.51849365234375,
    ),
)

VEHICLE_ROLE_PREFIX = "static_blocker_v5"
REACTIVE_VEHICLE_ROLE_PREFIX = "reactive_blocker_v5"
PEDESTRIAN_ROLE_PREFIX = "pedestrian_blocker_v5"
INACTIVE_FRONT_SENSOR_ROLE_PREFIX = "blocker_v5_inactive"
INACTIVE_FRONT_SENSOR_TICK_SECONDS = 1.0 / DEFAULT_UPDATE_HZ
INACTIVE_FRONT_SENSOR_MARGIN_M = 0.05
INACTIVE_CAMERA_IMAGE_WIDTH = 320
INACTIVE_CAMERA_IMAGE_HEIGHT = 180
INACTIVE_RADAR_POINTS_PER_SECOND = 1500
DEFAULT_EGO_VEHICLE_NETWORK_DEGRADATION_ZONE = (
    8.8272619247,
    62.2164764404,
    18.0,
)
DEFAULT_EGO_PEDESTRIAN_NETWORK_DEGRADATION_ZONE = (
    90.3899993896,
    43.2900009155,
    18.0,
)
DEFAULT_TRAFFIC_LIGHT_SENSOR_HEIGHT_M = 15.0
DEFAULT_TRAFFIC_LIGHT_SENSOR_YAW_OFFSET_DEG = 0.0
DEFAULT_TRAFFIC_LIGHT_SENSOR_PITCH_DEG = -35.0
DEFAULT_TRAFFIC_LIGHT_SENSOR_ROUTE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "recorded_pedestrian_route_1.json",
)
DEFAULT_TRAFFIC_LIGHT_SENSOR_ROUTE_RADIUS_M = 12.0
DEFAULT_MAX_TRAFFIC_LIGHT_SENSOR_POLES = 6
MAX_TRAFFIC_LIGHT_SENSOR_POLES = 16
MAX_TRAFFIC_LIGHT_SENSOR_ROUTE_RADIUS_M = 1000.0
TRAFFIC_LIGHT_ROUTE_DEDUPE_DISTANCE_M = 0.01
TRAFFIC_LIGHT_ROUTE_CONTROL_DEDUPE_DISTANCE_M = 0.05
TRAFFIC_LIGHT_ROUTE_ENDPOINT_TOLERANCE_M = 2.0
TRAFFIC_LIGHT_ROUTE_CONTROL_TOLERANCE_M = 2.0
TRAFFIC_LIGHT_ROUTE_VERTICAL_TOLERANCE_M = 0.5
TRAFFIC_LIGHT_ROUTE_DISTANCE_EPSILON_M = 1.0e-6
MIN_TRAFFIC_LIGHT_SENSOR_PITCH_DEG = -89.0
MAX_TRAFFIC_LIGHT_SENSOR_PITCH_DEG = -0.1
MAX_TRAFFIC_LIGHT_SENSOR_HEIGHT_M = 1000.0
TRAFFIC_LIGHT_GROUND_Z_TOLERANCE_M = 5.0
STATE_WAITING = "WAITING"
STATE_ACTIVE = "ACTIVE"
STATE_HOLDING = "HOLDING"
STATE_RESPAWN_PENDING = "RESPAWN_PENDING"
REACTIVE_STATE_WAITING = "WAITING"
REACTIVE_STATE_ALIGNING = "ALIGNING"
REACTIVE_STATE_ACTIVE = "ACTIVE"
REACTIVE_STATE_RUNOUT = "RUNOUT"
REACTIVE_STATE_CONTACT_SETTLING = "CONTACT_SETTLING"
REACTIVE_STATE_CONTACT_HOLDING = "CONTACT_HOLDING"
REACTIVE_STATE_RESPAWN_PENDING = "RESPAWN_PENDING"


def finite_float(value: str) -> float:
    """Parse a finite floating-point command-line value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def positive_float(value: str) -> float:
    parsed = finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = finite_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="HOST",
        help="CARLA server host (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=2000,
        type=int,
        metavar="PORT",
        help="CARLA RPC port (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        default=DEFAULT_CLIENT_TIMEOUT_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="CARLA client timeout (default: %(default)s seconds)",
    )
    network_profile = parser.add_argument_group(
        "shared network-degradation profile"
    )
    ego_vehicle_zone = network_profile.add_mutually_exclusive_group()
    ego_vehicle_zone.add_argument(
        "--ego-vehicle-network-degradation-zone",
        "--vehicle-network-degradation-zone",
        dest="ego_vehicle_network_degradation_zone",
        nargs=3,
        default=None,
        type=finite_float,
        metavar=("X", "Y", "RADIUS"),
        help=(
            "replace the ego-vehicle cellular-degradation zone in CARLA "
            "metres (default: {:.3f} {:.3f} {:.1f})".format(
                *DEFAULT_EGO_VEHICLE_NETWORK_DEGRADATION_ZONE
            )
        ),
    )
    ego_vehicle_zone.add_argument(
        "--disable-ego-vehicle-network-degradation-zone",
        "--disable-vehicle-network-degradation-zone",
        dest="disable_ego_vehicle_network_degradation_zone",
        action="store_true",
        help="disable only the ego-vehicle network-degradation zone",
    )
    ego_pedestrian_zone = network_profile.add_mutually_exclusive_group()
    ego_pedestrian_zone.add_argument(
        "--ego-pedestrian-network-degradation-zone",
        "--pedestrian-network-degradation-zone",
        dest="ego_pedestrian_network_degradation_zone",
        nargs=3,
        default=None,
        type=finite_float,
        metavar=("X", "Y", "RADIUS"),
        help=(
            "replace the ego-pedestrian cellular-degradation zone in CARLA "
            "metres (default: {:.3f} {:.3f} {:.1f})".format(
                *DEFAULT_EGO_PEDESTRIAN_NETWORK_DEGRADATION_ZONE
            )
        ),
    )
    ego_pedestrian_zone.add_argument(
        "--disable-ego-pedestrian-network-degradation-zone",
        "--disable-pedestrian-network-degradation-zone",
        dest="disable_ego_pedestrian_network_degradation_zone",
        action="store_true",
        help="disable only the ego-pedestrian network-degradation zone",
    )
    network_profile.add_argument(
        "--disable-network-degradation",
        action="store_true",
        help="disable both published network-degradation zones",
    )
    network_profile.add_argument(
        "--start-active-spatial-map-sensors",
        action="store_true",
        help=(
            "deprecated no-op retained for launch compatibility; scenario "
            "camera/radar markers are virtual and streaming always publishes "
            "disabled"
        ),
    )
    network_profile.add_argument(
        "--replace-existing-network-profile",
        action="store_true",
        help=(
            "destroy only pre-existing sb5 network-profile metadata before "
            "publishing the requested profile; use only when those actors "
            "are known to be stale (compatible profiles are reused by default)"
        ),
    )
    traffic_light_sensors = parser.add_argument_group(
        "deprecated traffic-light physical-sensor options (accepted no-ops)"
    )
    traffic_light_sensors.add_argument(
        "--traffic-light-sensor-height",
        default=DEFAULT_TRAFFIC_LIGHT_SENSOR_HEIGHT_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "deprecated no-op; retained for launch compatibility "
            "(former default: %(default)s m)"
        ),
    )
    traffic_light_sensors.add_argument(
        "--traffic-light-sensor-yaw",
        "--traffic-light-sensor-yaw-offset",
        dest="traffic_light_sensor_yaw",
        default=DEFAULT_TRAFFIC_LIGHT_SENSOR_YAW_OFFSET_DEG,
        type=finite_float,
        metavar="DEGREES",
        help=(
            "deprecated no-op; retained for launch compatibility "
            "(former default: %(default)s degrees)"
        ),
    )
    traffic_light_sensors.add_argument(
        "--traffic-light-sensor-pitch",
        default=DEFAULT_TRAFFIC_LIGHT_SENSOR_PITCH_DEG,
        type=finite_float,
        metavar="DEGREES",
        help=(
            "deprecated no-op; retained for launch compatibility "
            "(former default: %(default)s degrees)"
        ),
    )
    traffic_light_sensors.add_argument(
        "--traffic-light-sensor-route",
        "--traffic-light-route-config",
        "--pedestrian-route-config",
        "--replay-route",
        dest="traffic_light_sensor_route",
        default=DEFAULT_TRAFFIC_LIGHT_SENSOR_ROUTE_PATH,
        metavar="PATH",
        help=(
            "deprecated no-op; route files are no longer loaded by this "
            "script (former default: recorded_pedestrian_route_1.json)"
        ),
    )
    traffic_light_sensors.add_argument(
        "--traffic-light-sensor-route-radius",
        "--traffic-light-route-proximity-radius",
        dest="traffic_light_sensor_route_radius",
        default=DEFAULT_TRAFFIC_LIGHT_SENSOR_ROUTE_RADIUS_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "deprecated no-op retained for launch compatibility "
            "(former default: %(default)s m)"
        ),
    )
    traffic_light_sensors.add_argument(
        "--max-traffic-light-sensor-poles",
        "--traffic-light-sensor-max-poles",
        dest="max_traffic_light_sensor_poles",
        default=DEFAULT_MAX_TRAFFIC_LIGHT_SENSOR_POLES,
        type=positive_int,
        metavar="COUNT",
        help=(
            "deprecated no-op retained for launch compatibility "
            "(former default: %(default)s; former hard maximum: {})".format(
                MAX_TRAFFIC_LIGHT_SENSOR_POLES
            )
        ),
    )
    traffic_light_sensors.add_argument(
        "--no-traffic-light-sensors",
        action="store_true",
        help=(
            "deprecated no-op; physical traffic-light RGB/radar deployment "
            "is always disabled"
        ),
    )
    parser.add_argument(
        "--pedestrian-blueprint",
        "--blueprint",
        "--pedestrian-type",
        dest="blueprint",
        default=DEFAULT_PEDESTRIAN_BLUEPRINT,
        metavar="TYPE_ID",
        help=(
            "preferred walker.pedestrian.* blueprint; unavailable IDs use the "
            "first installed pedestrian ID (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--vehicle-blueprint",
        "--vehicle-type",
        dest="vehicle_blueprint",
        default=DEFAULT_VEHICLE_BLUEPRINT,
        metavar="TYPE_ID",
        help=(
            "exact vehicle.* blueprint for reactive, command-line, and "
            "built-in vehicle targets without a static per-target type; "
            "default static blockers #4-#9 remain vehicle.nissan.patrol "
            "(default for all other targets: %(default)s)"
        ),
    )
    pedestrian_placement = parser.add_mutually_exclusive_group()
    pedestrian_placement.add_argument(
        "--location",
        "--pedestrian-location",
        dest="pedestrian_locations",
        action="append",
        nargs=4,
        type=finite_float,
        metavar=("X", "Y", "Z", "YAW"),
        help=(
            "replace the default pedestrian locations; repeat for any number "
            "of pedestrians"
        ),
    )
    pedestrian_placement.add_argument(
        "--from-spectator",
        action="store_true",
        help=(
            "place one pedestrian below the current free-floating spectator "
            "camera, using a ground projection or nearest navigation sample"
        ),
    )
    pedestrian_placement.add_argument(
        "--no-pedestrian-blockers",
        "--no-pedestrians",
        dest="no_pedestrian_blockers",
        action="store_true",
        help="do not spawn pedestrian blockers",
    )
    vehicle_placement = parser.add_mutually_exclusive_group()
    vehicle_placement.add_argument(
        "--vehicle-location",
        "--blocker-vehicle-location",
        dest="vehicle_locations",
        action="append",
        nargs=4,
        type=finite_float,
        metavar=("X", "Y", "Z", "YAW"),
        help=(
            "replace the default vehicle locations; repeat for any number "
            "of static vehicles"
        ),
    )
    vehicle_placement.add_argument(
        "--no-vehicle-blockers",
        "--no-vehicles",
        dest="no_vehicle_blockers",
        action="store_true",
        help="do not spawn static vehicle blockers",
    )
    parser.add_argument(
        "--no-reactive-vehicle",
        action="store_true",
        help=(
            "keep every configured vehicle frozen; disable the vehicle-to-"
            "ego-pedestrian crosswalk encounter"
        ),
    )
    parser.add_argument(
        "--reactive-vehicle-index",
        default=DEFAULT_REACTIVE_VEHICLE_INDEX,
        type=positive_int,
        metavar="INDEX",
        help=(
            "one-based configured vehicle index used for the crosswalk "
            "encounter (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--ego-pedestrian-role-name",
        default=DEFAULT_EGO_PEDESTRIAN_ROLE_NAME,
        metavar="NAME",
        help=(
            "pedestrian_head_camera_client_v8.py/v9.py controlled-walker "
            "role_name "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--ego-pedestrian-actor-id",
        default=None,
        type=positive_int,
        metavar="ID",
        help=(
            "optional exact ego-pedestrian actor ID; role lookup is preferred "
            "because it survives Y respawns"
        ),
    )
    parser.add_argument(
        "--reactive-attack-location",
        "--ego-pedestrian-target",
        dest="reactive_attack_location",
        nargs=2,
        default=DEFAULT_REACTIVE_ATTACK_LOCATION,
        type=finite_float,
        metavar=("X", "Y"),
        help=(
            "nominal crosswalk anchor in CARLA world XY coordinates; the "
            "live impact station is selected on the nearby lane route "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-route-sampling-resolution",
        default=DEFAULT_REACTIVE_ROUTE_SAMPLING_RESOLUTION_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "GlobalRoutePlanner waypoint spacing for the road-following "
            "vehicle route (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-route-lookahead",
        default=DEFAULT_REACTIVE_ROUTE_LOOKAHEAD_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "lane-route lookahead used by reactive vehicle steering "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-route-home-projection-limit",
        default=DEFAULT_REACTIVE_ROUTE_HOME_PROJECTION_LIMIT_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "maximum distance the captured blocker home may be moved to the "
            "nearest Driving waypoint (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-route-attack-offset-limit",
        default=DEFAULT_REACTIVE_ROUTE_ATTACK_OFFSET_LIMIT_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "maximum pedestrian-target offset from the planned lane "
            "centerline (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-route-deviation-limit",
        default=DEFAULT_REACTIVE_ROUTE_DEVIATION_LIMIT_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "maximum reactive-vehicle center deviation from its lane route "
            "before the encounter is recycled (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-route-max-length",
        default=DEFAULT_REACTIVE_ROUTE_MAX_LENGTH_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "maximum accepted road-route distance to the crosswalk, guarding "
            "against a wrong-way lane detour (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-route-max-lateral-acceleration",
        default=DEFAULT_REACTIVE_ROUTE_MAX_LATERAL_ACCELERATION_MPS2,
        type=positive_float,
        metavar="MPS2",
        help=(
            "lateral-acceleration budget used to derive a curve-safe route "
            "speed cap (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-route-steer-limit",
        default=DEFAULT_REACTIVE_STEER_LIMIT_RADIANS,
        type=positive_float,
        metavar="RADIANS",
        help=(
            "maximum Ackermann steering angle for lane-route following "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-intercept-route-window",
        default=DEFAULT_REACTIVE_INTERCEPT_ROUTE_WINDOW_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "route distance before/after the nominal crosswalk searched for "
            "the live pedestrian crossing (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-min-crossing-angle",
        default=DEFAULT_REACTIVE_MIN_CROSSING_ANGLE_DEGREES,
        type=nonnegative_float,
        metavar="DEGREES",
        help=(
            "minimum acute angle between pedestrian motion and the vehicle "
            "lane route (default: %(default)s)"
        ),
    )
    best_effort_group = parser.add_mutually_exclusive_group()
    best_effort_group.add_argument(
        "--reactive-best-effort-launch",
        dest="reactive_best_effort_launch",
        action="store_true",
        help=(
            "launch at the safest maximum route speed when synchronized "
            "contact is already physically infeasible"
        ),
    )
    best_effort_group.add_argument(
        "--no-reactive-best-effort-launch",
        dest="reactive_best_effort_launch",
        action="store_false",
        help="retain the legacy behavior of waiting when contact is infeasible",
    )
    parser.set_defaults(
        reactive_best_effort_launch=DEFAULT_REACTIVE_BEST_EFFORT_LAUNCH
    )
    parser.add_argument(
        "--reactive-trigger-distance",
        default=DEFAULT_REACTIVE_TRIGGER_DISTANCE_M,
        type=positive_float,
        metavar="METERS",
        help="ego-pedestrian approach radius (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-rearm-distance",
        default=DEFAULT_REACTIVE_REARM_DISTANCE_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "distance the ego pedestrian must leave the target after a run "
            "before another run can arm (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-min-pedestrian-speed",
        default=DEFAULT_REACTIVE_MIN_PEDESTRIAN_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help="minimum ego-pedestrian speed for activation (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-min-closing-speed",
        default=DEFAULT_REACTIVE_MIN_CLOSING_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help="minimum closing speed toward the target (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-max-cross-track",
        default=DEFAULT_REACTIVE_MAX_CROSS_TRACK_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "maximum constant-velocity miss distance for an inbound "
            "pedestrian (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-min-pedestrian-eta",
        default=DEFAULT_REACTIVE_MIN_PEDESTRIAN_ETA_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="minimum accepted target ETA (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-max-pedestrian-eta",
        default=DEFAULT_REACTIVE_MAX_PEDESTRIAN_ETA_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="maximum accepted target ETA (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-pedestrian-acceleration-limit",
        default=DEFAULT_REACTIVE_PEDESTRIAN_ACCELERATION_LIMIT_MPS2,
        type=positive_float,
        metavar="MPS2",
        help=(
            "maximum radial pedestrian acceleration used by target ETA "
            "prediction (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-vehicle-control",
        choices=("ackermann", "constant-velocity"),
        default="ackermann",
        help=(
            "vehicle motion model; ackermann preserves normal rigid-body "
            "response, constant-velocity is a scripted deterministic mode "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-vehicle-speed-min",
        default=DEFAULT_REACTIVE_VEHICLE_SPEED_MIN_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help="minimum nonzero synchronized command speed (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-vehicle-speed-max",
        default=DEFAULT_REACTIVE_VEHICLE_SPEED_MAX_MPS,
        type=positive_float,
        metavar="MPS",
        help="maximum synchronized vehicle speed (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-vehicle-max-acceleration",
        default=DEFAULT_REACTIVE_VEHICLE_MAX_ACCELERATION_MPS2,
        type=positive_float,
        metavar="MPS2",
        help="maximum planned acceleration (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-vehicle-max-deceleration",
        default=DEFAULT_REACTIVE_VEHICLE_MAX_DECELERATION_MPS2,
        type=positive_float,
        metavar="MPS2",
        help="maximum planned deceleration magnitude (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-impact-lead",
        default=DEFAULT_REACTIVE_IMPACT_LEAD_SECONDS,
        type=nonnegative_float,
        metavar="SECONDS",
        help=(
            "command/physics latency allowance subtracted from pedestrian "
            "ETA (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-launch-margin",
        default=DEFAULT_REACTIVE_LAUNCH_MARGIN_SECONDS,
        type=nonnegative_float,
        metavar="SECONDS",
        help=(
            "how early, relative to the optimistic vehicle travel time, the "
            "staged vehicle may leave its road home (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-approach-hold-distance",
        default=DEFAULT_REACTIVE_APPROACH_HOLD_DISTANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help=(
            "pre-commit front-bumper hold distance upstream of the contact "
            "plane when pedestrian timing moves later (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-approach-hold-timeout",
        default=DEFAULT_REACTIVE_APPROACH_HOLD_TIMEOUT_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help=(
            "maximum time to wait at road home or in a pre-commit hold before "
            "classifying the encounter as a miss (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-commit-distance",
        default=DEFAULT_REACTIVE_COMMIT_DISTANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help=(
            "front distance at which the vehicle stops braking away from the "
            "encounter (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-commit-speed",
        default=DEFAULT_REACTIVE_COMMIT_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help="minimum speed inside the committed zone (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-runout-speed",
        default=DEFAULT_REACTIVE_RUNOUT_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help="collision-free miss drive-through speed (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-post-distance",
        default=DEFAULT_REACTIVE_POST_DISTANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help="distance the vehicle front travels beyond target (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-active-timeout",
        default=DEFAULT_REACTIVE_ACTIVE_TIMEOUT_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="maximum synchronized-approach duration (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-runout-timeout",
        default=DEFAULT_REACTIVE_RUNOUT_TIMEOUT_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="maximum post-event drive-through duration (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-target-pass-distance",
        default=DEFAULT_REACTIVE_TARGET_PASS_DISTANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help=(
            "pedestrian progress beyond the point that declares a miss "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-impact-settle-time",
        default=DEFAULT_REACTIVE_IMPACT_SETTLE_SECONDS,
        type=nonnegative_float,
        metavar="SECONDS",
        help=(
            "physics-only coast time after collision before service braking "
            "begins (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-contact-hold-time",
        default=DEFAULT_REACTIVE_CONTACT_HOLD_SECONDS,
        type=nonnegative_float,
        metavar="SECONDS",
        help=(
            "time the collided vehicle remains visible after the physics "
            "settle interval (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-contact-stop-speed",
        default=DEFAULT_REACTIVE_CONTACT_STOP_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help=(
            "speed below which the parking brake may hold a collided vehicle "
            "at rest (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reactive-respawn-delay",
        default=DEFAULT_REACTIVE_RESPAWN_DELAY_SECONDS,
        type=nonnegative_float,
        metavar="SECONDS",
        help="delay after retiring the attack vehicle (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-respawn-retry",
        default=DEFAULT_REACTIVE_RESPAWN_RETRY_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="retry interval while its home is occupied (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-respawn-clearance",
        default=DEFAULT_REACTIVE_RESPAWN_CLEARANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help="required clear radius around vehicle home (default: %(default)s)",
    )
    parser.add_argument(
        "--reactive-activation-z-lift",
        default=DEFAULT_REACTIVE_ACTIVATION_Z_LIFT_M,
        type=nonnegative_float,
        metavar="METERS",
        help=(
            "temporary height added before enabling vehicle physics; keep 0 "
            "unless the captured home embeds the vehicle (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--pedestrian-z-offset",
        "--z-offset",
        dest="z_offset",
        default=DEFAULT_PEDESTRIAN_Z_OFFSET_M,
        type=finite_float,
        metavar="METERS",
        help="height added to requested ground Z at spawn (default: %(default)s)",
    )
    parser.add_argument(
        "--vehicle-z-offset",
        default=DEFAULT_VEHICLE_Z_OFFSET_M,
        type=finite_float,
        metavar="METERS",
        help=(
            "height added to each vehicle ground-target Z at spawn "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--nav-samples",
        default=DEFAULT_NAVIGATION_SAMPLES,
        type=positive_int,
        metavar="COUNT",
        help="random navigation samples used by spawn fallback (default: %(default)s)",
    )
    parser.add_argument(
        "--nav-search-radius",
        default=DEFAULT_NAVIGATION_SEARCH_RADIUS_M,
        type=positive_float,
        metavar="METERS",
        help="maximum sampled fallback distance (default: %(default)s)",
    )
    parser.add_argument(
        "--placement-tolerance",
        default=DEFAULT_PLACEMENT_TOLERANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help="relocation verification tolerance (default: %(default)s)",
    )
    parser.add_argument(
        "--spectator-ground-search",
        default=DEFAULT_SPECTATOR_GROUND_SEARCH_M,
        type=positive_float,
        metavar="METERS",
        help="vertical ground-projection search below spectator (default: %(default)s)",
    )
    parser.add_argument(
        "--ego-role-name",
        default=DEFAULT_EGO_ROLE_NAME,
        metavar="NAME",
        help="manual_control_ar_v8.py --rolename value (default: %(default)s)",
    )
    parser.add_argument(
        "--ego-actor-id",
        default=None,
        type=positive_int,
        metavar="ID",
        help="optional exact ego actor ID; role lookup is preferred across respawns",
    )
    parser.add_argument(
        "--pedestrian-speed",
        default=DEFAULT_PEDESTRIAN_SPEED_MPS,
        type=positive_float,
        metavar="MPS",
        help=(
            "maximum physical speed available to the timed pedestrian "
            "(default: %(default)s m/s)"
        ),
    )
    parser.add_argument(
        "--min-pedestrian-speed",
        default=DEFAULT_MIN_PEDESTRIAN_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help=(
            "minimum speed accepted when arming a synchronized intercept; "
            "active retiming may command a lower speed (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--impact-target",
        choices=("front", "center"),
        default=DEFAULT_IMPACT_TARGET,
        help=(
            "vehicle reference synchronized with the pedestrian at L2; "
            "front uses oriented bounding-box contact geometry, while center "
            "restores the former actor-origin behavior (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--front-impact-margin",
        default=DEFAULT_FRONT_IMPACT_MARGIN_M,
        type=nonnegative_float,
        metavar="METERS",
        help=(
            "additional front-reference lead distance used to compensate "
            "control/update latency; applies only to --impact-target front "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--walker-control-speed-scale",
        default=DEFAULT_WALKER_CONTROL_SPEED_SCALE,
        type=positive_float,
        metavar="FACTOR",
        help=(
            "animation-only WalkerControl speed multiplier, capped at "
            "{:.1f} m/s; physical motion uses the synchronized speed up to "
            "--pedestrian-speed "
            "(default: %(default)s)"
        ).format(DEFAULT_WALKER_ANIMATION_SPEED_CAP_MPS),
    )
    parser.add_argument(
        "--min-ego-speed",
        default=DEFAULT_MIN_EGO_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help="minimum ego speed for activation (default: %(default)s m/s)",
    )
    parser.add_argument(
        "--min-closing-speed",
        default=DEFAULT_MIN_CLOSING_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help="minimum ego closing speed toward pedestrian (default: %(default)s m/s)",
    )
    parser.add_argument(
        "--trigger-distance",
        default=DEFAULT_TRIGGER_DISTANCE_M,
        type=positive_float,
        metavar="METERS",
        help="maximum ego-to-pedestrian activation range (default: %(default)s)",
    )
    parser.add_argument(
        "--max-approach-angle",
        default=DEFAULT_MAX_APPROACH_ANGLE_DEG,
        type=positive_float,
        metavar="DEGREES",
        help="maximum heading angle to a waiting pedestrian (default: %(default)s)",
    )
    parser.add_argument(
        "--max-lateral-offset",
        default=DEFAULT_MAX_LATERAL_OFFSET_M,
        type=positive_float,
        metavar="METERS",
        help="maximum pedestrian offset from ego travel line (default: %(default)s)",
    )
    parser.add_argument(
        "--reaction-time",
        default=DEFAULT_REACTION_TIME_SECONDS,
        type=nonnegative_float,
        metavar="SECONDS",
        help="modeled driver reaction time (default: %(default)s)",
    )
    parser.add_argument(
        "--max-brake-deceleration",
        default=DEFAULT_MAX_BRAKE_DECELERATION_MPS2,
        type=positive_float,
        metavar="MPS2",
        help="modeled hard-braking deceleration (default: %(default)s)",
    )
    parser.add_argument(
        "--braking-margin",
        default=DEFAULT_BRAKING_MARGIN_M,
        type=nonnegative_float,
        metavar="METERS",
        help="additional modeled stopping-distance margin (default: %(default)s)",
    )
    parser.add_argument(
        "--max-intercept-time",
        default=DEFAULT_MAX_INTERCEPT_TIME_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="maximum accepted intercept horizon (default: %(default)s)",
    )
    parser.add_argument(
        "--min-intercept-time",
        default=DEFAULT_MIN_INTERCEPT_TIME_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="minimum accepted intercept horizon (default: %(default)s)",
    )
    parser.add_argument(
        "--prediction-acceleration-limit",
        default=DEFAULT_PREDICTION_ACCELERATION_LIMIT_MPS2,
        type=positive_float,
        metavar="MPS2",
        help=(
            "maximum XY acceleration magnitude used by prediction, limiting "
            "single-frame physics spikes (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--acceleration-smoothing",
        default=DEFAULT_ACCELERATION_SMOOTHING,
        type=nonnegative_float,
        metavar="FACTOR",
        help=(
            "new-sample weight for active ego acceleration filtering in [0,1] "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--active-perpendicular-tolerance",
        default=DEFAULT_ACTIVE_PERPENDICULAR_TOLERANCE_DEG,
        type=positive_float,
        metavar="DEGREES",
        help=(
            "maximum active ego-tangent change allowed before recycling the "
            "locked perpendicular encounter (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--active-timeout",
        default=DEFAULT_ACTIVE_TIMEOUT_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="maximum crossing duration before recycling (default: %(default)s)",
    )
    parser.add_argument(
        "--expire-distance",
        default=DEFAULT_EXPIRE_DISTANCE_M,
        type=positive_float,
        metavar="METERS",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--collision-distance",
        default=DEFAULT_COLLISION_DISTANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help=(
            "center-distance collision fallback used when a collision sensor "
            "is unavailable; oriented-box overlap is always checked "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--crossing-extra-distance",
        default=DEFAULT_CROSSING_EXTRA_DISTANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--intercept-arrival-tolerance",
        default=DEFAULT_INTERCEPT_ARRIVAL_TOLERANCE_M,
        type=positive_float,
        metavar="METERS",
        help="distance at which the pedestrian is held at L2 (default: %(default)s)",
    )
    parser.add_argument(
        "--near-miss-distance",
        default=DEFAULT_NEAR_MISS_DISTANCE_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "maximum closest footprint gap for a collision-free pass or "
            "hard-braking stop to count as a near miss (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--hard-brake-deceleration",
        default=DEFAULT_HARD_BRAKE_DECELERATION_MPS2,
        type=positive_float,
        metavar="MPS2",
        help=(
            "longitudinal deceleration magnitude that latches a hard-brake "
            "near-miss candidate (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--stopped-ego-speed",
        default=DEFAULT_STOPPED_EGO_SPEED_MPS,
        type=nonnegative_float,
        metavar="MPS",
        help="ego speed treated as stopped during near-miss detection (default: %(default)s)",
    )
    parser.add_argument(
        "--post-event-hold",
        default=DEFAULT_POST_EVENT_HOLD_SECONDS,
        type=nonnegative_float,
        metavar="SECONDS",
        help=(
            "simulation-time wait with the pedestrian visible at the event "
            "location before respawn (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--respawn-delay",
        default=DEFAULT_RESPAWN_DELAY_SECONDS,
        type=nonnegative_float,
        metavar="SECONDS",
        help=(
            "additional simulation-time delay used for non-event recycling "
            "after the actor is retired "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--respawn-retry-interval",
        default=DEFAULT_RESPAWN_RETRY_INTERVAL_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help=(
            "retry interval when the home transform is occupied "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--respawn-clearance",
        default=DEFAULT_RESPAWN_CLEARANCE_M,
        type=nonnegative_float,
        metavar="METERS",
        help=(
            "minimum XY distance from the original target to every vehicle "
            "footprint before respawning (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--motion-stall-timeout",
        default=DEFAULT_MOTION_STALL_TIMEOUT_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help=(
            "simulation time without sufficient crossing progress before "
            "stall recovery (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--motion-stall-min-progress",
        default=DEFAULT_MOTION_STALL_MIN_PROGRESS_M,
        type=positive_float,
        metavar="METERS",
        help=(
            "progress needed to reset the motion-stall watchdog "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--stall-recovery-step",
        default=DEFAULT_STALL_RECOVERY_STEP_M,
        type=nonnegative_float,
        metavar="METERS",
        help=(
            "maximum scripted transform step per update after a verified "
            "stall; vehicle-footprint sweeps clamp each step, and 0 recycles "
            "the stalled walker instead "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-motion-command-failures",
        default=DEFAULT_MAX_MOTION_COMMAND_FAILURES,
        type=positive_int,
        metavar="COUNT",
        help=(
            "consecutive motion-command failures before recycling a walker "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-scripted-recovery-time",
        default=DEFAULT_MAX_SCRIPTED_RECOVERY_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help=(
            "maximum simulation time allowed in scripted stall recovery "
            "before recycling the walker (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-active-pedestrians",
        default=DEFAULT_MAX_ACTIVE_PEDESTRIANS,
        type=positive_int,
        metavar="COUNT",
        help="maximum simultaneous activations (default: %(default)s)",
    )
    parser.add_argument(
        "--update-hz",
        default=DEFAULT_UPDATE_HZ,
        type=positive_float,
        metavar="HZ",
        help="maximum decision/control update rate (default: %(default)s)",
    )
    parser.add_argument(
        "--tick-timeout",
        default=DEFAULT_TICK_TIMEOUT_SECONDS,
        type=positive_float,
        metavar="SECONDS",
        help="passive wait_for_tick timeout (default: %(default)s)",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--intercept-debug",
        dest="intercept_debug",
        action="store_true",
        help="draw the predicted ego path, perpendicular crossing line, and L2",
    )
    debug_group.add_argument(
        "--no-intercept-debug",
        dest="intercept_debug",
        action="store_false",
        help="disable the live intercept debug geometry",
    )
    parser.set_defaults(intercept_debug=True)

    args = parser.parse_args(argv)
    args.host = args.host.strip()
    args.blueprint = args.blueprint.strip()
    args.vehicle_blueprint = args.vehicle_blueprint.strip()
    args.ego_role_name = args.ego_role_name.strip()
    args.ego_pedestrian_role_name = args.ego_pedestrian_role_name.strip()
    if not args.host:
        parser.error("--host must not be empty")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    per_zone_network_options_used = any(
        (
            args.ego_vehicle_network_degradation_zone is not None,
            args.disable_ego_vehicle_network_degradation_zone,
            args.ego_pedestrian_network_degradation_zone is not None,
            args.disable_ego_pedestrian_network_degradation_zone,
        )
    )
    if args.disable_network_degradation and per_zone_network_options_used:
        parser.error(
            "--disable-network-degradation cannot be combined with per-zone "
            "network-degradation options"
        )
    configured_network_zones = []
    if not args.disable_network_degradation:
        if not args.disable_ego_vehicle_network_degradation_zone:
            vehicle_zone = (
                args.ego_vehicle_network_degradation_zone
                if args.ego_vehicle_network_degradation_zone is not None
                else DEFAULT_EGO_VEHICLE_NETWORK_DEGRADATION_ZONE
            )
            try:
                configured_network_zones.append(
                    normalize_zone(
                        NETWORK_PROFILE_VEHICLE_ZONE_INDEX,
                        *vehicle_zone
                    )
                )
            except NetworkProfileError as exc:
                parser.error(
                    "invalid ego-vehicle network-degradation zone: {}".format(
                        exc
                    )
                )
        if not args.disable_ego_pedestrian_network_degradation_zone:
            pedestrian_zone = (
                args.ego_pedestrian_network_degradation_zone
                if args.ego_pedestrian_network_degradation_zone is not None
                else DEFAULT_EGO_PEDESTRIAN_NETWORK_DEGRADATION_ZONE
            )
            try:
                configured_network_zones.append(
                    normalize_zone(
                        NETWORK_PROFILE_PEDESTRIAN_ZONE_INDEX,
                        *pedestrian_zone
                    )
                )
            except NetworkProfileError as exc:
                parser.error(
                    "invalid ego-pedestrian network-degradation zone: {}".format(
                        exc
                    )
                )
    args.network_degradation_zones = tuple(configured_network_zones)
    if not args.blueprint.startswith("walker.pedestrian."):
        parser.error("--blueprint must be an exact walker.pedestrian.* ID")
    if not args.vehicle_blueprint.startswith("vehicle."):
        parser.error("--vehicle-blueprint must be an exact vehicle.* ID")
    if args.no_vehicle_blockers and args.no_pedestrian_blockers:
        parser.error("at least one blocker category must remain enabled")
    if (
        not args.no_pedestrian_blockers
        and args.ego_actor_id is None
        and not args.ego_role_name
    ):
        parser.error("--ego-role-name must not be empty without --ego-actor-id")
    if (
        not args.no_vehicle_blockers
        and not args.no_reactive_vehicle
        and args.ego_pedestrian_actor_id is None
        and not args.ego_pedestrian_role_name
    ):
        parser.error(
            "--ego-pedestrian-role-name must not be empty without "
            "--ego-pedestrian-actor-id"
        )
    if (
        args.reactive_min_pedestrian_eta
        >= args.reactive_max_pedestrian_eta
    ):
        parser.error(
            "--reactive-min-pedestrian-eta must be less than "
            "--reactive-max-pedestrian-eta"
        )
    if args.reactive_impact_lead >= args.reactive_min_pedestrian_eta:
        parser.error(
            "--reactive-impact-lead must be less than "
            "--reactive-min-pedestrian-eta"
        )
    if args.reactive_vehicle_speed_min > args.reactive_vehicle_speed_max:
        parser.error(
            "--reactive-vehicle-speed-min cannot exceed "
            "--reactive-vehicle-speed-max"
        )
    if args.reactive_commit_speed > args.reactive_vehicle_speed_max:
        parser.error(
            "--reactive-commit-speed cannot exceed "
            "--reactive-vehicle-speed-max"
        )
    if args.reactive_runout_speed > args.reactive_vehicle_speed_max:
        parser.error(
            "--reactive-runout-speed cannot exceed "
            "--reactive-vehicle-speed-max"
        )
    if not 0.1 <= args.reactive_route_sampling_resolution <= 2.0:
        parser.error(
            "--reactive-route-sampling-resolution must be between 0.1 and 2 meters"
        )
    if args.reactive_route_steer_limit > math.pi / 2.0:
        parser.error(
            "--reactive-route-steer-limit must not exceed pi/2 radians"
        )
    if not 0.0 < args.reactive_min_crossing_angle < 90.0:
        parser.error(
            "--reactive-min-crossing-angle must be greater than 0 and less "
            "than 90 degrees"
        )
    if args.reactive_route_max_length <= args.reactive_post_distance:
        parser.error(
            "--reactive-route-max-length must exceed --reactive-post-distance"
        )
    if not 0.0 < args.max_approach_angle < 90.0:
        parser.error("--max-approach-angle must be between 0 and 90 degrees")
    if args.min_pedestrian_speed > args.pedestrian_speed:
        parser.error("--min-pedestrian-speed cannot exceed --pedestrian-speed")
    if args.min_intercept_time >= args.max_intercept_time:
        parser.error("--min-intercept-time must be less than --max-intercept-time")
    if args.acceleration_smoothing > 1.0:
        parser.error("--acceleration-smoothing must be between 0 and 1")
    if args.active_perpendicular_tolerance >= 45.0:
        parser.error("--active-perpendicular-tolerance must be less than 45 degrees")
    return args


@dataclass(frozen=True)
class VehicleTarget:
    x: float
    y: float
    z: float
    yaw: float
    source: str
    blueprint_id: Optional[str] = None


@dataclass(frozen=True)
class StaticVehicleSpawnResult:
    actor: object
    transform: carla.Transform
    placement_source: str


@dataclass(frozen=True)
class PedestrianTarget:
    x: float
    y: float
    z: float
    yaw: float
    source: str

    def location(self) -> carla.Location:
        return carla.Location(x=self.x, y=self.y, z=self.z)


@dataclass(frozen=True)
class InterceptSolution:
    time_seconds: float
    target_x: float
    target_y: float
    tangent_x: float
    tangent_y: float
    pedestrian_direction_x: float
    pedestrian_direction_y: float
    required_pedestrian_speed: float
    pedestrian_distance: float
    ego_travel: float
    acceleration_x: float
    acceleration_y: float
    front_contact_offset: float = 0.0


@dataclass(frozen=True)
class LineInterceptUpdate:
    time_seconds: float
    target_x: float
    target_y: float
    required_pedestrian_speed: float
    longitudinal_acceleration: float
    perpendicular_error_degrees: float


@dataclass(frozen=True)
class ActiveUpdateResult:
    action: str
    reason: str


@dataclass(frozen=True)
class TriggerDecision:
    intercept: InterceptSolution
    ego_speed: float
    separation: float
    closing_speed: float
    lateral_offset: float
    approach_angle_degrees: float
    ego_travel: float
    effective_travel: float
    stopping_distance: float


@dataclass(frozen=True)
class InactiveFrontSensorPair:
    """Initially unsubscribed camera/radar actors at one rigid shared pose."""

    camera: object
    radar: object


@dataclass(frozen=True)
class TrafficLightRouteCandidate:
    """One pole root ranked against the ego-pedestrian route polyline."""

    traffic_light: object
    actor_id: int
    distance_m: float
    route_progress_m: float
    segment_index: int


@dataclass(frozen=True)
class PreparedTrafficLightSensorDeployment:
    """Read-only validated mount/log data for one selected pole pair."""

    candidate: TrafficLightRouteCandidate
    mount_transform: object
    ground_z: float
    ground_source: str
    root_x: float
    root_y: float
    sensor_world_z: float
    world_yaw: float
    pitch: float


@dataclass(frozen=True)
class PublishedNetworkProfileActors:
    """Actors owned by one committed shared network-profile publication."""

    session_token: str
    zones: Tuple[NetworkDegradationZone, ...]
    zone_actors: Tuple[object, ...]
    manifest_actor: object
    start_active_sensors: bool
    owns_actors: bool = True


@dataclass
class PedestrianState:
    index: int
    actor: Optional[object]
    target: PedestrianTarget
    sensor: Optional[object] = None
    state: str = STATE_WAITING
    generation: int = 1
    active_since: Optional[float] = None
    active_ego_id: Optional[int] = None
    last_separation: Optional[float] = None
    crossing_origin: Optional[Tuple[float, float]] = None
    crossing_direction: Optional[Tuple[float, float]] = None
    crossing_endpoint: Optional[Tuple[float, float]] = None
    crossing_distance: Optional[float] = None
    ego_path_direction: Optional[Tuple[float, float]] = None
    front_contact_offset: float = 0.0
    commanded_pedestrian_speed: Optional[float] = None
    filtered_ego_acceleration: Optional[Tuple[float, float]] = None
    last_ego_intercept_signed_distance: Optional[float] = None
    minimum_ego_surface_gap: Optional[float] = None
    hard_brake_seen: bool = False
    intercept_reached: bool = False
    pending_near_miss_reason: Optional[str] = None
    last_progress: Optional[float] = None
    last_progress_time: Optional[float] = None
    motion_command_failures: int = 0
    stall_recovery_count: int = 0
    scripted_recovery_active: bool = False
    scripted_recovery_started_at: Optional[float] = None
    last_motion_update_time: Optional[float] = None
    last_debug_draw_time: Optional[float] = None
    hold_until: Optional[float] = None
    hold_reason: Optional[str] = None
    respawn_due: Optional[float] = None
    retired_actor_id: Optional[int] = None
    retired_sensor_id: Optional[int] = None


@dataclass(frozen=True)
class PedestrianTargetApproach:
    """Current ego-pedestrian estimate toward one lane-route intersection."""

    distance: float
    speed: float
    closing_speed: float
    cross_track_miss: float
    eta_seconds: float
    direction_x: float
    direction_y: float
    radial_acceleration: float
    target_x: float = 0.0
    target_y: float = 0.0
    route_station: float = 0.0
    route_tangent_x: float = 1.0
    route_tangent_y: float = 0.0
    crossing_angle_degrees: float = 90.0


@dataclass(frozen=True)
class ReactiveVehicleCommand:
    speed: float
    acceleration: float
    time_available: float


@dataclass(frozen=True)
class ReactiveVehicleContact:
    """Strongest collision-sensor event for one other actor in a tick."""

    actor_id: int
    type_id: str
    role_name: str
    frame: int
    impulse_magnitude: float


@dataclass(frozen=True)
class ReactiveRouteProjection:
    """Closest XY projection onto the preplanned Driving-lane polyline."""

    station: float
    segment_index: int
    projected_x: float
    projected_y: float
    tangent_x: float
    tangent_y: float
    gap: float
    signed_cross_track: float


@dataclass(frozen=True)
class ReactiveVehicleRoutePlan:
    """Immutable road route shared by every respawn generation."""

    requested_home: VehicleTarget
    home: VehicleTarget
    locations: Tuple[carla.Location, ...]
    distances: Tuple[float, ...]
    attack_station: float
    route_length: float
    attack_tangent_x: float
    attack_tangent_y: float
    home_snap_distance: float
    attack_lateral_offset: float
    maximum_curvature: float
    speed_limit: float
    home_road_id: int
    home_lane_id: int
    attack_road_id: int
    attack_lane_id: int


@dataclass
class ReactiveVehicleState:
    """Owned lifecycle for the one vehicle that attacks the crosswalk."""

    index: int
    actor: Optional[object]
    target: VehicleTarget
    attack_x: float
    attack_y: float
    origin_x: float
    origin_y: float
    direction_x: float
    direction_y: float
    attack_distance: float
    attack_yaw: float
    nominal_attack_x: float
    nominal_attack_y: float
    nominal_attack_distance: float
    nominal_direction_x: float
    nominal_direction_y: float
    rearm_attack_x: float
    rearm_attack_y: float
    requested_target: VehicleTarget
    route_locations: Tuple[carla.Location, ...]
    route_distances: Tuple[float, ...]
    route_length: float
    home_snap_distance: float
    attack_lateral_offset: float
    maximum_route_curvature: float
    route_speed_limit: float
    home_road_id: int
    home_lane_id: int
    attack_road_id: int
    attack_lane_id: int
    sensor: Optional[object] = None
    home_clearance_ignored_actor_ids: Tuple[int, ...] = ()
    state: str = REACTIVE_STATE_WAITING
    generation: int = 1
    armed: bool = True
    active_target_id: Optional[int] = None
    pedestrian_direction: Optional[Tuple[float, float]] = None
    pedestrian_contact_support: float = 0.0
    front_support: float = 0.0
    active_since: Optional[float] = None
    aligning_since: Optional[float] = None
    runout_since: Optional[float] = None
    runout_start_front_progress: Optional[float] = None
    contact_since: Optional[float] = None
    contact_settle_until: Optional[float] = None
    contact_hold_until: Optional[float] = None
    contact_settle_wall_until: Optional[float] = None
    contact_hold_wall_until: Optional[float] = None
    contact_brake_applied: bool = False
    impact_speed: float = 0.0
    impact_impulse: float = 0.0
    last_progress: Optional[float] = None
    last_progress_time: Optional[float] = None
    last_status_log_time: Optional[float] = None
    commanded_speed: float = 0.0
    contact_detected: bool = False
    outcome: Optional[str] = None
    respawn_due: Optional[float] = None
    retired_actor_id: Optional[int] = None
    retired_sensor_id: Optional[int] = None
    route_segment_index: int = 0
    route_progress: float = 0.0
    route_projection_gap: float = 0.0
    predicted_contact_time: Optional[float] = None
    best_effort_launch: bool = False
    committed: bool = False
    approach_hold_since: Optional[float] = None
    previous_contact_sample_time: Optional[float] = None
    previous_vehicle_location: Optional[Tuple[float, float, float]] = None
    previous_pedestrian_location: Optional[Tuple[float, float, float]] = None
    transient_failure_count: int = 0
    transient_failure_reason: Optional[str] = None
    transient_failure_since: Optional[float] = None
    transient_failure_wall_since: Optional[float] = None


def planar_distance(first: carla.Location, second: carla.Location) -> float:
    return math.hypot(float(first.x - second.x), float(first.y - second.y))


def normalized_xy(x_coord: float, y_coord: float) -> Optional[Tuple[float, float]]:
    magnitude = math.hypot(float(x_coord), float(y_coord))
    if magnitude <= 1e-9:
        return None
    return float(x_coord) / magnitude, float(y_coord) / magnitude


def copy_carla_location(location: carla.Location) -> carla.Location:
    return carla.Location(
        x=float(location.x),
        y=float(location.y),
        z=float(location.z),
    )


def dedupe_route_locations(
    locations: Sequence[carla.Location],
    minimum_distance: float = 0.05,
) -> Tuple[carla.Location, ...]:
    result: List[carla.Location] = []
    for location in locations:
        copied = copy_carla_location(location)
        if result and planar_distance(result[-1], copied) < minimum_distance:
            continue
        result.append(copied)
    return tuple(result)


def route_cumulative_distances(
    locations: Sequence[carla.Location],
) -> Tuple[float, ...]:
    if not locations:
        return tuple()
    distances = [0.0]
    for previous, current in zip(locations, locations[1:]):
        distances.append(distances[-1] + planar_distance(previous, current))
    return tuple(distances)


def project_xy_onto_route(
    locations: Sequence[carla.Location],
    distances: Sequence[float],
    x_coord: float,
    y_coord: float,
    start_segment: int = 0,
    end_segment: Optional[int] = None,
) -> Optional[ReactiveRouteProjection]:
    """Project one XY point onto a bounded range of route segments."""
    segment_count = min(len(locations), len(distances)) - 1
    if segment_count <= 0:
        return None
    first_segment = max(0, min(int(start_segment), segment_count - 1))
    last_segment = segment_count - 1
    if end_segment is not None:
        last_segment = max(
            first_segment,
            min(int(end_segment), segment_count - 1),
        )
    best: Optional[ReactiveRouteProjection] = None
    px = float(x_coord)
    py = float(y_coord)
    for segment_index in range(first_segment, last_segment + 1):
        start = locations[segment_index]
        end = locations[segment_index + 1]
        delta_x = float(end.x - start.x)
        delta_y = float(end.y - start.y)
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 1.0e-10:
            continue
        fraction = (
            (px - float(start.x)) * delta_x
            + (py - float(start.y)) * delta_y
        ) / length_squared
        fraction = max(0.0, min(1.0, fraction))
        projected_x = float(start.x) + fraction * delta_x
        projected_y = float(start.y) + fraction * delta_y
        error_x = px - projected_x
        error_y = py - projected_y
        gap = math.hypot(error_x, error_y)
        segment_length = math.sqrt(length_squared)
        tangent_x = delta_x / segment_length
        tangent_y = delta_y / segment_length
        projection = ReactiveRouteProjection(
            station=float(distances[segment_index]) + fraction * segment_length,
            segment_index=segment_index,
            projected_x=projected_x,
            projected_y=projected_y,
            tangent_x=tangent_x,
            tangent_y=tangent_y,
            gap=gap,
            signed_cross_track=tangent_x * error_y - tangent_y * error_x,
        )
        if best is None or projection.gap < best.gap:
            best = projection
    return best


def route_pose_at_station(
    locations: Sequence[carla.Location],
    distances: Sequence[float],
    station: float,
) -> Optional[Tuple[carla.Location, Tuple[float, float], int]]:
    """Interpolate a location and tangent at one cumulative route station."""
    count = min(len(locations), len(distances))
    if count < 2:
        return None
    target_station = max(0.0, min(float(station), float(distances[count - 1])))
    segment_index = count - 2
    for index in range(count - 1):
        if target_station <= float(distances[index + 1]) + 1.0e-9:
            segment_index = index
            break
    start = locations[segment_index]
    end = locations[segment_index + 1]
    delta_x = float(end.x - start.x)
    delta_y = float(end.y - start.y)
    segment_length = math.hypot(delta_x, delta_y)
    if segment_length <= 1.0e-9:
        return None
    fraction = (
        target_station - float(distances[segment_index])
    ) / segment_length
    fraction = max(0.0, min(1.0, fraction))
    location = carla.Location(
        x=float(start.x) + fraction * delta_x,
        y=float(start.y) + fraction * delta_y,
        z=float(start.z) + fraction * float(end.z - start.z),
    )
    return (
        location,
        (delta_x / segment_length, delta_y / segment_length),
        segment_index,
    )


def route_maximum_curvature(
    locations: Sequence[carla.Location],
    maximum_station: Optional[float] = None,
    distances: Optional[Sequence[float]] = None,
) -> float:
    """Return the maximum three-point XY curvature of a route prefix."""
    maximum = 0.0
    for index in range(1, len(locations) - 1):
        if (
            maximum_station is not None
            and distances is not None
            and index < len(distances)
            and float(distances[index]) > float(maximum_station)
        ):
            break
        first = locations[index - 1]
        middle = locations[index]
        last = locations[index + 1]
        side_a = planar_distance(first, middle)
        side_b = planar_distance(middle, last)
        side_c = planar_distance(first, last)
        denominator = side_a * side_b * side_c
        if denominator <= 1.0e-6:
            continue
        doubled_area = abs(
            float(middle.x - first.x) * float(last.y - first.y)
            - float(middle.y - first.y) * float(last.x - first.x)
        )
        curvature = 2.0 * doubled_area / denominator
        if math.isfinite(curvature):
            maximum = max(maximum, curvature)
    return maximum


def choose_straightest_waypoint(current_waypoint, candidates):
    if not candidates:
        return None
    try:
        current_forward = current_waypoint.transform.get_forward_vector()
    except (AttributeError, RuntimeError):
        return candidates[0]
    best = None
    best_alignment = None
    for candidate in candidates:
        try:
            if candidate.lane_type != carla.LaneType.Driving:
                continue
            forward = candidate.transform.get_forward_vector()
            alignment = (
                float(current_forward.x) * float(forward.x)
                + float(current_forward.y) * float(forward.y)
            )
        except (AttributeError, RuntimeError):
            continue
        if best_alignment is None or alignment > best_alignment:
            best = candidate
            best_alignment = alignment
    return best


def build_reactive_vehicle_route_plan(
    carla_map: carla.Map,
    requested_home: VehicleTarget,
    args: argparse.Namespace,
) -> ReactiveVehicleRoutePlan:
    """Plan the lane-centered blocker route once without mutating the world."""
    if GlobalRoutePlanner is None:
        raise ValueError(
            "CARLA GlobalRoutePlanner is unavailable; refusing unsafe direct "
            "reactive-vehicle motion"
        )
    raw_home = carla.Location(
        x=float(requested_home.x),
        y=float(requested_home.y),
        z=float(requested_home.z),
    )
    raw_attack = carla.Location(
        x=float(args.reactive_attack_location[0]),
        y=float(args.reactive_attack_location[1]),
        z=float(requested_home.z),
    )
    try:
        home_waypoint = carla_map.get_waypoint(
            raw_home,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        attack_waypoint = carla_map.get_waypoint(
            raw_attack,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except (AttributeError, RuntimeError) as exc:
        raise ValueError(
            "unable to project the reactive vehicle route onto Driving lanes: {}".format(
                exc
            )
        ) from exc
    if home_waypoint is None or attack_waypoint is None:
        raise ValueError(
            "reactive vehicle home and attack point both require Driving waypoints"
        )
    home_location = copy_carla_location(home_waypoint.transform.location)
    attack_lane_location = copy_carla_location(
        attack_waypoint.transform.location
    )
    home_snap_distance = planar_distance(raw_home, home_location)
    attack_lateral_offset = planar_distance(raw_attack, attack_lane_location)
    if home_snap_distance > args.reactive_route_home_projection_limit:
        raise ValueError(
            "nearest Driving waypoint is {:.2f} m from reactive vehicle home; "
            "limit is {:.2f} m".format(
                home_snap_distance,
                args.reactive_route_home_projection_limit,
            )
        )
    if attack_lateral_offset > args.reactive_route_attack_offset_limit:
        raise ValueError(
            "pedestrian attack point is {:.2f} m from its Driving-lane "
            "centerline; limit is {:.2f} m".format(
                attack_lateral_offset,
                args.reactive_route_attack_offset_limit,
            )
        )
    try:
        planner = GlobalRoutePlanner(
            carla_map,
            float(args.reactive_route_sampling_resolution),
        )
        route_trace = list(
            planner.trace_route(home_location, attack_lane_location)
        )
    except Exception as exc:
        raise ValueError(
            "unable to plan reactive Driving-lane route: {}".format(exc)
        ) from exc
    if not route_trace:
        raise ValueError("reactive Driving-lane route is unreachable")
    matching_attack_lane = any(
        int(waypoint.road_id) == int(attack_waypoint.road_id)
        and int(waypoint.section_id) == int(attack_waypoint.section_id)
        and int(waypoint.lane_id) == int(attack_waypoint.lane_id)
        for waypoint, _road_option in route_trace
    )
    if not matching_attack_lane:
        raise ValueError(
            "planned route does not enter the Driving lane containing the "
            "pedestrian attack point"
        )

    start_forward = home_waypoint.transform.get_forward_vector()
    route_locations: List[carla.Location] = [home_location]
    still_trimming_start = True
    for waypoint, _road_option in route_trace:
        location = waypoint.transform.location
        if still_trimming_start:
            forward_progress = (
                float(location.x - home_location.x) * float(start_forward.x)
                + float(location.y - home_location.y) * float(start_forward.y)
            )
            if forward_progress < -0.05:
                continue
            still_trimming_start = False
        route_locations.append(location)
    route_locations.append(attack_lane_location)
    deduped_to_attack = list(dedupe_route_locations(route_locations))
    if len(deduped_to_attack) < 2:
        raise ValueError("planned reactive route has fewer than two points")
    attack_distances = route_cumulative_distances(deduped_to_attack)
    attack_station = float(attack_distances[-1])
    if attack_station > args.reactive_route_max_length:
        raise ValueError(
            "reactive road route to the crosswalk is {:.2f} m, exceeding "
            "--reactive-route-max-length {:.2f} m; check lane direction".format(
                attack_station,
                args.reactive_route_max_length,
            )
        )

    required_extension = (
        float(args.reactive_intercept_route_window)
        + float(args.reactive_post_distance)
        + float(args.reactive_route_lookahead)
        + REACTIVE_ROUTE_DYNAMIC_LOOKAHEAD_MAX_M
        + 2.0 * float(args.reactive_route_sampling_resolution)
    )
    extension_travel = 0.0
    current_waypoint = attack_waypoint
    maximum_steps = max(
        20,
        int(math.ceil(required_extension / args.reactive_route_sampling_resolution))
        + 20,
    )
    for _ in range(maximum_steps):
        if extension_travel + 1.0e-6 >= required_extension:
            break
        try:
            candidates = current_waypoint.next(
                float(args.reactive_route_sampling_resolution)
            )
        except (AttributeError, RuntimeError):
            candidates = []
        next_waypoint = choose_straightest_waypoint(
            current_waypoint,
            candidates,
        )
        if next_waypoint is None:
            break
        next_location = next_waypoint.transform.location
        step_distance = planar_distance(
            current_waypoint.transform.location,
            next_location,
        )
        if step_distance <= 1.0e-4:
            break
        route_locations.append(next_location)
        extension_travel += step_distance
        current_waypoint = next_waypoint
    route_locations = list(dedupe_route_locations(route_locations))
    route_distances = route_cumulative_distances(route_locations)
    route_length = float(route_distances[-1])
    if route_length + 1.0e-6 < attack_station + required_extension:
        raise ValueError(
            "Driving-lane route ends only {:.2f} m after the crosswalk; "
            "{:.2f} m is required for runout and steering".format(
                route_length - attack_station,
                required_extension,
            )
        )

    attack_forward = attack_waypoint.transform.get_forward_vector()
    attack_direction = normalized_xy(
        float(attack_forward.x),
        float(attack_forward.y),
    )
    if attack_direction is None:
        raise ValueError("attack Driving waypoint has no usable forward tangent")
    curvature = route_maximum_curvature(
        route_locations,
        maximum_station=(
            attack_station
            + float(args.reactive_intercept_route_window)
            + float(args.reactive_post_distance)
            + float(args.reactive_route_lookahead)
            + REACTIVE_ROUTE_DYNAMIC_LOOKAHEAD_MAX_M
        ),
        distances=route_distances,
    )
    speed_limit = float(args.reactive_vehicle_speed_max)
    if curvature > 1.0e-6:
        speed_limit = min(
            speed_limit,
            math.sqrt(
                float(args.reactive_route_max_lateral_acceleration) / curvature
            ),
        )
    if speed_limit + 1.0e-9 < args.reactive_vehicle_speed_min:
        raise ValueError(
            "curve-safe reactive route speed {:.2f} m/s is below configured "
            "minimum {:.2f} m/s".format(
                speed_limit,
                args.reactive_vehicle_speed_min,
            )
        )
    home = VehicleTarget(
        x=float(home_location.x),
        y=float(home_location.y),
        z=float(home_location.z),
        yaw=float(home_waypoint.transform.rotation.yaw),
        source="{}+nearest-Driving-waypoint".format(requested_home.source),
        blueprint_id=requested_home.blueprint_id,
    )
    return ReactiveVehicleRoutePlan(
        requested_home=requested_home,
        home=home,
        locations=tuple(route_locations),
        distances=tuple(route_distances),
        attack_station=attack_station,
        route_length=route_length,
        attack_tangent_x=attack_direction[0],
        attack_tangent_y=attack_direction[1],
        home_snap_distance=home_snap_distance,
        attack_lateral_offset=attack_lateral_offset,
        maximum_curvature=curvature,
        speed_limit=speed_limit,
        home_road_id=int(home_waypoint.road_id),
        home_lane_id=int(home_waypoint.lane_id),
        attack_road_id=int(attack_waypoint.road_id),
        attack_lane_id=int(attack_waypoint.lane_id),
    )


def bounding_box_contains_location(actor, location: carla.Location) -> bool:
    """Return whether a world location lies inside an actor's oriented box."""
    try:
        return bool(
            actor.bounding_box.contains(
                location,
                actor.get_transform(),
            )
        )
    except (AttributeError, RuntimeError):
        return False


def convex_hull_xy(
    points: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], ...]:
    """Return the counter-clockwise convex hull of projected XY points."""
    unique_points = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique_points) <= 1:
        return tuple(unique_points)

    def cross(
        origin: Tuple[float, float],
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: List[Tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-9:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-9:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def actor_box_geometry(actor):
    """Return an actor's projected convex footprint and world Z interval."""
    try:
        vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
    except (AttributeError, RuntimeError):
        return None
    if not vertices:
        return None
    footprint = convex_hull_xy(
        [(float(vertex.x), float(vertex.y)) for vertex in vertices]
    )
    if len(footprint) < 3:
        return None
    z_values = [float(vertex.z) for vertex in vertices]
    return footprint, min(z_values), max(z_values)


def actor_planar_bounding_radius(actor) -> Optional[float]:
    """Return a conservative local XY radius for cheap contact rejection."""
    try:
        box = actor.bounding_box
        return math.hypot(
            float(box.location.x),
            float(box.location.y),
        ) + math.hypot(
            float(box.extent.x),
            float(box.extent.y),
        )
    except (AttributeError, RuntimeError):
        return None


def convex_footprints_overlap(
    first: Sequence[Tuple[float, float]],
    second: Sequence[Tuple[float, float]],
    margin: float = 0.0,
) -> bool:
    """Test two convex XY polygons with the separating-axis theorem."""
    total_margin = max(0.0, float(margin))
    for polygon in (first, second):
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            edge_x = end[0] - start[0]
            edge_y = end[1] - start[1]
            magnitude = math.hypot(edge_x, edge_y)
            if magnitude <= 1e-9:
                continue
            axis_x = -edge_y / magnitude
            axis_y = edge_x / magnitude
            first_projection = [
                point[0] * axis_x + point[1] * axis_y for point in first
            ]
            second_projection = [
                point[0] * axis_x + point[1] * axis_y for point in second
            ]
            if (
                max(first_projection) + total_margin < min(second_projection)
                or max(second_projection) + total_margin < min(first_projection)
            ):
                return False
    return True


def vertical_intervals_overlap(
    first_minimum: float,
    first_maximum: float,
    second_minimum: float,
    second_maximum: float,
    margin: float = 0.0,
) -> bool:
    total_margin = max(0.0, float(margin))
    return not (
        first_maximum + total_margin < second_minimum
        or second_maximum + total_margin < first_minimum
    )


def actor_bounding_boxes_overlap(
    first_actor,
    second_actor,
    horizontal_margin: float = 0.0,
    vertical_margin: float = 0.0,
) -> bool:
    """Return whether two actor boxes overlap in projected XY and world Z."""
    first_geometry = actor_box_geometry(first_actor)
    second_geometry = actor_box_geometry(second_actor)
    if first_geometry is None or second_geometry is None:
        return False
    first_footprint, first_min_z, first_max_z = first_geometry
    second_footprint, second_min_z, second_max_z = second_geometry
    return vertical_intervals_overlap(
        first_min_z,
        first_max_z,
        second_min_z,
        second_max_z,
        vertical_margin,
    ) and convex_footprints_overlap(
        first_footprint,
        second_footprint,
        horizontal_margin,
    )


def point_inside_convex_footprint(
    point: Tuple[float, float],
    footprint: Sequence[Tuple[float, float]],
) -> bool:
    """Return whether an XY point is inside or on a convex polygon."""
    sign = 0
    for index, start in enumerate(footprint):
        end = footprint[(index + 1) % len(footprint)]
        cross = (
            (end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])
        )
        if abs(cross) <= 1e-9:
            continue
        current_sign = 1 if cross > 0.0 else -1
        if sign == 0:
            sign = current_sign
        elif current_sign != sign:
            return False
    return True


def point_to_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    edge_x = end[0] - start[0]
    edge_y = end[1] - start[1]
    length_squared = edge_x * edge_x + edge_y * edge_y
    if length_squared <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * edge_x
        + (point[1] - start[1]) * edge_y
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    nearest_x = start[0] + projection * edge_x
    nearest_y = start[1] + projection * edge_y
    return math.hypot(point[0] - nearest_x, point[1] - nearest_y)


def point_to_footprint_distance(
    point: Tuple[float, float],
    footprint: Sequence[Tuple[float, float]],
) -> float:
    if point_inside_convex_footprint(point, footprint):
        return 0.0
    return min(
        point_to_segment_distance(
            point,
            footprint[index],
            footprint[(index + 1) % len(footprint)],
        )
        for index in range(len(footprint))
    )


def actor_footprint_near_location(
    actor,
    location: carla.Location,
    horizontal_clearance: float,
    vertical_clearance: float,
) -> Optional[bool]:
    """Test a target point against an expanded actor footprint and Z range."""
    geometry = actor_box_geometry(actor)
    if geometry is None:
        try:
            actor_location = actor.get_location()
        except (AttributeError, RuntimeError):
            return None
        return (
            abs(float(actor_location.z - location.z))
            <= max(0.0, float(vertical_clearance))
            and planar_distance(actor_location, location)
            <= max(0.0, float(horizontal_clearance))
        )
    footprint, minimum_z, maximum_z = geometry
    if not (
        minimum_z - max(0.0, float(vertical_clearance))
        <= float(location.z)
        <= maximum_z + max(0.0, float(vertical_clearance))
    ):
        return False
    return point_to_footprint_distance(
        (float(location.x), float(location.y)),
        footprint,
    ) <= max(0.0, float(horizontal_clearance))


def blocking_actor_at_location(
    world: carla.World,
    location: carla.Location,
    clearance: float,
    ignored_ids: Sequence[int] = (),
):
    """Find a vehicle/walker whose box or center blocks a target location."""
    ignored = {int(actor_id) for actor_id in ignored_ids}
    try:
        actors = world.get_actors()
    except RuntimeError as exc:
        raise RuntimeError(
            "actor inventory is unavailable during target occupancy check"
        ) from exc
    for actor in actors:
        try:
            if int(actor.id) in ignored or not actor.is_alive:
                continue
            if not (
                str(actor.type_id).startswith("vehicle.")
                or str(actor.type_id).startswith("walker.")
            ):
                continue
            proximity = actor_footprint_near_location(
                actor,
                location,
                clearance,
                DEFAULT_SPAWN_OCCUPANCY_VERTICAL_CLEARANCE_M,
            )
            if proximity is None:
                LOG.warning(
                    "Treating actor id=%d type=%s as occupying a target "
                    "because its geometry is temporarily unavailable",
                    actor.id,
                    actor.type_id,
                )
                return actor
            if proximity:
                return actor
        except (AttributeError, RuntimeError):
            continue
    return None


def clamped_acceleration_xy(
    acceleration: carla.Vector3D,
    maximum_magnitude: float,
) -> Tuple[float, float]:
    """Return a finite, magnitude-limited XY acceleration sample."""
    try:
        acceleration_x = float(acceleration.x)
        acceleration_y = float(acceleration.y)
    except (AttributeError, TypeError, ValueError):
        return 0.0, 0.0
    if not math.isfinite(acceleration_x) or not math.isfinite(acceleration_y):
        return 0.0, 0.0
    magnitude = math.hypot(acceleration_x, acceleration_y)
    limit = max(0.0, float(maximum_magnitude))
    if magnitude > limit > 0.0:
        scale = limit / magnitude
        acceleration_x *= scale
        acceleration_y *= scale
    return acceleration_x, acceleration_y


def positive_radial_arrival_time(
    distance: float,
    closing_speed: float,
    radial_acceleration: float,
) -> Optional[float]:
    """Solve 0.5*a*t^2 + v*t = distance for the earliest inbound root."""
    distance_value = float(distance)
    speed_value = float(closing_speed)
    acceleration_value = float(radial_acceleration)
    if not all(
        math.isfinite(value)
        for value in (distance_value, speed_value, acceleration_value)
    ):
        return None
    if distance_value <= 1.0e-6:
        return 0.0
    if speed_value <= 1.0e-9:
        return None
    if abs(acceleration_value) <= 1.0e-6:
        return distance_value / speed_value
    discriminant = (
        speed_value * speed_value
        + 2.0 * acceleration_value * distance_value
    )
    if discriminant < 0.0:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates = []
    for numerator in (-speed_value + root, -speed_value - root):
        arrival_time = numerator / acceleration_value
        if arrival_time <= 1.0e-6 or not math.isfinite(arrival_time):
            continue
        # A braking prediction has a second mathematical root after reversing.
        # A pedestrian does not reverse through the target in this model.
        arrival_speed = speed_value + acceleration_value * arrival_time
        if arrival_speed < -1.0e-6:
            continue
        candidates.append(arrival_time)
    return min(candidates) if candidates else None


def evaluate_pedestrian_target_approach(
    location: carla.Location,
    velocity: carla.Vector3D,
    acceleration: carla.Vector3D,
    target_x: float,
    target_y: float,
    trigger_distance: float,
    minimum_speed: float,
    minimum_closing_speed: float,
    maximum_cross_track_miss: float,
    minimum_eta: float,
    maximum_eta: float,
    acceleration_limit: float,
) -> Optional[PedestrianTargetApproach]:
    """Validate an inbound manual pedestrian and estimate its target ETA."""
    relative_x = float(target_x) - float(location.x)
    relative_y = float(target_y) - float(location.y)
    distance = math.hypot(relative_x, relative_y)
    if not math.isfinite(distance) or distance > float(trigger_distance):
        return None
    target_direction = normalized_xy(relative_x, relative_y)
    if target_direction is None:
        return None
    velocity_x = float(velocity.x)
    velocity_y = float(velocity.y)
    speed = math.hypot(velocity_x, velocity_y)
    if not math.isfinite(speed) or speed < float(minimum_speed):
        return None
    closing_speed = (
        velocity_x * target_direction[0]
        + velocity_y * target_direction[1]
    )
    if closing_speed < float(minimum_closing_speed):
        return None

    acceleration_xy = clamped_acceleration_xy(
        acceleration,
        acceleration_limit,
    )
    radial_acceleration = (
        acceleration_xy[0] * target_direction[0]
        + acceleration_xy[1] * target_direction[1]
    )
    eta = positive_radial_arrival_time(
        distance,
        closing_speed,
        radial_acceleration,
    )
    if eta is None:
        return None
    if eta < float(minimum_eta) or eta > float(maximum_eta):
        return None
    predicted_x = (
        float(location.x)
        + velocity_x * eta
        + 0.5 * acceleration_xy[0] * eta * eta
    )
    predicted_y = (
        float(location.y)
        + velocity_y * eta
        + 0.5 * acceleration_xy[1] * eta * eta
    )
    cross_track_miss = math.hypot(
        predicted_x - float(target_x),
        predicted_y - float(target_y),
    )
    if cross_track_miss > float(maximum_cross_track_miss):
        return None
    movement_direction = normalized_xy(velocity_x, velocity_y)
    if movement_direction is None:
        return None
    return PedestrianTargetApproach(
        distance=distance,
        speed=speed,
        closing_speed=closing_speed,
        cross_track_miss=cross_track_miss,
        eta_seconds=eta,
        direction_x=movement_direction[0],
        direction_y=movement_direction[1],
        radial_acceleration=radial_acceleration,
        target_x=float(target_x),
        target_y=float(target_y),
    )


def cross_product_xy(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
) -> float:
    return float(first_x) * float(second_y) - float(first_y) * float(second_x)


def evaluate_pedestrian_route_crossing(
    location: carla.Location,
    velocity: carla.Vector3D,
    acceleration: carla.Vector3D,
    route_locations: Sequence[carla.Location],
    route_distances: Sequence[float],
    nominal_station: float,
    station_window: float,
    minimum_route_station: float,
    trigger_distance: float,
    minimum_speed: float,
    minimum_closing_speed: float,
    maximum_cross_track_miss: float,
    minimum_eta: float,
    maximum_eta: float,
    acceleration_limit: float,
    minimum_crossing_angle_degrees: float,
    allow_late_best_effort: bool,
) -> Optional[PedestrianTargetApproach]:
    """Predict where the pedestrian motion ray crosses the Driving route.

    The vehicle route remains immutable and lane-centred.  Only the encounter
    station is selected dynamically, which permits a user to take a nearby
    part of the same crosswalk without making the vehicle chase onto a
    sidewalk.  The returned target is latched when the attack starts.
    """
    count = min(len(route_locations), len(route_distances))
    if count < 2:
        return None
    velocity_x = float(velocity.x)
    velocity_y = float(velocity.y)
    speed = math.hypot(velocity_x, velocity_y)
    if not math.isfinite(speed) or speed < float(minimum_speed):
        return None
    direction = normalized_xy(velocity_x, velocity_y)
    if direction is None:
        return None
    direction_x, direction_y = direction
    acceleration_xy = clamped_acceleration_xy(acceleration, acceleration_limit)
    radial_acceleration = (
        acceleration_xy[0] * direction_x
        + acceleration_xy[1] * direction_y
    )
    lateral_acceleration = cross_product_xy(
        direction_x,
        direction_y,
        acceleration_xy[0],
        acceleration_xy[1],
    )
    window_start = max(
        0.0,
        float(nominal_station) - max(0.0, float(station_window)),
        float(minimum_route_station),
    )
    window_end = min(
        float(route_distances[count - 1]),
        float(nominal_station) + max(0.0, float(station_window)),
    )
    if window_end + 1.0e-9 < window_start:
        return None

    point_x = float(location.x)
    point_y = float(location.y)
    candidates: List[PedestrianTargetApproach] = []
    for index in range(count - 1):
        segment_station_start = float(route_distances[index])
        segment_station_end = float(route_distances[index + 1])
        if segment_station_end < window_start - 1.0e-9:
            continue
        if segment_station_start > window_end + 1.0e-9:
            break
        start = route_locations[index]
        end = route_locations[index + 1]
        segment_x = float(end.x - start.x)
        segment_y = float(end.y - start.y)
        segment_length = math.hypot(segment_x, segment_y)
        if segment_length <= 1.0e-9:
            continue
        clip_start = max(segment_station_start, window_start)
        clip_end = min(segment_station_end, window_end)
        if clip_end + 1.0e-9 < clip_start:
            continue
        start_fraction = (
            clip_start - segment_station_start
        ) / segment_length
        end_fraction = (
            clip_end - segment_station_start
        ) / segment_length
        clipped_start_x = float(start.x) + start_fraction * segment_x
        clipped_start_y = float(start.y) + start_fraction * segment_y
        clipped_x = (end_fraction - start_fraction) * segment_x
        clipped_y = (end_fraction - start_fraction) * segment_y
        clipped_length = math.hypot(clipped_x, clipped_y)
        if clipped_length <= 1.0e-9:
            continue
        denominator = cross_product_xy(
            direction_x,
            direction_y,
            clipped_x,
            clipped_y,
        )
        if abs(denominator) <= 1.0e-8:
            continue
        relative_start_x = clipped_start_x - point_x
        relative_start_y = clipped_start_y - point_y
        ray_distance = cross_product_xy(
            relative_start_x,
            relative_start_y,
            clipped_x,
            clipped_y,
        ) / denominator
        segment_fraction = cross_product_xy(
            relative_start_x,
            relative_start_y,
            direction_x,
            direction_y,
        ) / denominator
        if ray_distance < -1.0e-6 or not -1.0e-6 <= segment_fraction <= 1.000001:
            continue
        ray_distance = max(0.0, ray_distance)
        if ray_distance > float(trigger_distance):
            continue
        segment_fraction = max(0.0, min(1.0, segment_fraction))
        route_tangent_x = segment_x / segment_length
        route_tangent_y = segment_y / segment_length
        crossing_angle = math.degrees(
            math.acos(
                max(
                    0.0,
                    min(
                        1.0,
                        abs(
                            direction_x * route_tangent_x
                            + direction_y * route_tangent_y
                        ),
                    ),
                )
            )
        )
        if crossing_angle < float(minimum_crossing_angle_degrees):
            continue
        if speed < float(minimum_closing_speed):
            continue
        eta = positive_radial_arrival_time(
            ray_distance,
            speed,
            radial_acceleration,
        )
        if eta is None or eta > float(maximum_eta):
            continue
        if eta < float(minimum_eta) and not allow_late_best_effort:
            continue
        cross_track_miss = abs(0.5 * lateral_acceleration * eta * eta)
        if cross_track_miss > float(maximum_cross_track_miss):
            continue
        target_x = point_x + direction_x * ray_distance
        target_y = point_y + direction_y * ray_distance
        route_station = clip_start + segment_fraction * clipped_length
        candidates.append(
            PedestrianTargetApproach(
                distance=ray_distance,
                speed=speed,
                closing_speed=speed,
                cross_track_miss=cross_track_miss,
                eta_seconds=max(0.0, eta),
                direction_x=direction_x,
                direction_y=direction_y,
                radial_acceleration=radial_acceleration,
                target_x=target_x,
                target_y=target_y,
                route_station=route_station,
                route_tangent_x=route_tangent_x,
                route_tangent_y=route_tangent_y,
                crossing_angle_degrees=crossing_angle,
            )
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda approach: (
            approach.eta_seconds,
            abs(approach.route_station - float(nominal_station)),
        ),
    )


def minimum_vehicle_travel_time(
    distance: float,
    current_speed: float,
    maximum_speed: float,
    maximum_acceleration: float,
) -> float:
    """Optimistic lower bound for a forward vehicle to cover ``distance``."""
    remaining = max(0.0, float(distance))
    speed = max(0.0, min(float(current_speed), float(maximum_speed)))
    speed_limit = max(1.0e-6, float(maximum_speed))
    acceleration = max(1.0e-6, float(maximum_acceleration))
    if remaining <= 0.0:
        return 0.0
    if speed >= speed_limit - 1.0e-9:
        return remaining / speed_limit
    acceleration_time = (speed_limit - speed) / acceleration
    acceleration_distance = (
        speed * acceleration_time
        + 0.5 * acceleration * acceleration_time * acceleration_time
    )
    if remaining <= acceleration_distance:
        return (
            -speed + math.sqrt(speed * speed + 2.0 * acceleration * remaining)
        ) / acceleration
    return acceleration_time + (
        remaining - acceleration_distance
    ) / speed_limit


def reactive_vehicle_timing_window(
    front_remaining: float,
    current_speed: float,
    pedestrian_eta: float,
    maximum_speed: float,
    args: argparse.Namespace,
) -> Tuple[bool, float, float]:
    """Return whether the staged vehicle should move for the current ETA."""
    remaining = max(0.0, float(front_remaining))
    speed_limit = max(1.0e-6, float(maximum_speed))
    if args.reactive_vehicle_control == "constant-velocity":
        minimum_time = remaining / speed_limit
    else:
        minimum_time = minimum_vehicle_travel_time(
            remaining,
            current_speed,
            speed_limit,
            args.reactive_vehicle_max_acceleration,
        )
    available_time = max(
        0.0,
        float(pedestrian_eta) - float(args.reactive_impact_lead),
    )
    ready = available_time <= (
        minimum_time + float(args.reactive_launch_margin)
    )
    return ready, minimum_time, available_time


def reactive_vehicle_commit_ready(
    front_remaining: float,
    current_speed: float,
    pedestrian_eta: float,
    maximum_speed: float,
    args: argparse.Namespace,
) -> bool:
    """Decide when the final non-braking collision approach is justified."""
    remaining = float(front_remaining)
    if remaining <= 1.0e-6:
        return False
    timing_ready, _minimum_time, _available_time = (
        reactive_vehicle_timing_window(
            remaining,
            current_speed,
            pedestrian_eta,
            maximum_speed,
            args,
        )
    )
    if not timing_ready:
        return False
    stopping_distance = (
        max(0.0, float(current_speed)) ** 2
        / (2.0 * max(1.0e-6, float(args.reactive_vehicle_max_deceleration)))
        + max(0.0, float(current_speed))
        * max(0.0, float(args.reactive_impact_lead))
    )
    no_return_distance = (
        float(args.reactive_approach_hold_distance) + stopping_distance
    )
    return remaining <= max(
        float(args.reactive_commit_distance),
        no_return_distance,
    )


def reactive_vehicle_approach_hold_command(
    front_remaining: float,
    current_speed: float,
    hold_distance: float,
    maximum_deceleration: float,
) -> ReactiveVehicleCommand:
    """Brake toward an upstream front-bumper hold line without teleporting."""
    speed_now = max(0.0, float(current_speed))
    distance_to_hold = max(
        0.0,
        float(front_remaining) - max(0.0, float(hold_distance)),
    )
    if speed_now <= 1.0e-3:
        acceleration = 0.0
    elif distance_to_hold <= 1.0e-3:
        acceleration = -max(1.0e-6, float(maximum_deceleration))
    else:
        required_deceleration = speed_now * speed_now / (
            2.0 * distance_to_hold
        )
        acceleration = -min(
            max(1.0e-6, float(maximum_deceleration)),
            required_deceleration,
        )
    return ReactiveVehicleCommand(
        speed=0.0,
        acceleration=acceleration,
        time_available=0.0,
    )


def synchronized_vehicle_command(
    front_remaining: float,
    current_speed: float,
    pedestrian_eta: float,
    impact_lead: float,
    minimum_speed: float,
    maximum_speed: float,
    maximum_acceleration: float,
    maximum_deceleration: float,
    commit_distance: float,
    commit_speed: float,
    control_mode: str,
    commit_ready: bool = False,
) -> ReactiveVehicleCommand:
    """Return a bounded speed/acceleration command for simultaneous arrival.

    Minimum/commit speed floors are intentionally disabled until a fresh
    pedestrian ETA makes the final approach ready.  Inside that committed
    zone the floor ramps continuously from zero at its upstream boundary to
    the configured impact speed at contact, avoiding the old 1-to-5 m/s step.
    """
    remaining = max(0.0, float(front_remaining))
    speed_now = max(0.0, float(current_speed))
    speed_limit = max(1.0e-6, float(maximum_speed))
    acceleration_limit = max(1.0e-6, float(maximum_acceleration))
    deceleration_limit = max(1.0e-6, float(maximum_deceleration))
    time_available = max(
        0.05,
        float(pedestrian_eta) - max(0.0, float(impact_lead)),
    )
    if control_mode == "constant-velocity":
        desired_speed = remaining / time_available
        acceleration = 0.0
    else:
        unconstrained_acceleration = 2.0 * (
            remaining - speed_now * time_available
        ) / (time_available * time_available)
        unconstrained_final_speed = (
            speed_now + unconstrained_acceleration * time_available
        )
        if speed_now >= speed_limit:
            desired_speed = speed_limit
            acceleration = max(
                -deceleration_limit,
                (desired_speed - speed_now) / time_available,
            )
        elif unconstrained_final_speed > speed_limit:
            # Accelerate to vmax, then cruise. Solving the trapezoidal
            # distance profile avoids the late arrival caused by merely
            # clamping an unconstrained terminal speed.
            denominator = 2.0 * (
                speed_limit * time_available - remaining
            )
            if denominator > 1.0e-9:
                acceleration = (
                    (speed_limit - speed_now) ** 2 / denominator
                )
            else:
                acceleration = acceleration_limit
            acceleration = min(acceleration_limit, max(0.0, acceleration))
            desired_speed = speed_limit
        elif unconstrained_final_speed < 0.0 and speed_now > 1.0e-6:
            # Stop at the target rather than following the nonphysical second
            # half of a constant-deceleration trajectory into reverse.
            required_deceleration = (
                speed_now * speed_now / max(2.0 * remaining, 1.0e-9)
            )
            acceleration = -min(deceleration_limit, required_deceleration)
            desired_speed = 0.0
        else:
            acceleration = max(
                -deceleration_limit,
                min(acceleration_limit, unconstrained_acceleration),
            )
            desired_speed = speed_now + acceleration * time_available
    desired_speed = max(0.0, min(speed_limit, desired_speed))
    unclamped_speed = desired_speed
    if (
        commit_ready
        and float(commit_distance) > 1.0e-6
        and remaining <= float(commit_distance)
    ):
        commit_progress = max(
            0.0,
            min(
                1.0,
                1.0 - remaining / float(commit_distance),
            ),
        )
        committed_speed_floor = commit_progress * max(
            float(minimum_speed),
            float(commit_speed),
        )
        desired_speed = max(desired_speed, committed_speed_floor)
    desired_speed = min(speed_limit, desired_speed)
    if control_mode != "constant-velocity" and abs(
        desired_speed - unclamped_speed
    ) > 1.0e-9:
        acceleration = (
            desired_speed - speed_now
        ) / time_available
        acceleration = max(
            -deceleration_limit,
            min(acceleration_limit, acceleration),
        )
    return ReactiveVehicleCommand(
        speed=desired_speed,
        acceleration=acceleration,
        time_available=time_available,
    )


def bounded_acceleration_toward_speed(
    current_speed: float,
    target_speed: float,
    maximum_acceleration: float,
    maximum_deceleration: float,
    horizon_seconds: float = 1.0,
) -> float:
    acceleration = (
        float(target_speed) - max(0.0, float(current_speed))
    ) / max(0.05, float(horizon_seconds))
    return max(
        -float(maximum_deceleration),
        min(float(maximum_acceleration), acceleration),
    )


def predicted_ego_xy(
    ego_location: carla.Location,
    ego_velocity: carla.Vector3D,
    acceleration_xy: Tuple[float, float],
    time_seconds: float,
) -> Tuple[float, float]:
    time_value = float(time_seconds)
    return (
        float(ego_location.x)
        + float(ego_velocity.x) * time_value
        + 0.5 * acceleration_xy[0] * time_value * time_value,
        float(ego_location.y)
        + float(ego_velocity.y) * time_value
        + 0.5 * acceleration_xy[1] * time_value * time_value,
    )


def predicted_ego_path_distance(
    ego_velocity: carla.Vector3D,
    acceleration_xy: Tuple[float, float],
    time_seconds: float,
) -> float:
    """Numerically integrate planar speed over a short prediction horizon."""
    duration = max(0.0, float(time_seconds))
    if duration <= 0.0:
        return 0.0
    sample_count = max(8, min(128, int(math.ceil(duration / 0.05))))
    step = duration / sample_count

    def speed_at(sample_time: float) -> float:
        return math.hypot(
            float(ego_velocity.x) + acceleration_xy[0] * sample_time,
            float(ego_velocity.y) + acceleration_xy[1] * sample_time,
        )

    distance = 0.5 * (speed_at(0.0) + speed_at(duration))
    for index in range(1, sample_count):
        distance += speed_at(index * step)
    return distance * step


def ego_contact_reference_line_value(
    ego_location: carla.Location,
    ego_velocity: carla.Vector3D,
    line_origin: Tuple[float, float],
    line_normal: Tuple[float, float],
    front_contact_offset: float,
) -> float:
    """Return signed contact-reference distance across a directed line.

    Negative is before the line and non-negative means that the configured
    front reference has reached or passed it. A stopped ego uses the locked
    line normal so hard-braking near-miss checks retain the vehicle support.
    """
    tangent_direction = normalized_xy(
        float(ego_velocity.x),
        float(ego_velocity.y),
    ) or line_normal
    return (
        (float(ego_location.x) - line_origin[0]) * line_normal[0]
        + (float(ego_location.y) - line_origin[1]) * line_normal[1]
        + max(0.0, float(front_contact_offset))
        * (
            tangent_direction[0] * line_normal[0]
            + tangent_direction[1] * line_normal[1]
        )
    )


def solve_intercept(
    ego_location: carla.Location,
    ego_velocity: carla.Vector3D,
    ego_acceleration: carla.Vector3D,
    pedestrian_location: carla.Location,
    minimum_pedestrian_speed: float,
    maximum_pedestrian_speed: float,
    minimum_time: float,
    maximum_time: float,
    acceleration_limit: float,
    front_contact_offset: float = 0.0,
) -> Optional[InterceptSolution]:
    """Find the first acceleration-aware perpendicular XY intercept.

    With ``X(t) = E + V*t + 0.5*A*t^2``, a line from pedestrian P
    to the ego front reference ``F(t) = X(t) + D*unit(V + A*t)`` is
    perpendicular to the predicted ego tangent when
    ``(X(t) - P) dot (V + A*t) + D*|V + A*t| == 0``. ``D=0`` preserves
    center-target behavior. The root is found by bounded scanning and
    bisection so the implementation has no NumPy dependency.
    """
    velocity_x = float(ego_velocity.x)
    velocity_y = float(ego_velocity.y)
    initial_speed = math.hypot(velocity_x, velocity_y)
    initial_direction = normalized_xy(velocity_x, velocity_y)
    if initial_direction is None or initial_speed <= 1.0e-6:
        return None
    contact_offset = float(front_contact_offset)
    if not math.isfinite(contact_offset) or contact_offset < 0.0:
        return None
    acceleration_xy = clamped_acceleration_xy(
        ego_acceleration,
        acceleration_limit,
    )
    difference_x = float(ego_location.x - pedestrian_location.x)
    difference_y = float(ego_location.y - pedestrian_location.y)

    def perpendicular_function(time_value: float) -> float:
        predicted_x, predicted_y = predicted_ego_xy(
            ego_location,
            ego_velocity,
            acceleration_xy,
            time_value,
        )
        tangent_x = velocity_x + acceleration_xy[0] * time_value
        tangent_y = velocity_y + acceleration_xy[1] * time_value
        tangent_speed = math.hypot(tangent_x, tangent_y)
        return (
            (predicted_x - float(pedestrian_location.x)) * tangent_x
            + (predicted_y - float(pedestrian_location.y)) * tangent_y
            + contact_offset * tangent_speed
        )

    # A negative derivative of squared distance means the predicted ego is
    # approaching the pedestrian's perpendicular projection.
    previous_time = 0.0
    previous_value = (
        difference_x * velocity_x
        + difference_y * velocity_y
        + contact_offset * initial_speed
    )
    if not math.isfinite(previous_value) or previous_value >= 0.0:
        return None
    horizon = float(maximum_time)
    step_count = max(64, min(512, int(math.ceil(horizon / 0.025))))
    for step_index in range(1, step_count + 1):
        current_time = horizon * step_index / step_count
        current_value = perpendicular_function(current_time)
        if not math.isfinite(current_value):
            return None
        if previous_value < 0.0 <= current_value:
            lower = previous_time
            upper = current_time
            for _ in range(48):
                midpoint = 0.5 * (lower + upper)
                if perpendicular_function(midpoint) < 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
            intercept_time = 0.5 * (lower + upper)
            previous_time = current_time
            previous_value = current_value
            if intercept_time < float(minimum_time):
                continue

            ego_center_x, ego_center_y = predicted_ego_xy(
                ego_location,
                ego_velocity,
                acceleration_xy,
                intercept_time,
            )
            tangent_x = velocity_x + acceleration_xy[0] * intercept_time
            tangent_y = velocity_y + acceleration_xy[1] * intercept_time
            tangent_direction = normalized_xy(tangent_x, tangent_y)
            if tangent_direction is None:
                continue
            # Reject the nonphysical branch of a braking forecast after the
            # constant-acceleration polynomial has reversed the vehicle.
            if (
                tangent_x * initial_direction[0]
                + tangent_y * initial_direction[1]
                <= 0.05
            ):
                continue
            # L2 represents the pedestrian-center position at scheduled front
            # contact. Moving it ahead of the ego origin by the stored support
            # distance fixes the former center-timing/side-impact mismatch.
            target_x = ego_center_x + contact_offset * tangent_direction[0]
            target_y = ego_center_y + contact_offset * tangent_direction[1]
            pedestrian_delta_x = target_x - float(pedestrian_location.x)
            pedestrian_delta_y = target_y - float(pedestrian_location.y)
            pedestrian_distance = math.hypot(
                pedestrian_delta_x,
                pedestrian_delta_y,
            )
            pedestrian_direction = normalized_xy(
                pedestrian_delta_x,
                pedestrian_delta_y,
            )
            if pedestrian_direction is None:
                continue
            perpendicular_residual = abs(
                pedestrian_direction[0] * tangent_direction[0]
                + pedestrian_direction[1] * tangent_direction[1]
            )
            if perpendicular_residual > 1.0e-4:
                continue
            required_speed = pedestrian_distance / intercept_time
            if not (
                float(minimum_pedestrian_speed)
                <= required_speed
                <= float(maximum_pedestrian_speed)
            ):
                continue
            return InterceptSolution(
                time_seconds=intercept_time,
                target_x=target_x,
                target_y=target_y,
                tangent_x=tangent_direction[0],
                tangent_y=tangent_direction[1],
                pedestrian_direction_x=pedestrian_direction[0],
                pedestrian_direction_y=pedestrian_direction[1],
                required_pedestrian_speed=required_speed,
                pedestrian_distance=pedestrian_distance,
                ego_travel=predicted_ego_path_distance(
                    ego_velocity,
                    acceleration_xy,
                    intercept_time,
                ),
                acceleration_x=acceleration_xy[0],
                acceleration_y=acceleration_xy[1],
                front_contact_offset=contact_offset,
            )
        previous_time = current_time
        previous_value = current_value
    return None


def solve_crossing_line_intercept(
    ego_location: carla.Location,
    ego_velocity: carla.Vector3D,
    acceleration_xy: Tuple[float, float],
    pedestrian_location: carla.Location,
    crossing_origin: Tuple[float, float],
    crossing_direction: Tuple[float, float],
    ego_path_direction: Tuple[float, float],
    maximum_time: float,
    front_contact_offset: float = 0.0,
) -> Optional[LineInterceptUpdate]:
    """Predict the ego front reference's next locked-line intersection."""
    normal_x, normal_y = ego_path_direction
    contact_offset = float(front_contact_offset)
    if not math.isfinite(contact_offset) or contact_offset < 0.0:
        return None
    center_constant = (
        (float(ego_location.x) - crossing_origin[0]) * normal_x
        + (float(ego_location.y) - crossing_origin[1]) * normal_y
    )
    linear = (
        float(ego_velocity.x) * normal_x
        + float(ego_velocity.y) * normal_y
    )
    quadratic_acceleration = (
        acceleration_xy[0] * normal_x + acceleration_xy[1] * normal_y
    )
    roots: List[float] = []
    epsilon = 1.0e-9
    if contact_offset <= epsilon:
        # Preserve the original center-target quadratic exactly.
        if abs(quadratic_acceleration) <= epsilon:
            if linear <= epsilon:
                return None
            roots.append(-center_constant / linear)
        else:
            discriminant = (
                linear * linear
                - 2.0 * quadratic_acceleration * center_constant
            )
            if discriminant < 0.0:
                return None
            square_root = math.sqrt(max(0.0, discriminant))
            roots.extend(
                (
                    (-linear - square_root) / quadratic_acceleration,
                    (-linear + square_root) / quadratic_acceleration,
                )
            )
    else:
        # The vehicle's front reference follows its predicted tangent. On a
        # turning trajectory its projection onto the locked line normal is
        # D*dot(unit(V+A*t), normal), not a fixed +D. Scan/bisect this bounded
        # smooth function so the active plan stays consistent with activation.
        def contact_line_value(time_value: float) -> Optional[float]:
            predicted_x, predicted_y = predicted_ego_xy(
                ego_location,
                ego_velocity,
                acceleration_xy,
                time_value,
            )
            tangent_direction = normalized_xy(
                float(ego_velocity.x) + acceleration_xy[0] * time_value,
                float(ego_velocity.y) + acceleration_xy[1] * time_value,
            )
            if tangent_direction is None:
                return None
            return (
                (predicted_x - crossing_origin[0]) * normal_x
                + (predicted_y - crossing_origin[1]) * normal_y
                + contact_offset
                * (
                    tangent_direction[0] * normal_x
                    + tangent_direction[1] * normal_y
                )
            )

        previous_time = 0.0
        previous_value = contact_line_value(previous_time)
        if previous_value is None or previous_value >= 0.0:
            return None
        horizon = float(maximum_time)
        step_count = max(64, min(512, int(math.ceil(horizon / 0.025))))
        sampled_values: List[Tuple[float, float]] = [
            (previous_time, previous_value)
        ]
        for step_index in range(1, step_count + 1):
            current_time = horizon * step_index / step_count
            current_value = contact_line_value(current_time)
            if current_value is None or not math.isfinite(current_value):
                return None
            sampled_values.append((current_time, current_value))
            if previous_value < 0.0 <= current_value:
                lower = previous_time
                upper = current_time
                for _ in range(48):
                    midpoint = 0.5 * (lower + upper)
                    midpoint_value = contact_line_value(midpoint)
                    if midpoint_value is None:
                        return None
                    if midpoint_value < 0.0:
                        lower = midpoint
                    else:
                        upper = midpoint
                roots.append(0.5 * (lower + upper))
            previous_time = current_time
            previous_value = current_value
        if not roots:
            # A turning/braking trajectory can touch the locked plane in a
            # very narrow interval and return to the same side between 25 ms
            # samples. Refine around the best sampled local maximum before
            # declaring the encounter infeasible.
            peak_index = max(
                range(len(sampled_values)),
                key=lambda index: sampled_values[index][1],
            )
            if 0 < peak_index < len(sampled_values) - 1:
                lower_time = sampled_values[peak_index - 1][0]
                upper_time = sampled_values[peak_index + 1][0]
                for _ in range(48):
                    first_third = (2.0 * lower_time + upper_time) / 3.0
                    second_third = (lower_time + 2.0 * upper_time) / 3.0
                    first_value = contact_line_value(first_third)
                    second_value = contact_line_value(second_third)
                    if first_value is None or second_value is None:
                        return None
                    if first_value < second_value:
                        lower_time = first_third
                    else:
                        upper_time = second_third
                peak_time = 0.5 * (lower_time + upper_time)
                peak_value = contact_line_value(peak_time)
                if peak_value is not None and peak_value >= -1.0e-6:
                    lower = sampled_values[peak_index - 1][0]
                    upper = peak_time
                    if contact_line_value(lower) is not None:
                        for _ in range(48):
                            midpoint = 0.5 * (lower + upper)
                            midpoint_value = contact_line_value(midpoint)
                            if midpoint_value is None:
                                return None
                            if midpoint_value < 0.0:
                                lower = midpoint
                            else:
                                upper = midpoint
                        roots.append(0.5 * (lower + upper))
    for intercept_time in sorted(
        root for root in roots if 1.0e-4 < root <= float(maximum_time)
    ):
        predicted_normal_speed = linear + quadratic_acceleration * intercept_time
        if predicted_normal_speed <= 0.05:
            continue
        center_x, center_y = predicted_ego_xy(
            ego_location,
            ego_velocity,
            acceleration_xy,
            intercept_time,
        )
        tangent_direction = normalized_xy(
            float(ego_velocity.x) + acceleration_xy[0] * intercept_time,
            float(ego_velocity.y) + acceleration_xy[1] * intercept_time,
        )
        if tangent_direction is None:
            continue
        target_x = center_x + contact_offset * tangent_direction[0]
        target_y = center_y + contact_offset * tangent_direction[1]
        # Remove only numerical root error. The contact reference, including
        # its tangent-direction component along the crossing line, is retained.
        line_error = (
            (target_x - crossing_origin[0]) * normal_x
            + (target_y - crossing_origin[1]) * normal_y
        )
        target_x -= line_error * normal_x
        target_y -= line_error * normal_y
        target_along_line = (
            (target_x - crossing_origin[0]) * crossing_direction[0]
            + (target_y - crossing_origin[1]) * crossing_direction[1]
        )
        if target_along_line < -1.0e-4:
            continue
        pedestrian_distance = math.hypot(
            target_x - float(pedestrian_location.x),
            target_y - float(pedestrian_location.y),
        )
        perpendicular_residual = abs(
            crossing_direction[0] * tangent_direction[0]
            + crossing_direction[1] * tangent_direction[1]
        )
        return LineInterceptUpdate(
            time_seconds=intercept_time,
            target_x=target_x,
            target_y=target_y,
            required_pedestrian_speed=pedestrian_distance / intercept_time,
            longitudinal_acceleration=quadratic_acceleration,
            perpendicular_error_degrees=math.degrees(
                math.asin(max(0.0, min(1.0, perpendicular_residual)))
            ),
        )
    return None


def evaluate_trigger(
    ego_transform: carla.Transform,
    ego_velocity: carla.Vector3D,
    ego_acceleration: carla.Vector3D,
    pedestrian_location: carla.Location,
    minimum_pedestrian_speed: float,
    maximum_pedestrian_speed: float,
    min_ego_speed: float,
    min_closing_speed: float,
    trigger_distance: float,
    max_approach_angle_degrees: float,
    max_lateral_offset: float,
    reaction_time: float,
    max_brake_deceleration: float,
    braking_margin: float,
    minimum_intercept_time: float,
    max_intercept_time: float,
    prediction_acceleration_limit: float,
    collision_clearance: float = 0.0,
    front_contact_offset: Optional[float] = None,
) -> Optional[TriggerDecision]:
    """Return an activation decision when all approach and timing gates pass."""
    ego_location = ego_transform.location
    relative_x = float(pedestrian_location.x - ego_location.x)
    relative_y = float(pedestrian_location.y - ego_location.y)
    separation = math.hypot(relative_x, relative_y)
    if separation <= 1e-6 or separation > float(trigger_distance):
        return None

    velocity_x = float(ego_velocity.x)
    velocity_y = float(ego_velocity.y)
    ego_speed = math.hypot(velocity_x, velocity_y)
    if ego_speed < float(min_ego_speed):
        return None
    velocity_direction = normalized_xy(velocity_x, velocity_y)
    if velocity_direction is None:
        return None

    relative_unit = (relative_x / separation, relative_y / separation)
    closing_speed = velocity_x * relative_unit[0] + velocity_y * relative_unit[1]
    if closing_speed < float(min_closing_speed):
        return None

    forward = ego_transform.get_forward_vector()
    forward_direction = normalized_xy(float(forward.x), float(forward.y))
    if forward_direction is None:
        return None
    approach_cosine = max(
        -1.0,
        min(
            1.0,
            forward_direction[0] * relative_unit[0]
            + forward_direction[1] * relative_unit[1],
        ),
    )
    approach_angle = math.degrees(math.acos(approach_cosine))
    if approach_angle > float(max_approach_angle_degrees):
        return None

    lateral_offset = abs(
        velocity_direction[0] * relative_y
        - velocity_direction[1] * relative_x
    )
    if lateral_offset > float(max_lateral_offset):
        return None

    intercept = solve_intercept(
        ego_location,
        ego_velocity,
        ego_acceleration,
        pedestrian_location,
        minimum_pedestrian_speed,
        maximum_pedestrian_speed,
        minimum_intercept_time,
        max_intercept_time,
        prediction_acceleration_limit,
        0.0 if front_contact_offset is None else front_contact_offset,
    )
    if intercept is None:
        return None

    ego_travel = intercept.ego_travel
    # Center targeting predicts the actor origin at L2, so retain v4's former
    # clearance subtraction. Front targeting already solves the earlier time
    # at which the shifted contact reference reaches L2; subtracting clearance
    # again would activate too late and double-count the vehicle nose.
    if front_contact_offset is None:
        effective_travel = max(
            0.0,
            ego_travel - max(0.0, collision_clearance),
        )
    else:
        effective_travel = max(0.0, ego_travel)
    stopping_distance = (
        ego_speed * float(reaction_time)
        + ego_speed * ego_speed / (2.0 * float(max_brake_deceleration))
        + float(braking_margin)
    )
    if effective_travel > stopping_distance:
        return None

    return TriggerDecision(
        intercept=intercept,
        ego_speed=ego_speed,
        separation=separation,
        closing_speed=closing_speed,
        lateral_offset=lateral_offset,
        approach_angle_degrees=approach_angle,
        ego_travel=ego_travel,
        effective_travel=effective_travel,
        stopping_distance=stopping_distance,
    )


class CollisionRegistry:
    """Small thread-safe mailbox populated by CARLA collision callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registered_pedestrian_ids = set()
        self._vehicle_pairs = set()

    def register(self, pedestrian_id: int) -> None:
        pedestrian_id = int(pedestrian_id)
        with self._lock:
            self._registered_pedestrian_ids.add(pedestrian_id)
            self._vehicle_pairs = {
                pair for pair in self._vehicle_pairs if pair[0] != pedestrian_id
            }

    def unregister(self, pedestrian_id: int) -> None:
        pedestrian_id = int(pedestrian_id)
        with self._lock:
            self._registered_pedestrian_ids.discard(pedestrian_id)
            self._vehicle_pairs = {
                pair for pair in self._vehicle_pairs if pair[0] != pedestrian_id
            }

    def record(self, pedestrian_id: int, event) -> None:
        other_actor = getattr(event, "other_actor", None)
        other_id = getattr(other_actor, "id", None)
        other_type_id = str(getattr(other_actor, "type_id", ""))
        if other_id is None or not other_type_id.startswith("vehicle."):
            return
        pedestrian_id = int(pedestrian_id)
        with self._lock:
            if pedestrian_id not in self._registered_pedestrian_ids:
                return
            self._vehicle_pairs.add((pedestrian_id, int(other_id)))

    def consume_vehicle_hit(self, pedestrian_id: int) -> Optional[int]:
        pedestrian_id = int(pedestrian_id)
        with self._lock:
            vehicle_ids = sorted(
                pair[1]
                for pair in self._vehicle_pairs
                if pair[0] == pedestrian_id
            )
            if not vehicle_ids:
                return None
            self._vehicle_pairs = {
                pair for pair in self._vehicle_pairs if pair[0] != pedestrian_id
            }
            return vehicle_ids[0]


class ReactiveVehicleCollisionRegistry:
    """Thread-safe vehicle-sensor mailbox consumed only by the main loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registered_vehicle_ids = set()
        self._contacts = {}

    def register(self, vehicle_id: int) -> None:
        vehicle_id = int(vehicle_id)
        with self._lock:
            self._registered_vehicle_ids.add(vehicle_id)
            self._contacts = {
                key: contact
                for key, contact in self._contacts.items()
                if key[0] != vehicle_id
            }

    def unregister(self, vehicle_id: int) -> None:
        vehicle_id = int(vehicle_id)
        with self._lock:
            self._registered_vehicle_ids.discard(vehicle_id)
            self._contacts = {
                key: contact
                for key, contact in self._contacts.items()
                if key[0] != vehicle_id
            }

    def record(self, vehicle_id: int, event) -> None:
        other_actor = getattr(event, "other_actor", None)
        other_id = getattr(other_actor, "id", None)
        other_type_id = str(getattr(other_actor, "type_id", ""))
        if other_id is None:
            return
        try:
            role_name = str(other_actor.attributes.get("role_name", ""))
        except (AttributeError, RuntimeError):
            role_name = ""
        try:
            frame = int(getattr(event, "frame", -1))
        except (TypeError, ValueError):
            frame = -1
        normal_impulse = getattr(event, "normal_impulse", None)
        try:
            impulse_magnitude = math.sqrt(
                float(normal_impulse.x) ** 2
                + float(normal_impulse.y) ** 2
                + float(normal_impulse.z) ** 2
            )
        except (AttributeError, TypeError, ValueError):
            impulse_magnitude = 0.0
        if not math.isfinite(impulse_magnitude):
            impulse_magnitude = 0.0
        vehicle_id = int(vehicle_id)
        contact = ReactiveVehicleContact(
            actor_id=int(other_id),
            type_id=other_type_id,
            role_name=role_name,
            frame=frame,
            impulse_magnitude=impulse_magnitude,
        )
        with self._lock:
            if vehicle_id not in self._registered_vehicle_ids:
                return
            key = (vehicle_id, int(other_id))
            existing = self._contacts.get(key)
            if (
                existing is None
                or contact.impulse_magnitude >= existing.impulse_magnitude
            ):
                self._contacts[key] = contact

    def consume_contacts(
        self,
        vehicle_id: int,
    ) -> List[ReactiveVehicleContact]:
        vehicle_id = int(vehicle_id)
        with self._lock:
            contacts = sorted(
                (
                    contact
                    for key, contact in self._contacts.items()
                    if key[0] == vehicle_id
                ),
                key=lambda contact: (
                    contact.frame,
                    contact.actor_id,
                    contact.type_id,
                ),
            )
            self._contacts = {
                key: contact
                for key, contact in self._contacts.items()
                if key[0] != vehicle_id
            }
            return contacts


class NavigationSampler:
    """Lazily sample unique pedestrian navigation points once per process."""

    def __init__(self, world: carla.World, sample_count: int) -> None:
        self._world = world
        self._sample_count = int(sample_count)
        self._locations: Optional[List[carla.Location]] = None

    def locations(self) -> List[carla.Location]:
        if self._locations is not None:
            return self._locations
        locations: List[carla.Location] = []
        attempts = 0
        maximum_attempts = max(self._sample_count * 6, self._sample_count)
        while len(locations) < self._sample_count and attempts < maximum_attempts:
            attempts += 1
            try:
                location = self._world.get_random_location_from_navigation()
            except RuntimeError:
                break
            if location is None:
                continue
            copied = carla.Location(
                x=float(location.x),
                y=float(location.y),
                z=float(location.z),
            )
            if any(planar_distance(copied, existing) < 0.5 for existing in locations):
                continue
            locations.append(copied)
        self._locations = locations
        LOG.info(
            "Navigation fallback collected %d/%d unique random samples",
            len(locations),
            self._sample_count,
        )
        return locations

    def nearest(
        self,
        target: carla.Location,
        maximum_distance: float,
    ) -> List[carla.Location]:
        ranked = sorted(
            self.locations(),
            key=lambda location: planar_distance(location, target),
        )
        return [
            location
            for location in ranked
            if planar_distance(location, target) <= maximum_distance
        ]


def resolve_vehicle_targets(args: argparse.Namespace) -> List[VehicleTarget]:
    if args.no_vehicle_blockers:
        return []
    if args.vehicle_locations is None:
        locations = DEFAULT_VEHICLE_LOCATIONS
        source = "built-in-capture"
    else:
        locations = args.vehicle_locations
        source = "command-line"
    targets: List[VehicleTarget] = []
    for index, values in enumerate(locations, 1):
        blueprint_id = None
        target_source = source
        if (
            source == "built-in-capture"
            and index in DEFAULT_STATIC_PATROL_INDICES
        ):
            blueprint_id = DEFAULT_ADDITIONAL_PATROL_BLUEPRINT
            if index == DEFAULT_ADDITIONAL_PATROL_INDEX and (
                args.no_reactive_vehicle
                or args.reactive_vehicle_index != DEFAULT_REACTIVE_VEHICLE_INDEX
            ):
                values = DEFAULT_ADDITIONAL_PATROL_CLEARANCE_LOCATION
                target_source = "built-in-capture+static-clearance"
        targets.append(
            VehicleTarget(
                x=float(values[0]),
                y=float(values[1]),
                z=float(values[2]),
                yaw=float(values[3]),
                source=target_source,
                blueprint_id=blueprint_id,
            )
        )
    return targets


def resolve_pedestrian_targets(
    world: carla.World,
    args: argparse.Namespace,
    navigation: NavigationSampler,
) -> List[PedestrianTarget]:
    if args.no_pedestrian_blockers:
        return []
    if args.pedestrian_locations is not None:
        return [
            PedestrianTarget(
                x=float(values[0]),
                y=float(values[1]),
                z=float(values[2]),
                yaw=float(values[3]),
                source="command-line",
            )
            for values in args.pedestrian_locations
        ]
    if not args.from_spectator:
        return [
            PedestrianTarget(
                x=float(values[0]),
                y=float(values[1]),
                z=float(values[2]),
                yaw=float(values[3]),
                source="built-in-capture",
            )
            for values in DEFAULT_PEDESTRIAN_LOCATIONS
        ]

    spectator_transform = world.get_spectator().get_transform()
    spectator_location = spectator_transform.location
    LOG.info(
        "Spectator raw transform x=%.3f y=%.3f z=%.3f pitch=%.2f yaw=%.2f roll=%.2f",
        spectator_location.x,
        spectator_location.y,
        spectator_location.z,
        spectator_transform.rotation.pitch,
        spectator_transform.rotation.yaw,
        spectator_transform.rotation.roll,
    )

    ground_location = None
    try:
        ground_projection = world.ground_projection(
            spectator_location,
            args.spectator_ground_search,
        )
        if ground_projection is not None:
            ground_location = ground_projection.location
    except (AttributeError, RuntimeError):
        pass

    if ground_location is None:
        nearest = navigation.nearest(
            spectator_location,
            args.nav_search_radius,
        )
        if not nearest:
            raise RuntimeError(
                "spectator ground projection failed and no nearby navigation "
                "sample is available"
            )
        ground_location = nearest[0]
        source = "spectator-nearest-navigation"
    else:
        source = "spectator-ground-projection"

    LOG.info(
        "Spectator-derived pedestrian ground target x=%.3f y=%.3f z=%.3f source=%s",
        ground_location.x,
        ground_location.y,
        ground_location.z,
        source,
    )
    return [
        PedestrianTarget(
            x=float(ground_location.x),
            y=float(ground_location.y),
            z=float(ground_location.z),
            yaw=float(spectator_transform.rotation.yaw),
            source=source,
        )
    ]


def find_pedestrian_blueprint(
    world: carla.World,
    blueprint_id: str,
    role_name: str,
):
    blueprint_library = world.get_blueprint_library()
    try:
        blueprint = blueprint_library.find(blueprint_id)
    except (IndexError, RuntimeError):
        blueprint = None
    if blueprint is None:
        try:
            candidates = sorted(
                (
                    candidate
                    for candidate in blueprint_library.filter(
                        "walker.pedestrian.*"
                    )
                    if candidate.id.startswith("walker.pedestrian.")
                ),
                key=lambda candidate: candidate.id,
            )
        except (AttributeError, RuntimeError):
            candidates = []
        if not candidates:
            raise ValueError(
                "pedestrian blueprint {!r} is unavailable and the CARLA "
                "library contains no walker.pedestrian.* fallback".format(
                    blueprint_id
                )
            )
        blueprint = candidates[0]
        LOG.warning(
            "Pedestrian blueprint %r is unavailable; using deterministic "
            "fallback %r",
            blueprint_id,
            blueprint.id,
        )
    if not blueprint.id.startswith("walker.pedestrian."):
        raise ValueError("blueprint {!r} is not a pedestrian".format(blueprint.id))
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
    return blueprint


def find_vehicle_blueprint(
    world: carla.World,
    blueprint_id: str,
    role_name: str,
):
    try:
        blueprint = world.get_blueprint_library().find(blueprint_id)
    except (IndexError, RuntimeError) as exc:
        raise ValueError(
            "vehicle blueprint {!r} is unavailable".format(blueprint_id)
        ) from exc
    if not blueprint.id.startswith("vehicle."):
        raise ValueError("blueprint {!r} is not a vehicle".format(blueprint.id))
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
    return blueprint


def vehicle_target_transform(
    target: VehicleTarget,
    z_offset: float,
) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(target.x),
            y=float(target.y),
            z=float(target.z + z_offset),
        ),
        carla.Rotation(yaw=float(target.yaw)),
    )


def spawn_static_vehicle(
    world: carla.World,
    target: VehicleTarget,
    index: int,
    args: argparse.Namespace,
) -> Optional[StaticVehicleSpawnResult]:
    role_name = "{}_{}".format(VEHICLE_ROLE_PREFIX, index)
    blueprint = find_vehicle_blueprint(
        world,
        target.blueprint_id or args.vehicle_blueprint,
        role_name,
    )
    candidates: List[Tuple[str, VehicleTarget]] = [("requested", target)]
    if (
        index == DEFAULT_ADDITIONAL_PATROL_INDEX
        and target.blueprint_id == DEFAULT_ADDITIONAL_PATROL_BLUEPRINT
        and target.source.startswith("built-in-capture")
    ):
        fallback_values = DEFAULT_ADDITIONAL_PATROL_CLEARANCE_LOCATION
        fallback_target = VehicleTarget(
            x=float(fallback_values[0]),
            y=float(fallback_values[1]),
            z=float(fallback_values[2]),
            yaw=float(fallback_values[3]),
            source="built-in-capture+shoulder-fallback",
            blueprint_id=DEFAULT_ADDITIONAL_PATROL_BLUEPRINT,
        )
        if (
            planar_distance(
                carla.Location(x=target.x, y=target.y),
                carla.Location(x=fallback_target.x, y=fallback_target.y),
            )
            > 1.0e-4
        ):
            candidates.append(
                ("lane-centered-shoulder-fallback", fallback_target)
            )

    attempted_transforms: List[carla.Transform] = []
    for placement_source, candidate in candidates:
        transform = vehicle_target_transform(candidate, args.vehicle_z_offset)
        attempted_transforms.append(transform)
        vehicle = try_spawn_actor(world, blueprint, transform)
        if vehicle is None:
            continue
        if placement_source != "requested":
            LOG.warning(
                "Static vehicle #%d requested pose was unavailable; using %s "
                "at x=%.3f y=%.3f z=%.3f yaw=%.2f",
                index,
                placement_source,
                transform.location.x,
                transform.location.y,
                transform.location.z,
                transform.rotation.yaw,
            )
        return StaticVehicleSpawnResult(
            actor=vehicle,
            transform=transform,
            placement_source=placement_source,
        )

    attempted_text = "; ".join(
        "x={:.3f} y={:.3f} z={:.3f} yaw={:.2f}".format(
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.yaw,
        )
        for transform in attempted_transforms
    )
    LOG.warning(
        "Static vehicle #%d failed to spawn at %d candidate(s): %s; "
        "locations may be occupied or intersect map geometry",
        index,
        len(attempted_transforms),
        attempted_text,
    )
    return None


def spawn_reactive_vehicle(
    world: carla.World,
    target: VehicleTarget,
    index: int,
    args: argparse.Namespace,
):
    role_name = "{}_{}".format(REACTIVE_VEHICLE_ROLE_PREFIX, index)
    blueprint = find_vehicle_blueprint(
        world,
        args.vehicle_blueprint,
        role_name,
    )
    transform = vehicle_target_transform(target, args.vehicle_z_offset)
    vehicle = try_spawn_actor(world, blueprint, transform)
    if vehicle is None:
        LOG.warning(
            "Reactive vehicle #%d failed to spawn at home "
            "x=%.3f y=%.3f z=%.3f yaw=%.2f; retry will be deferred",
            index,
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.yaw,
        )
    return vehicle


def make_reactive_vehicle_state(
    index: int,
    route_plan: ReactiveVehicleRoutePlan,
    actor,
    args: argparse.Namespace,
) -> ReactiveVehicleState:
    target_x = float(args.reactive_attack_location[0])
    target_y = float(args.reactive_attack_location[1])
    state = ReactiveVehicleState(
        index=index,
        actor=actor,
        target=route_plan.home,
        attack_x=target_x,
        attack_y=target_y,
        origin_x=float(route_plan.home.x),
        origin_y=float(route_plan.home.y),
        direction_x=route_plan.attack_tangent_x,
        direction_y=route_plan.attack_tangent_y,
        attack_distance=route_plan.attack_station,
        attack_yaw=math.degrees(
            math.atan2(
                route_plan.attack_tangent_y,
                route_plan.attack_tangent_x,
            )
        ),
        nominal_attack_x=target_x,
        nominal_attack_y=target_y,
        nominal_attack_distance=route_plan.attack_station,
        nominal_direction_x=route_plan.attack_tangent_x,
        nominal_direction_y=route_plan.attack_tangent_y,
        rearm_attack_x=target_x,
        rearm_attack_y=target_y,
        requested_target=route_plan.requested_home,
        route_locations=route_plan.locations,
        route_distances=route_plan.distances,
        route_length=route_plan.route_length,
        home_snap_distance=route_plan.home_snap_distance,
        attack_lateral_offset=route_plan.attack_lateral_offset,
        maximum_route_curvature=route_plan.maximum_curvature,
        route_speed_limit=route_plan.speed_limit,
        home_road_id=route_plan.home_road_id,
        home_lane_id=route_plan.home_lane_id,
        attack_road_id=route_plan.attack_road_id,
        attack_lane_id=route_plan.attack_lane_id,
    )
    if actor is None:
        state.generation = 0
        state.state = REACTIVE_STATE_RESPAWN_PENDING
        state.respawn_due = 0.0
    return state


def target_transform(target: PedestrianTarget, z_offset: float) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(target.x),
            y=float(target.y),
            z=float(target.z + z_offset),
        ),
        carla.Rotation(yaw=float(target.yaw)),
    )


def candidate_spawn_transforms(
    carla_map: carla.Map,
    target: PedestrianTarget,
    z_offset: float,
    navigation: NavigationSampler,
    search_radius: float,
) -> List[Tuple[str, carla.Transform]]:
    desired_location = target.location()
    candidates: List[Tuple[str, carla.Transform]] = []
    try:
        sidewalk_waypoint = carla_map.get_waypoint(
            desired_location,
            project_to_road=True,
            lane_type=carla.LaneType.Sidewalk,
        )
    except (AttributeError, RuntimeError):
        sidewalk_waypoint = None
    if sidewalk_waypoint is not None:
        location = sidewalk_waypoint.transform.location
        if planar_distance(location, desired_location) <= search_radius:
            candidates.append(
                (
                    "nearest-sidewalk-waypoint",
                    carla.Transform(
                        carla.Location(
                            x=float(location.x),
                            y=float(location.y),
                            z=float(location.z + z_offset),
                        ),
                        carla.Rotation(yaw=float(target.yaw)),
                    ),
                )
            )

    for location in navigation.nearest(desired_location, search_radius):
        candidates.append(
            (
                "sampled-navigation",
                carla.Transform(
                    carla.Location(
                        x=float(location.x),
                        y=float(location.y),
                        z=float(location.z + z_offset),
                    ),
                    carla.Rotation(yaw=float(target.yaw)),
                ),
            )
        )
    return candidates


def try_spawn_actor(world: carla.World, blueprint, transform: carla.Transform):
    try:
        return world.try_spawn_actor(blueprint, transform)
    except RuntimeError:
        return None


def stop_walker(walker) -> None:
    if walker is None:
        return
    try:
        walker.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
    except (AttributeError, RuntimeError):
        pass
    try:
        walker.set_target_angular_velocity(
            carla.Vector3D(x=0.0, y=0.0, z=0.0)
        )
    except (AttributeError, RuntimeError):
        pass
    try:
        control = carla.WalkerControl()
        control.direction = carla.Vector3D(x=0.0, y=0.0, z=0.0)
        control.speed = 0.0
        control.jump = False
        walker.apply_control(control)
    except RuntimeError:
        pass


def configure_walker_for_collision(walker) -> None:
    try:
        walker.set_simulate_physics(True)
    except (AttributeError, RuntimeError):
        pass
    try:
        walker.set_collisions(True)
    except (AttributeError, RuntimeError):
        pass
    stop_walker(walker)


def spawn_pedestrian(
    world: carla.World,
    carla_map: carla.Map,
    target: PedestrianTarget,
    index: int,
    args: argparse.Namespace,
    navigation: NavigationSampler,
):
    role_name = "{}_{}".format(PEDESTRIAN_ROLE_PREFIX, index)
    blueprint = find_pedestrian_blueprint(world, args.blueprint, role_name)
    desired_transform = target_transform(target, args.z_offset)
    walker = try_spawn_actor(world, blueprint, desired_transform)
    if walker is not None:
        configure_walker_for_collision(walker)
        LOG.info(
            "Spawned pedestrian #%d id=%d directly at requested transform",
            index,
            walker.id,
        )
        return walker

    LOG.warning(
        "Direct pedestrian spawn #%d failed; searching nearby spawn candidates",
        index,
    )
    blocking_actor = blocking_actor_at_location(
        world,
        desired_transform.location,
        DEFAULT_SPAWN_OCCUPANCY_CLEARANCE_M,
    )
    if blocking_actor is not None:
        raise RuntimeError(
            "requested pedestrian target is occupied by actor id={} type={}".format(
                blocking_actor.id,
                blocking_actor.type_id,
            )
        )
    candidates = candidate_spawn_transforms(
        carla_map,
        target,
        args.z_offset,
        navigation,
        args.nav_search_radius,
    )
    for source, fallback_transform in candidates:
        walker = try_spawn_actor(world, blueprint, fallback_transform)
        if walker is None:
            continue
        try:
            blocking_actor = blocking_actor_at_location(
                world,
                desired_transform.location,
                DEFAULT_SPAWN_OCCUPANCY_CLEARANCE_M,
                ignored_ids=(int(walker.id),),
            )
        except RuntimeError:
            destroy_actor(walker, "fallback pedestrian")
            raise
        if blocking_actor is not None:
            destroy_actor(walker, "fallback pedestrian")
            raise RuntimeError(
                "requested pedestrian target became occupied by actor "
                "id={} type={}".format(
                    blocking_actor.id,
                    blocking_actor.type_id,
                )
            )
        try:
            walker.set_transform(desired_transform)
        except RuntimeError:
            try:
                walker.destroy()
            except RuntimeError:
                pass
            walker = None
            continue
        configure_walker_for_collision(walker)
        try:
            final_location = walker.get_transform().location
            relocation_error = planar_distance(
                final_location,
                desired_transform.location,
            )
        except RuntimeError:
            relocation_error = 0.0
        if relocation_error > args.placement_tolerance:
            LOG.warning(
                "Pedestrian #%d relocation confirmation is %.2f m from target; "
                "the server snapshot may update on the next master tick",
                index,
                relocation_error,
            )
        LOG.info(
            "Spawned pedestrian #%d id=%d at %s, then relocated to requested "
            "x=%.3f y=%.3f z=%.3f",
            index,
            walker.id,
            source,
            desired_transform.location.x,
            desired_transform.location.y,
            desired_transform.location.z,
        )
        return walker
    raise RuntimeError(
        "no pedestrian spawn candidate is available within {:.1f} m of "
        "target #{}".format(args.nav_search_radius, index)
    )


def inactive_front_sensor_mount_transform(parent) -> carla.Transform:
    """Build one front-facing local pose shared by RGB camera and radar."""
    try:
        type_id = str(parent.type_id)
    except (AttributeError, RuntimeError):
        type_id = ""
    is_pedestrian = type_id.startswith("walker.pedestrian.")
    fallback_x = 0.35 if is_pedestrian else 2.50
    fallback_z = 1.55 if is_pedestrian else 1.00
    front_x = fallback_x
    lateral_y = 0.0
    height_z = fallback_z
    try:
        bounding_box = parent.bounding_box
        location = bounding_box.location
        extent = bounding_box.extent
        candidate_values = (
            float(location.x),
            float(location.y),
            float(location.z),
            float(extent.x),
            float(extent.z),
        )
        if not all(math.isfinite(value) for value in candidate_values):
            raise ValueError("non-finite blocker bounding box")
        front_x = (
            candidate_values[0]
            + max(0.0, candidate_values[3])
            + INACTIVE_FRONT_SENSOR_MARGIN_M
        )
        lateral_y = candidate_values[1]
        center_upper_z = candidate_values[2] + 0.5 * max(
            0.0,
            candidate_values[4],
        )
        if is_pedestrian:
            height_z = max(1.45, min(2.00, center_upper_z))
        else:
            height_z = max(0.55, center_upper_z)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return carla.Transform(
        carla.Location(x=front_x, y=lateral_y, z=height_z),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )


def configure_inactive_sensor_blueprint(
    world: carla.World,
    blueprint_id: str,
    role_name: str,
    attributes: Sequence[Tuple[str, object]],
):
    """Return a stream-capable blueprint carrying legacy inactive metadata."""
    try:
        blueprint = world.get_blueprint_library().find(blueprint_id)
    except (IndexError, RuntimeError) as exc:
        raise RuntimeError(
            "required sensor blueprint {!r} is unavailable".format(
                blueprint_id,
            )
        ) from exc
    if not blueprint.has_attribute("role_name"):
        raise RuntimeError(
            "sensor blueprint {!r} has no role_name attribute".format(
                blueprint_id,
            )
        )
    blueprint.set_attribute("role_name", role_name)
    for name, value in attributes:
        if blueprint.has_attribute(name):
            blueprint.set_attribute(name, str(value))
    return blueprint


def inactive_front_sensor_ids(
    pair: Optional[InactiveFrontSensorPair],
) -> Tuple[int, ...]:
    if pair is None:
        return ()
    actor_ids: List[int] = []
    for sensor in (pair.camera, pair.radar):
        sensor_id = getattr(sensor, "id", None)
        if sensor_id is not None:
            actor_ids.append(int(sensor_id))
    return tuple(actor_ids)


def sensor_is_listening(sensor) -> bool:
    """Support CARLA builds exposing is_listening as a method or property."""
    listening = getattr(sensor, "is_listening", False)
    if callable(listening):
        listening = listening()
    return bool(listening)


def destroy_inactive_front_sensor_pair(
    pair: Optional[InactiveFrontSensorPair],
    actor_kind: str,
) -> None:
    """Destroy unsubscribed sensors directly; this client never listened."""
    if pair is None:
        return
    destroy_actor(pair.radar, "{} radar".format(actor_kind))
    destroy_actor(pair.camera, "{} RGB camera".format(actor_kind))


def inactive_sensor_role_names(
    host_kind: str,
    index: int,
    generation: int,
) -> Tuple[str, str]:
    """Return stable RGB/radar roles for one parent generation."""
    role_suffix = "{}_{}_g{}".format(host_kind, index, generation)
    return (
        "{}_camera_{}".format(INACTIVE_FRONT_SENSOR_ROLE_PREFIX, role_suffix),
        "{}_radar_{}".format(INACTIVE_FRONT_SENSOR_ROLE_PREFIX, role_suffix),
    )


def spawn_inactive_front_sensor_pair(
    world: carla.World,
    parent,
    host_kind: str,
    index: int,
    generation: int,
    mount_transform: Optional[carla.Transform] = None,
    ownership_sink: Optional[List[InactiveFrontSensorPair]] = None,
) -> InactiveFrontSensorPair:
    """Reject obsolete physical spatial-sensor deployment unconditionally.

    Spatial-map clients now synthesize display-only camera/radar markers from
    blocker parents and traffic-light roots. Keeping this named guard makes an
    accidental legacy call fail before looking up a blueprint or mutating the
    CARLA world.
    """
    del world, parent, index, generation, mount_transform, ownership_sink
    raise RuntimeError(
        "physical RGB/radar deployment is disabled for {} hosts; spatial-map "
        "clients must use virtual markers".format(host_kind)
    )


def dedupe_route_xy_points(
    points: Sequence[Tuple[float, float]],
    minimum_distance_m: float = TRAFFIC_LIGHT_ROUTE_DEDUPE_DISTANCE_M,
) -> Tuple[Tuple[float, float], ...]:
    """Remove consecutive near-duplicates without changing route order."""
    minimum_distance = max(0.0, float(minimum_distance_m))
    result: List[Tuple[float, float]] = []
    for x_coord, y_coord in points:
        point = (float(x_coord), float(y_coord))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("pedestrian route contains a non-finite XY point")
        if result and math.hypot(
            point[0] - result[-1][0],
            point[1] - result[-1][1],
        ) < minimum_distance:
            continue
        result.append(point)
    return tuple(result)


def route_config_xyz_points(
    values: Sequence[dict],
) -> Tuple[Tuple[float, float, float], ...]:
    """Convert already validated route locations into XYZ tuples."""
    return tuple(
        (float(value["x"]), float(value["y"]), float(value["z"]))
        for value in values
    )


def spatial_route_point_distance(
    first: Tuple[float, float, float],
    second: Tuple[float, float, float],
) -> float:
    return math.hypot(
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    )


def dedupe_route_xyz_points(
    points: Sequence[Tuple[float, float, float]],
    minimum_distance_m: float,
) -> Tuple[Tuple[float, float, float], ...]:
    """Match v13's consecutive 3D route-point deduplication."""
    minimum_distance = max(0.0, float(minimum_distance_m))
    result: List[Tuple[float, float, float]] = []
    for x_coord, y_coord, z_coord in points:
        point = (float(x_coord), float(y_coord), float(z_coord))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("pedestrian route contains a non-finite XYZ point")
        if (
            result
            and spatial_route_point_distance(result[-1], point)
            < minimum_distance
        ):
            continue
        result.append(point)
    return tuple(result)


def planar_route_point_distance(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def traffic_light_sensor_route_polyline(
    route_config: dict,
) -> Tuple[Tuple[Tuple[float, float], ...], str]:
    """Resolve v13's 3D path source, then project it into CARLA world XY."""
    start_value = route_config["start"]["location"]
    visual_start = (
        float(start_value["x"]),
        float(start_value["y"]),
        float(start_value["z"])
        - float(route_config.get("spawn_height_offset_m", 0.0)),
    )
    control_values = route_config_xyz_points(
        route_config["intermediate_waypoints"]
    )
    control_points = (visual_start,) + control_values + route_config_xyz_points(
        (route_config["end"]["location"],)
    )
    controls = dedupe_route_xyz_points(
        control_points,
        TRAFFIC_LIGHT_ROUTE_CONTROL_DEDUPE_DISTANCE_M,
    )
    if len(controls) < 2:
        raise ValueError(
            "traffic-light sensor pedestrian route has fewer than two "
            "distinct control points"
        )

    planned = dedupe_route_xyz_points(
        route_config_xyz_points(route_config.get("planned_path", ())),
        TRAFFIC_LIGHT_ROUTE_DEDUPE_DISTANCE_M,
    )
    planned_valid = len(planned) >= 2
    if planned_valid:
        planned_valid = (
            spatial_route_point_distance(planned[0], controls[0])
            <= TRAFFIC_LIGHT_ROUTE_ENDPOINT_TOLERANCE_M
            and spatial_route_point_distance(planned[-1], controls[-1])
            <= TRAFFIC_LIGHT_ROUTE_ENDPOINT_TOLERANCE_M
            and abs(planned[0][2] - controls[0][2])
            <= TRAFFIC_LIGHT_ROUTE_VERTICAL_TOLERANCE_M
            and abs(planned[-1][2] - controls[-1][2])
            <= TRAFFIC_LIGHT_ROUTE_VERTICAL_TOLERANCE_M
        )
    if planned_valid and len(controls) > 2:
        control_index = 1
        for point in planned[1:-1]:
            if (
                spatial_route_point_distance(point, controls[control_index])
                <= TRAFFIC_LIGHT_ROUTE_CONTROL_TOLERANCE_M
                and abs(point[2] - controls[control_index][2])
                <= TRAFFIC_LIGHT_ROUTE_VERTICAL_TOLERANCE_M
            ):
                control_index += 1
                if control_index == len(controls) - 1:
                    break
        planned_valid = control_index == len(controls) - 1
    if planned_valid:
        maximum_gap = max(
            5.0,
            float(route_config["route_sampling_resolution_m"]) * 3.0,
        )
        planned_valid = all(
            spatial_route_point_distance(first, second) <= maximum_gap
            for first, second in zip(planned, planned[1:])
        )

    if planned_valid:
        selected_xyz = planned
        source = "saved planned_path"
    else:
        selected_xyz = controls
        source = "validated route controls"
    selected_xy = dedupe_route_xy_points(
        tuple((point[0], point[1]) for point in selected_xyz),
        TRAFFIC_LIGHT_ROUTE_DEDUPE_DISTANCE_M,
    )
    if len(selected_xy) < 2:
        raise ValueError(
            "traffic-light sensor pedestrian route has fewer than two "
            "distinct XY points after ground-plane projection"
        )
    return selected_xy, source


def closest_route_polyline_position(
    x_coord: float,
    y_coord: float,
    route_points: Sequence[Tuple[float, float]],
) -> Tuple[float, float, int]:
    """Return planar distance, route arclength, and closest segment index."""
    if len(route_points) < 2:
        raise ValueError("route polyline needs at least two XY points")
    point_x = float(x_coord)
    point_y = float(y_coord)
    if not math.isfinite(point_x) or not math.isfinite(point_y):
        raise ValueError("query point must have finite XY coordinates")

    best_key = None
    cumulative_length = 0.0
    for segment_index, (start, end) in enumerate(
        zip(route_points, route_points[1:])
    ):
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length_squared = delta_x * delta_x + delta_y * delta_y
        segment_length = math.sqrt(max(0.0, length_squared))
        if length_squared <= 1.0e-12:
            fraction = 0.0
        else:
            fraction = (
                (point_x - start[0]) * delta_x
                + (point_y - start[1]) * delta_y
            ) / length_squared
            fraction = max(0.0, min(1.0, fraction))
        projected_x = start[0] + fraction * delta_x
        projected_y = start[1] + fraction * delta_y
        offset_x = point_x - projected_x
        offset_y = point_y - projected_y
        distance_squared = offset_x * offset_x + offset_y * offset_y
        progress = cumulative_length + fraction * segment_length
        key = (distance_squared, progress, segment_index)
        if best_key is None or key < best_key:
            best_key = key
        cumulative_length += segment_length

    if best_key is None:
        raise ValueError("route polyline has no segments")
    return math.sqrt(max(0.0, best_key[0])), best_key[1], best_key[2]


def select_traffic_lights_near_route(
    traffic_lights: Sequence[object],
    route_points: Sequence[Tuple[float, float]],
    radius_m: float,
    maximum_poles: int,
) -> Tuple[
    Tuple[TrafficLightRouteCandidate, ...],
    int,
    Optional[TrafficLightRouteCandidate],
    int,
]:
    """Rank route-near roots once and apply the complete-pair safety cap."""
    radius = float(radius_m)
    if isinstance(maximum_poles, bool):
        raise ValueError("traffic-light sensor pole cap must be an integer")
    try:
        maximum = int(maximum_poles)
        maximum_numeric = float(maximum_poles)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "traffic-light sensor pole cap must be an integer"
        ) from exc
    if not math.isfinite(maximum_numeric) or maximum_numeric != float(maximum):
        raise ValueError("traffic-light sensor pole cap must be an integer")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("traffic-light route radius must be positive and finite")
    if maximum <= 0:
        raise ValueError("traffic-light sensor pole cap must be positive")
    try:
        normalized_route_points = tuple(
            (float(point[0]), float(point[1]))
            for point in route_points
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("route polyline contains an invalid XY point") from exc
    if len(normalized_route_points) < 2:
        raise ValueError("route polyline needs at least two XY points")
    if not all(
        math.isfinite(value)
        for point in normalized_route_points
        for value in point
    ):
        raise ValueError("route polyline contains a non-finite XY point")

    ranked: List[TrafficLightRouteCandidate] = []
    invalid_count = 0
    for traffic_light in traffic_lights:
        try:
            actor_id = int(traffic_light.id)
            location = traffic_light.get_transform().location
            x_coord = float(location.x)
            y_coord = float(location.y)
            if not math.isfinite(x_coord) or not math.isfinite(y_coord):
                raise ValueError("non-finite traffic-light root location")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            invalid_count += 1
            LOG.warning(
                "Skipping traffic-light root id=%s during route selection: %s",
                getattr(traffic_light, "id", "unknown"),
                exc,
            )
            continue
        distance, progress, segment_index = closest_route_polyline_position(
            x_coord,
            y_coord,
            normalized_route_points,
        )
        ranked.append(
            TrafficLightRouteCandidate(
                traffic_light=traffic_light,
                actor_id=actor_id,
                distance_m=distance,
                route_progress_m=progress,
                segment_index=segment_index,
            )
        )

    ranked.sort(
        key=lambda candidate: (
            candidate.distance_m,
            candidate.route_progress_m,
            candidate.actor_id,
        )
    )
    eligible = [
        candidate
        for candidate in ranked
        if candidate.distance_m
        <= radius + TRAFFIC_LIGHT_ROUTE_DISTANCE_EPSILON_M
    ]
    selected = tuple(eligible[:maximum])
    closest = ranked[0] if ranked else None
    return selected, len(eligible), closest, invalid_count


def traffic_light_sensor_ground_z(
    carla_map: carla.Map,
    traffic_light,
    root_transform=None,
) -> Tuple[float, str]:
    """Resolve a stable ground Z near one traffic-light root."""
    if root_transform is None:
        root_transform = traffic_light.get_transform()
    root_z = float(root_transform.location.z)
    try:
        waypoint = carla_map.get_waypoint(
            root_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if waypoint is not None:
            waypoint_z = float(waypoint.transform.location.z)
            if (
                math.isfinite(waypoint_z)
                and abs(waypoint_z - root_z)
                <= TRAFFIC_LIGHT_GROUND_Z_TOLERANCE_M
            ):
                return waypoint_z, "nearest-map-waypoint"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    if not math.isfinite(root_z):
        raise RuntimeError(
            "traffic-light id={} has no finite ground reference".format(
                getattr(traffic_light, "id", "unknown")
            )
        )
    return root_z, "traffic-light-root"


def traffic_light_sensor_mount_transform(
    carla_map: carla.Map,
    traffic_light,
    height_m: float,
    yaw_offset_degrees: float,
    pitch_degrees: float,
    root_transform=None,
) -> Tuple[carla.Transform, float, str]:
    """Build a pole-local pose whose world Z is ground Z plus height."""
    if root_transform is None:
        root_transform = traffic_light.get_transform()
    root_location = root_transform.location
    ground_z, ground_source = traffic_light_sensor_ground_z(
        carla_map,
        traffic_light,
        root_transform,
    )
    desired_world_location = carla.Location(
        x=float(root_location.x),
        y=float(root_location.y),
        z=float(ground_z) + float(height_m),
    )
    try:
        inverse_matrix = root_transform.get_inverse_matrix()
        world_coordinates = (
            float(desired_world_location.x),
            float(desired_world_location.y),
            float(desired_world_location.z),
            1.0,
        )
        local_coordinates = tuple(
            sum(
                float(inverse_matrix[row][column])
                * world_coordinates[column]
                for column in range(4)
            )
            for row in range(3)
        )
        relative_location = carla.Location(
            x=local_coordinates[0],
            y=local_coordinates[1],
            z=local_coordinates[2],
        )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        # Town10HD traffic-light roots have yaw only, so this fallback remains
        # exact for the supported map even if a CARLA binding lacks the helper.
        relative_location = carla.Location(
            x=0.0,
            y=0.0,
            z=float(desired_world_location.z - root_location.z),
        )
    values = (
        float(relative_location.x),
        float(relative_location.y),
        float(relative_location.z),
        float(ground_z),
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            "traffic-light id={} produced a non-finite sensor mount".format(
                getattr(traffic_light, "id", "unknown")
            )
        )
    return (
        carla.Transform(
            relative_location,
            carla.Rotation(
                pitch=float(pitch_degrees),
                yaw=float(yaw_offset_degrees),
                roll=0.0,
            ),
        ),
        float(ground_z),
        ground_source,
    )


def discover_traffic_light_roots(world: carla.World) -> List[object]:
    """Return one live actor per CARLA traffic-light pole/control root."""
    try:
        candidates = world.get_actors().filter("traffic.traffic_light")
    except RuntimeError as exc:
        raise RuntimeError("unable to enumerate traffic-light actors: {}".format(exc))
    roots = {}
    for actor in candidates:
        try:
            if actor.is_alive:
                roots[int(actor.id)] = actor
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return [roots[actor_id] for actor_id in sorted(roots)]


def existing_traffic_light_sensor_conflicts(
    world: carla.World,
    traffic_lights: Sequence[object],
) -> List[Tuple[int, int, str]]:
    """Find role/parent matches from another or stale blocker process."""
    expected = {}
    for traffic_light in traffic_lights:
        parent_id = int(traffic_light.id)
        camera_role, radar_role = inactive_sensor_role_names(
            "traffic_light",
            parent_id,
            1,
        )
        expected[(parent_id, camera_role)] = True
        expected[(parent_id, radar_role)] = True
    conflicts = []
    try:
        actors = world.get_actors()
    except RuntimeError as exc:
        raise RuntimeError(
            "unable to inspect existing traffic-light sensor actors: {}".format(
                exc
            )
        ) from exc
    for sensor in actors:
        try:
            type_id = str(sensor.type_id)
            if not (
                type_id.startswith("sensor.camera.")
                or type_id == "sensor.other.radar"
            ):
                continue
            parent = sensor.parent
            if parent is None:
                continue
            parent_id = int(parent.id)
            role_name = str(sensor.attributes.get("role_name", ""))
            if (parent_id, role_name) in expected:
                conflicts.append((int(sensor.id), parent_id, role_name))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return sorted(conflicts)


def prepare_traffic_light_sensor_deployment(
    carla_map: carla.Map,
    candidate: TrafficLightRouteCandidate,
    args: argparse.Namespace,
) -> PreparedTrafficLightSensorDeployment:
    """Resolve every selected-pole RPC value before any actor mutation."""
    traffic_light = candidate.traffic_light
    try:
        root_transform = traffic_light.get_transform()
        root_x = float(root_transform.location.x)
        root_y = float(root_transform.location.y)
        root_yaw = float(root_transform.rotation.yaw)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "unable to read selected traffic-light pole id={} transform: "
            "{}".format(candidate.actor_id, exc)
        ) from exc
    if not all(math.isfinite(value) for value in (root_x, root_y, root_yaw)):
        raise RuntimeError(
            "selected traffic-light pole id={} has a non-finite "
            "transform".format(candidate.actor_id)
        )
    mount_transform, ground_z, ground_source = (
        traffic_light_sensor_mount_transform(
            carla_map,
            traffic_light,
            args.traffic_light_sensor_height,
            args.traffic_light_sensor_yaw,
            args.traffic_light_sensor_pitch,
            root_transform=root_transform,
        )
    )
    world_yaw = (
        root_yaw
        + float(args.traffic_light_sensor_yaw)
        + 180.0
    ) % 360.0 - 180.0
    sensor_world_z = float(ground_z) + float(
        args.traffic_light_sensor_height
    )
    values = (root_x, root_y, float(ground_z), sensor_world_z, world_yaw)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            "selected traffic-light pole id={} produced non-finite deployment "
            "metadata".format(candidate.actor_id)
        )
    return PreparedTrafficLightSensorDeployment(
        candidate=candidate,
        mount_transform=mount_transform,
        ground_z=float(ground_z),
        ground_source=str(ground_source),
        root_x=root_x,
        root_y=root_y,
        sensor_world_z=sensor_world_z,
        world_yaw=world_yaw,
        pitch=float(args.traffic_light_sensor_pitch),
    )


def spawn_inactive_traffic_light_sensor_pair(
    world: carla.World,
    deployment: PreparedTrafficLightSensorDeployment,
    ownership_sink: Optional[List[InactiveFrontSensorPair]] = None,
) -> InactiveFrontSensorPair:
    """Attach one pair using only preflighted pole and mount metadata."""
    candidate = deployment.candidate
    pair = None
    try:
        pair = spawn_inactive_front_sensor_pair(
            world,
            candidate.traffic_light,
            "traffic_light",
            candidate.actor_id,
            1,
            mount_transform=deployment.mount_transform,
            ownership_sink=ownership_sink,
        )
        LOG.info(
            "Traffic-light pole id=%d available pair camera=%d radar=%d "
            "ground_z=%.3f sensor_world=(%.3f, %.3f, %.3f) "
            "world_yaw=%.1f pitch=%.1f ground_source=%s",
            candidate.actor_id,
            pair.camera.id,
            pair.radar.id,
            deployment.ground_z,
            deployment.root_x,
            deployment.root_y,
            deployment.sensor_world_z,
            deployment.world_yaw,
            deployment.pitch,
            deployment.ground_source,
        )
        return pair
    except BaseException as exc:
        # Until this function returns, the caller cannot add the pair to its
        # cleanup list. Always roll back a completed but unreturned pair.
        destroy_inactive_front_sensor_pair(
            pair,
            "unreturned traffic-light sensor",
        )
        if isinstance(
            exc,
            (AttributeError, IndexError, RuntimeError, TypeError, ValueError),
        ):
            raise RuntimeError(
                "unable to attach traffic-light pole id={} sensor pair: "
                "{}".format(candidate.actor_id, exc)
            ) from exc
        raise


def spawn_collision_sensor(
    world: carla.World,
    walker,
    registry: CollisionRegistry,
):
    sensor = None
    try:
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        sensor = world.spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=walker,
        )
        registry.register(int(walker.id))
        sensor.listen(
            lambda event, pedestrian_id=int(walker.id): registry.record(
                pedestrian_id,
                event,
            )
        )
        return sensor
    except (IndexError, RuntimeError) as exc:
        registry.unregister(int(walker.id))
        if sensor is not None:
            try:
                sensor.stop()
            except (AttributeError, RuntimeError):
                pass
            destroy_actor(sensor, "collision sensor")
        LOG.warning(
            "Collision sensor unavailable for pedestrian id=%d: %s",
            walker.id,
            exc,
        )
        return None


def spawn_reactive_vehicle_collision_sensor(
    world: carla.World,
    vehicle,
    registry: ReactiveVehicleCollisionRegistry,
):
    sensor = None
    try:
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        sensor = world.spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=vehicle,
        )
        registry.register(int(vehicle.id))
        sensor.listen(
            lambda event, vehicle_id=int(vehicle.id): registry.record(
                vehicle_id,
                event,
            )
        )
        return sensor
    except (IndexError, RuntimeError) as exc:
        registry.unregister(int(vehicle.id))
        if sensor is not None:
            try:
                sensor.stop()
            except (AttributeError, RuntimeError):
                pass
            destroy_actor(sensor, "reactive vehicle collision sensor")
        LOG.warning(
            "Collision sensor unavailable for reactive vehicle id=%d; "
            "oriented-box contact fallback remains active: %s",
            vehicle.id,
            exc,
        )
        return None


def find_ego_vehicle(
    world: carla.World,
    role_name: str,
    actor_id: Optional[int],
):
    if actor_id is not None:
        try:
            actor = world.get_actor(int(actor_id))
        except RuntimeError:
            actor = None
        if actor is None or not actor.is_alive:
            return None, "ego actor id={} is unavailable".format(actor_id)
        if not str(actor.type_id).startswith("vehicle."):
            return None, "ego actor id={} is not a vehicle".format(actor_id)
        return actor, "ego actor id={}".format(actor_id)

    try:
        matches = [
            actor
            for actor in world.get_actors().filter("vehicle.*")
            if actor.is_alive
            and actor.attributes.get("role_name", "") == role_name
        ]
    except RuntimeError:
        matches = []
    if not matches:
        return None, "waiting for vehicle role_name={!r}".format(role_name)
    if len(matches) > 1:
        return None, "ambiguous role_name={!r}: {} vehicles".format(
            role_name,
            len(matches),
        )
    return matches[0], "ego role_name={!r} id={}".format(role_name, matches[0].id)


def find_ego_pedestrian(
    world: carla.World,
    role_name: str,
    actor_id: Optional[int],
):
    """Find exactly one user-controlled walker without confusing blockers."""
    if actor_id is not None:
        try:
            actor = world.get_actor(int(actor_id))
        except RuntimeError:
            actor = None
        if actor is None or not actor.is_alive:
            return None, "ego-pedestrian actor id={} is unavailable".format(
                actor_id
            )
        if not str(actor.type_id).startswith("walker.pedestrian."):
            return None, "ego-pedestrian actor id={} is not a walker".format(
                actor_id
            )
        return actor, "ego-pedestrian actor id={}".format(actor_id)

    try:
        matches = [
            actor
            for actor in world.get_actors().filter("walker.pedestrian.*")
            if actor.is_alive
            and actor.attributes.get("role_name", "") == role_name
        ]
    except RuntimeError:
        matches = []
    if not matches:
        return None, "waiting for walker role_name={!r}".format(role_name)
    if len(matches) > 1:
        return None, "ambiguous role_name={!r}: {} walkers".format(
            role_name,
            len(matches),
        )
    return matches[0], "ego-pedestrian role_name={!r} id={}".format(
        role_name,
        matches[0].id,
    )


def pedestrian_approach_for_actor(
    pedestrian,
    args: argparse.Namespace,
    state: Optional[ReactiveVehicleState] = None,
    route_crossing: bool = False,
    minimum_route_station: float = 0.0,
    tracking: bool = False,
) -> Optional[PedestrianTargetApproach]:
    """Return a validated approach for activation or active tracking.

    Activation keeps the configured speed/ETA gates so unrelated walkers do
    not launch the vehicle.  Once one pedestrian is latched, ``tracking`` uses
    a longer horizon and a small nonzero motion threshold.  A slow pedestrian
    can therefore make the vehicle wait instead of making the ETA disappear
    and accidentally reusing the previous high-speed command.
    """
    if pedestrian is None:
        return None
    try:
        location = pedestrian.get_location()
        velocity = pedestrian.get_velocity()
        acceleration = pedestrian.get_acceleration()
    except (AttributeError, RuntimeError):
        return None
    if state is not None and route_crossing:
        return evaluate_pedestrian_route_crossing(
            location,
            velocity,
            acceleration,
            state.route_locations,
            state.route_distances,
            state.nominal_attack_distance,
            args.reactive_intercept_route_window,
            minimum_route_station,
            args.reactive_trigger_distance,
            args.reactive_min_pedestrian_speed,
            args.reactive_min_closing_speed,
            args.reactive_max_cross_track,
            args.reactive_min_pedestrian_eta,
            args.reactive_max_pedestrian_eta,
            args.reactive_pedestrian_acceleration_limit,
            args.reactive_min_crossing_angle,
            args.reactive_best_effort_launch,
        )
    target_x = (
        float(args.reactive_attack_location[0])
        if state is None
        else float(state.attack_x)
    )
    target_y = (
        float(args.reactive_attack_location[1])
        if state is None
        else float(state.attack_y)
    )
    minimum_speed = float(args.reactive_min_pedestrian_speed)
    minimum_closing_speed = float(args.reactive_min_closing_speed)
    minimum_eta = float(args.reactive_min_pedestrian_eta)
    maximum_eta = float(args.reactive_max_pedestrian_eta)
    if tracking:
        minimum_speed = min(minimum_speed, 0.05)
        minimum_closing_speed = min(minimum_closing_speed, 0.05)
        minimum_eta = 0.01
        maximum_eta = max(
            maximum_eta,
            float(args.reactive_approach_hold_timeout),
        )
    approach = evaluate_pedestrian_target_approach(
        location,
        velocity,
        acceleration,
        target_x,
        target_y,
        args.reactive_trigger_distance,
        minimum_speed,
        minimum_closing_speed,
        args.reactive_max_cross_track,
        minimum_eta,
        maximum_eta,
        args.reactive_pedestrian_acceleration_limit,
    )
    if approach is None:
        return None
    if state is None:
        return approach
    return PedestrianTargetApproach(
        distance=approach.distance,
        speed=approach.speed,
        closing_speed=approach.closing_speed,
        cross_track_miss=approach.cross_track_miss,
        eta_seconds=approach.eta_seconds,
        direction_x=approach.direction_x,
        direction_y=approach.direction_y,
        radial_acceleration=approach.radial_acceleration,
        target_x=state.attack_x,
        target_y=state.attack_y,
        route_station=state.attack_distance,
        route_tangent_x=state.direction_x,
        route_tangent_y=state.direction_y,
        crossing_angle_degrees=math.degrees(
            math.acos(
                max(
                    0.0,
                    min(
                        1.0,
                        abs(
                            approach.direction_x * state.direction_x
                            + approach.direction_y * state.direction_y
                        ),
                    ),
                )
            )
        ),
    )


def actor_collision_clearance(
    ego,
    pedestrian,
    ego_velocity,
    direction_override: Optional[Tuple[float, float]] = None,
) -> float:
    """Approximate center-to-center travel removed by both bounding boxes."""
    direction = direction_override
    if direction is None:
        direction = normalized_xy(float(ego_velocity.x), float(ego_velocity.y))
    if direction is None:
        return 0.0
    try:
        transform = ego.get_transform()
        forward = transform.get_forward_vector()
        forward_direction = normalized_xy(float(forward.x), float(forward.y))
        if forward_direction is None:
            return 0.0
        right_direction = (-forward_direction[1], forward_direction[0])
        extent = ego.bounding_box.extent
        ego_extent = (
            abs(direction[0] * forward_direction[0] + direction[1] * forward_direction[1])
            * float(extent.x)
            + abs(direction[0] * right_direction[0] + direction[1] * right_direction[1])
            * float(extent.y)
        )
        pedestrian_extent = pedestrian.bounding_box.extent
        pedestrian_radius = math.hypot(
            float(pedestrian_extent.x),
            float(pedestrian_extent.y),
        )
        return max(0.0, ego_extent + pedestrian_radius)
    except (AttributeError, RuntimeError):
        return 0.0


def actor_leading_support(
    actor,
    direction: Tuple[float, float],
) -> Optional[float]:
    """Return the actor box's leading support from its origin in ``direction``.

    CARLA bounding boxes may have their own location and rotation relative to
    the actor. Projecting their transformed world vertices therefore gives a
    more faithful vehicle-nose distance than assuming ``extent.x`` alone.
    """
    unit_direction = normalized_xy(float(direction[0]), float(direction[1]))
    if unit_direction is None:
        return None
    try:
        transform = actor.get_transform()
        origin = transform.location
        vertices = actor.bounding_box.get_world_vertices(transform)
    except (AttributeError, RuntimeError):
        return None
    supports = []
    for vertex in vertices:
        try:
            support = (
                (float(vertex.x) - float(origin.x)) * unit_direction[0]
                + (float(vertex.y) - float(origin.y)) * unit_direction[1]
            )
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(support):
            supports.append(support)
    if not supports:
        return None
    return max(0.0, max(supports))


def front_impact_contact_offset(
    ego,
    pedestrian,
    ego_velocity: carla.Vector3D,
    lead_margin: float,
) -> Optional[float]:
    """Return center-to-pedestrian distance for a reliable front impact.

    The vehicle contribution uses the exact oriented leading box support. The
    walker contribution is a conservative planar radius because its yaw is
    realigned only after the interception decision. ``lead_margin`` schedules
    the walker slightly before first physical contact to absorb a control/tick
    delay while leaving the crossing point unchanged.
    """
    margin = max(0.0, float(lead_margin))
    direction = normalized_xy(float(ego_velocity.x), float(ego_velocity.y))
    if direction is None:
        return None
    vehicle_support = actor_leading_support(ego, direction)
    pedestrian_support = actor_planar_bounding_radius(pedestrian)
    if vehicle_support is not None and pedestrian_support is not None:
        return max(0.0, vehicle_support) + max(0.0, pedestrian_support) + margin
    # Do not freeze an extent-only approximation for the whole encounter: it
    # omits bbox local offsets/rotation and can materially understate the nose
    # of buses or custom vehicles. The WAITING state will retry next snapshot.
    return None


def walker_animation_speed(requested_speed: float, command_scale: float) -> float:
    """Return a bounded WalkerControl speed used only for run animation."""
    return min(
        DEFAULT_WALKER_ANIMATION_SPEED_CAP_MPS,
        max(0.0, float(requested_speed * command_scale)),
    )


def face_walker_toward(walker, target_x: float, target_y: float) -> bool:
    """Align the walker body with its crossing endpoint without moving it."""
    try:
        transform = walker.get_transform()
        direction = normalized_xy(
            target_x - float(transform.location.x),
            target_y - float(transform.location.y),
        )
        if direction is None:
            return False
        walker.set_transform(
            carla.Transform(
                carla.Location(
                    x=float(transform.location.x),
                    y=float(transform.location.y),
                    z=float(transform.location.z),
                ),
                carla.Rotation(
                    pitch=0.0,
                    yaw=math.degrees(math.atan2(direction[1], direction[0])),
                    roll=0.0,
                ),
            )
        )
        return True
    except (AttributeError, RuntimeError):
        return False


def walk_toward(
    walker,
    target_x: float,
    target_y: float,
    requested_speed: float,
    command_scale: float,
) -> bool:
    try:
        location = walker.get_location()
    except (AttributeError, RuntimeError):
        return False
    direction = normalized_xy(target_x - location.x, target_y - location.y)
    if direction is None:
        stop_walker(walker)
        return True
    root_velocity = carla.Vector3D(
        x=float(direction[0] * requested_speed),
        y=float(direction[1] * requested_speed),
        z=0.0,
    )
    control = carla.WalkerControl()
    control.direction = carla.Vector3D(
        x=float(direction[0]),
        y=float(direction[1]),
        z=0.0,
    )
    control.speed = walker_animation_speed(requested_speed, command_scale)
    control.jump = False
    try:
        # Direct WalkerControl motion is unusually slow and saturates near
        # 2 m/s in this CARLA build. Root velocity provides the deterministic
        # physical speed used by solve_intercept(); WalkerControl supplies the
        # matching run animation without changing the trajectory model.
        walker.set_target_velocity(root_velocity)
        walker.apply_control(control)
        return True
    except (AttributeError, RuntimeError):
        return False


def recovery_step_before_vehicles(
    walker,
    current_location: carla.Location,
    direction: Tuple[float, float],
    requested_step: float,
    vehicles,
) -> Tuple[float, Optional[int]]:
    """Clamp a scripted step before its swept walker footprint reaches a car."""
    step = max(0.0, float(requested_step))
    if step <= 1e-6:
        return step, None
    if vehicles is None:
        # None means the vehicle inventory could not be validated. An empty
        # iterable means it was validated and contains no vehicles.
        return 0.0, None
    walker_geometry = actor_box_geometry(walker)
    walker_radius = actor_planar_bounding_radius(walker)
    if walker_geometry is None or walker_radius is None:
        # A scripted transform must fail closed when its swept footprint cannot
        # be validated. Normal velocity/control commands remain active.
        return 0.0, None
    _, walker_min_z, walker_max_z = walker_geometry
    start = (float(current_location.x), float(current_location.y))
    end = (
        start[0] + direction[0] * step,
        start[1] + direction[1] * step,
    )
    safe_step = step
    blocking_vehicle_id = None
    sample_count = max(1, int(math.ceil(step / 0.05)))
    if sample_count > 500:
        # Preserve the advertised <=5 cm sweep resolution for custom values
        # instead of silently sampling a very large teleport too coarsely.
        return 0.0, None
    sample_spacing = step / sample_count

    for vehicle in vehicles:
        try:
            vehicle_id = int(vehicle.id)
        except (AttributeError, RuntimeError):
            return 0.0, None
        try:
            if not vehicle.is_alive:
                continue
        except (AttributeError, RuntimeError):
            return 0.0, vehicle_id
        try:
            vehicle_location = vehicle.get_location()
            vehicle_radius = actor_planar_bounding_radius(vehicle)
        except (AttributeError, RuntimeError):
            return 0.0, vehicle_id
        if vehicle_radius is None:
            return 0.0, vehicle_id
        if point_to_segment_distance(
            (float(vehicle_location.x), float(vehicle_location.y)),
            start,
            end,
        ) > walker_radius + vehicle_radius + 0.10:
            continue

        vehicle_geometry = actor_box_geometry(vehicle)
        if vehicle_geometry is None:
            if abs(
                float(vehicle_location.z - current_location.z)
            ) <= DEFAULT_COLLISION_CENTER_VERTICAL_TOLERANCE_M:
                return 0.0, vehicle_id
            continue
        footprint, vehicle_min_z, vehicle_max_z = vehicle_geometry
        if not vertical_intervals_overlap(
            walker_min_z,
            walker_max_z,
            vehicle_min_z,
            vehicle_max_z,
            DEFAULT_COLLISION_VERTICAL_CLEARANCE_M,
        ):
            continue

        for sample_index in range(sample_count + 1):
            distance_along_path = sample_index * sample_spacing
            sample = (
                start[0] + direction[0] * distance_along_path,
                start[1] + direction[1] * distance_along_path,
            )
            if point_to_footprint_distance(
                sample,
                footprint,
            ) > walker_radius + 0.02:
                continue
            candidate_step = max(0.0, distance_along_path - sample_spacing)
            if candidate_step < safe_step:
                safe_step = candidate_step
                blocking_vehicle_id = vehicle_id
            break

    return safe_step, blocking_vehicle_id


def apply_bounded_stall_recovery(
    walker,
    target_x: float,
    target_y: float,
    maximum_step: float,
    requested_speed: float,
    command_scale: float,
    vehicles=None,
) -> Optional[Tuple[float, Optional[int]]]:
    """Advance a stalled walker by a small collision-conscious XY step."""
    if maximum_step <= 0.0:
        return None
    try:
        transform = walker.get_transform()
    except (AttributeError, RuntimeError):
        return None
    direction = normalized_xy(
        target_x - float(transform.location.x),
        target_y - float(transform.location.y),
    )
    if direction is None:
        return None
    remaining = math.hypot(
        target_x - float(transform.location.x),
        target_y - float(transform.location.y),
    )
    step = min(float(maximum_step), remaining)
    if step <= 1e-6:
        return None
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    applied_step, blocking_vehicle_id = recovery_step_before_vehicles(
        walker,
        transform.location,
        direction,
        step,
        vehicles,
    )
    if applied_step <= 1e-6:
        if not walk_toward(
            walker,
            target_x,
            target_y,
            requested_speed,
            command_scale,
        ):
            return None
        return 0.0, blocking_vehicle_id
    try:
        walker.set_transform(
            carla.Transform(
                carla.Location(
                    x=float(
                        transform.location.x + direction[0] * applied_step
                    ),
                    y=float(
                        transform.location.y + direction[1] * applied_step
                    ),
                    z=float(transform.location.z),
                ),
                carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
            )
        )
    except RuntimeError:
        return None
    if not walk_toward(
        walker,
        target_x,
        target_y,
        requested_speed,
        command_scale,
    ):
        return None
    return applied_step, blocking_vehicle_id


def state_actor_alive(state: PedestrianState) -> bool:
    try:
        return bool(state.actor.is_alive)
    except (AttributeError, RuntimeError):
        return False


def vehicle_near_location(
    world: carla.World,
    location: carla.Location,
    maximum_distance: float,
) -> Optional[int]:
    if maximum_distance <= 0.0:
        return None
    try:
        vehicles = world.get_actors().filter("vehicle.*")
    except RuntimeError as exc:
        raise RuntimeError(
            "vehicle inventory is unavailable during respawn clearance check"
        ) from exc
    for vehicle in vehicles:
        try:
            vehicle_id = int(vehicle.id)
            if not vehicle.is_alive:
                continue
        except (AttributeError, RuntimeError):
            continue
        geometry = actor_box_geometry(vehicle)
        if geometry is not None:
            footprint, minimum_z, maximum_z = geometry
            if not (
                minimum_z - DEFAULT_RESPAWN_VERTICAL_CLEARANCE_M
                <= float(location.z)
                <= maximum_z + DEFAULT_RESPAWN_VERTICAL_CLEARANCE_M
            ):
                continue
            if point_to_footprint_distance(
                (float(location.x), float(location.y)),
                footprint,
            ) <= maximum_distance:
                return vehicle_id
            continue
        try:
            vehicle_location = vehicle.get_location()
        except (AttributeError, RuntimeError):
            LOG.warning(
                "Deferring respawn because vehicle id=%d geometry is "
                "temporarily unavailable",
                vehicle_id,
            )
            return vehicle_id
        if (
            abs(float(vehicle_location.z - location.z))
            <= DEFAULT_RESPAWN_VERTICAL_CLEARANCE_M
            and planar_distance(vehicle_location, location) <= maximum_distance
        ):
            return vehicle_id
    return None


def nearby_vehicle_contact(
    world: carla.World,
    state: PedestrianState,
    maximum_distance: float,
    allow_center_distance: bool,
    vehicles=None,
) -> Optional[Tuple[int, str]]:
    """Return a vertically valid box overlap or sensorless proximity hit."""
    if not state_actor_alive(state):
        return None
    try:
        pedestrian_location = state.actor.get_location()
    except (AttributeError, RuntimeError):
        return None
    if vehicles is None:
        try:
            vehicles = world.get_actors().filter("vehicle.*")
        except RuntimeError:
            return None
    pedestrian_radius = actor_planar_bounding_radius(state.actor)
    for vehicle in vehicles:
        try:
            if not vehicle.is_alive:
                continue
            vehicle_location = vehicle.get_location()
            vehicle_radius = actor_planar_bounding_radius(vehicle)
            if pedestrian_radius is not None and vehicle_radius is not None:
                rejection_distance = pedestrian_radius + vehicle_radius + 0.05
                if allow_center_distance:
                    rejection_distance = max(
                        rejection_distance,
                        maximum_distance,
                    )
                if planar_distance(
                    pedestrian_location,
                    vehicle_location,
                ) > rejection_distance:
                    continue
            if (
                actor_bounding_boxes_overlap(
                    vehicle,
                    state.actor,
                    horizontal_margin=0.05,
                    vertical_margin=0.05,
                )
                or bounding_box_contains_location(
                    vehicle,
                    pedestrian_location,
                )
                or bounding_box_contains_location(
                    state.actor,
                    vehicle_location,
                )
            ):
                return int(vehicle.id), "bounding-box overlap"
            if not allow_center_distance or maximum_distance <= 0.0:
                continue
            vehicle_geometry = actor_box_geometry(vehicle)
            pedestrian_geometry = actor_box_geometry(state.actor)
            if vehicle_geometry is None or pedestrian_geometry is None:
                if abs(
                    float(vehicle_location.z - pedestrian_location.z)
                ) > DEFAULT_COLLISION_CENTER_VERTICAL_TOLERANCE_M:
                    continue
            elif not vertical_intervals_overlap(
                vehicle_geometry[1],
                vehicle_geometry[2],
                pedestrian_geometry[1],
                pedestrian_geometry[2],
                DEFAULT_COLLISION_VERTICAL_CLEARANCE_M,
            ):
                continue
            if planar_distance(
                pedestrian_location,
                vehicle_location,
            ) <= maximum_distance:
                return int(vehicle.id), "sensorless center-distance fallback"
        except (AttributeError, RuntimeError):
            continue
    return None


def nearby_vehicle_id(
    world: carla.World,
    state: PedestrianState,
    maximum_distance: float,
    allow_center_distance: bool = True,
) -> Optional[int]:
    """Compatibility wrapper returning only the detected vehicle actor ID."""
    contact = nearby_vehicle_contact(
        world,
        state,
        maximum_distance,
        allow_center_distance,
    )
    return None if contact is None else contact[0]


def filtered_active_acceleration(
    state: PedestrianState,
    acceleration: carla.Vector3D,
    smoothing: float,
    acceleration_limit: float,
) -> Tuple[float, float]:
    sample = clamped_acceleration_xy(acceleration, acceleration_limit)
    previous = state.filtered_ego_acceleration
    if previous is None:
        filtered = sample
    else:
        weight = max(0.0, min(1.0, float(smoothing)))
        filtered = (
            previous[0] + weight * (sample[0] - previous[0]),
            previous[1] + weight * (sample[1] - previous[1]),
        )
    magnitude = math.hypot(filtered[0], filtered[1])
    if magnitude > float(acceleration_limit):
        scale = float(acceleration_limit) / magnitude
        filtered = filtered[0] * scale, filtered[1] * scale
    state.filtered_ego_acceleration = filtered
    return filtered


def draw_intercept_plan(
    world: Optional[carla.World],
    state: PedestrianState,
    ego_location: carla.Location,
    ego_velocity: carla.Vector3D,
    acceleration_xy: Tuple[float, float],
    prediction_time: float,
    pedestrian_location: carla.Location,
    simulation_time: float,
    args: argparse.Namespace,
) -> None:
    """Draw ephemeral trajectory/crossing geometry without changing the clock."""
    if (
        not args.intercept_debug
        or world is None
        or state.crossing_origin is None
        or state.crossing_endpoint is None
    ):
        return
    if (
        state.last_debug_draw_time is not None
        and simulation_time - state.last_debug_draw_time
        < DEFAULT_DEBUG_DRAW_INTERVAL_SECONDS
    ):
        return
    state.last_debug_draw_time = float(simulation_time)
    lifetime = max(0.3, 1.5 * DEFAULT_DEBUG_DRAW_INTERVAL_SECONDS)
    debug_height = float(pedestrian_location.z) + 0.15
    try:
        crossing_start = carla.Location(
            x=state.crossing_origin[0],
            y=state.crossing_origin[1],
            z=debug_height,
        )
        crossing_end = carla.Location(
            x=state.crossing_endpoint[0],
            y=state.crossing_endpoint[1],
            z=debug_height,
        )
        world.debug.draw_line(
            crossing_start,
            crossing_end,
            thickness=0.06,
            color=carla.Color(255, 64, 64),
            life_time=lifetime,
            persistent_lines=False,
        )
        world.debug.draw_point(
            crossing_end,
            size=0.14,
            color=carla.Color(255, 255, 0),
            life_time=lifetime,
            persistent_lines=False,
        )
        segment_count = 8
        contact_offset = max(0.0, float(state.front_contact_offset))
        initial_tangent = normalized_xy(
            float(ego_velocity.x),
            float(ego_velocity.y),
        ) or state.ego_path_direction
        previous = carla.Location(
            x=float(ego_location.x) + contact_offset * initial_tangent[0],
            y=float(ego_location.y) + contact_offset * initial_tangent[1],
            z=float(ego_location.z) + 0.15,
        )
        for index in range(1, segment_count + 1):
            sample_time = max(0.0, float(prediction_time)) * index / segment_count
            sample_x, sample_y = predicted_ego_xy(
                ego_location,
                ego_velocity,
                acceleration_xy,
                sample_time,
            )
            sample_tangent = normalized_xy(
                float(ego_velocity.x) + acceleration_xy[0] * sample_time,
                float(ego_velocity.y) + acceleration_xy[1] * sample_time,
            ) or state.ego_path_direction
            sample_x += contact_offset * sample_tangent[0]
            sample_y += contact_offset * sample_tangent[1]
            current = carla.Location(
                x=sample_x,
                y=sample_y,
                z=float(ego_location.z) + 0.15,
            )
            world.debug.draw_line(
                previous,
                current,
                thickness=0.04,
                color=carla.Color(64, 192, 255),
                life_time=lifetime,
                persistent_lines=False,
            )
            previous = current
    except (AttributeError, RuntimeError, TypeError):
        # Debug visualization must never affect blocker control.
        return


def observe_near_miss(
    state: PedestrianState,
    ego,
    ego_location: carla.Location,
    ego_velocity: carla.Vector3D,
    pedestrian_location: carla.Location,
    longitudinal_acceleration: float,
    args: argparse.Namespace,
) -> Optional[str]:
    """Latch a qualified stop-short or collision-free pass at L2."""
    if state.crossing_endpoint is None or state.ego_path_direction is None:
        return None
    target_x, target_y = state.crossing_endpoint
    path_x, path_y = state.ego_path_direction
    signed_distance = -ego_contact_reference_line_value(
        ego_location,
        ego_velocity,
        (target_x, target_y),
        (path_x, path_y),
        state.front_contact_offset,
    )
    center_distance = math.hypot(
        float(pedestrian_location.x) - float(ego_location.x),
        float(pedestrian_location.y) - float(ego_location.y),
    )
    clearance = actor_collision_clearance(
        ego,
        state.actor,
        ego_velocity,
        direction_override=state.ego_path_direction,
    )
    surface_gap = max(0.0, center_distance - clearance)
    if (
        state.minimum_ego_surface_gap is None
        or surface_gap < state.minimum_ego_surface_gap
    ):
        state.minimum_ego_surface_gap = surface_gap
    if longitudinal_acceleration <= -float(args.hard_brake_deceleration):
        state.hard_brake_seen = True

    ego_speed = math.hypot(float(ego_velocity.x), float(ego_velocity.y))
    previous_signed = state.last_ego_intercept_signed_distance
    state.last_ego_intercept_signed_distance = signed_distance
    crossed_line = (
        previous_signed is not None
        and previous_signed > 0.0
        and signed_distance <= 0.0
    )
    stopped_short = (
        state.hard_brake_seen
        and ego_speed <= float(args.stopped_ego_speed)
        and signed_distance >= -float(args.intercept_arrival_tolerance)
        and surface_gap <= float(args.near_miss_distance)
    )
    passed_nearby = (
        (crossed_line or signed_distance <= -max(0.25, clearance))
        and state.minimum_ego_surface_gap is not None
        and state.minimum_ego_surface_gap <= float(args.near_miss_distance)
    )
    reason = None
    if stopped_short:
        reason = (
            "near miss: ego hard-braked and stopped %.2f m before/near L2 "
            "(closest footprint gap %.2f m)"
            % (max(0.0, signed_distance), state.minimum_ego_surface_gap)
        )
    elif passed_nearby:
        reason = (
            "near miss: ego passed L2 without contact "
            "(closest footprint gap %.2f m)"
            % state.minimum_ego_surface_gap
        )
    if reason is not None:
        state.pending_near_miss_reason = reason
    if state.intercept_reached:
        return state.pending_near_miss_reason
    return None


def qualified_passed_near_miss_reason(
    state: PedestrianState,
    ego,
    ego_location: carla.Location,
    ego_velocity: carla.Vector3D,
    pedestrian_location: carla.Location,
    args: argparse.Namespace,
) -> Optional[str]:
    """Return a pass near miss immediately when the front window was missed."""
    if state.minimum_ego_surface_gap is None:
        center_distance = math.hypot(
            float(pedestrian_location.x) - float(ego_location.x),
            float(pedestrian_location.y) - float(ego_location.y),
        )
        clearance = actor_collision_clearance(
            ego,
            state.actor,
            ego_velocity,
            direction_override=state.ego_path_direction,
        )
        state.minimum_ego_surface_gap = max(0.0, center_distance - clearance)
    if state.minimum_ego_surface_gap <= float(args.near_miss_distance):
        return (
            "near miss: ego front passed L2 before pedestrian arrival "
            "(closest footprint gap %.2f m)"
            % state.minimum_ego_surface_gap
        )
    return None


def update_active_pedestrian(
    state: PedestrianState,
    ego,
    simulation_time: float,
    args: argparse.Namespace,
    vehicles=None,
    world: Optional[carla.World] = None,
) -> Optional[ActiveUpdateResult]:
    if not state_actor_alive(state):
        return ActiveUpdateResult("respawn", "walker actor became unavailable")
    try:
        pedestrian_location = state.actor.get_location()
    except RuntimeError:
        return None

    active_start = simulation_time if state.active_since is None else state.active_since
    if simulation_time - float(active_start) >= args.active_timeout:
        return ActiveUpdateResult(
            "respawn",
            "intercept encounter timed out before collision or near miss",
        )
    if (
        state.crossing_origin is None
        or state.crossing_direction is None
        or state.crossing_endpoint is None
        or state.crossing_distance is None
        or state.ego_path_direction is None
    ):
        return ActiveUpdateResult("respawn", "crossing geometry became unavailable")
    if ego is None:
        return None
    if state.active_ego_id is not None and int(ego.id) != state.active_ego_id:
        return ActiveUpdateResult("respawn", "ego actor changed during intercept")

    try:
        ego_location = ego.get_location()
        ego_velocity = ego.get_velocity()
        ego_acceleration = ego.get_acceleration()
    except (AttributeError, RuntimeError):
        return None
    acceleration_xy = filtered_active_acceleration(
        state,
        ego_acceleration,
        args.acceleration_smoothing,
        args.prediction_acceleration_limit,
    )
    longitudinal_acceleration = (
        acceleration_xy[0] * state.ego_path_direction[0]
        + acceleration_xy[1] * state.ego_path_direction[1]
    )
    if longitudinal_acceleration <= -float(args.hard_brake_deceleration):
        state.hard_brake_seen = True
    state.last_separation = planar_distance(pedestrian_location, ego_location)

    previous_motion_time = state.last_motion_update_time
    state.last_motion_update_time = float(simulation_time)
    motion_elapsed = (
        0.0
        if previous_motion_time is None
        else max(0.0, float(simulation_time - previous_motion_time))
    )

    prediction_time = 0.0
    if not state.intercept_reached:
        replan = solve_crossing_line_intercept(
            ego_location,
            ego_velocity,
            acceleration_xy,
            pedestrian_location,
            state.crossing_origin,
            state.crossing_direction,
            state.ego_path_direction,
            args.max_intercept_time,
            state.front_contact_offset,
        )
        if replan is not None:
            if (
                replan.perpendicular_error_degrees
                > float(args.active_perpendicular_tolerance)
            ):
                return ActiveUpdateResult(
                    "respawn",
                    "ego trajectory turned {:.1f} degrees away from the "
                    "locked perpendicular intercept".format(
                        replan.perpendicular_error_degrees
                    ),
                )
            if (
                replan.required_pedestrian_speed
                > float(args.pedestrian_speed) + 1.0e-3
            ):
                return ActiveUpdateResult(
                    "respawn",
                    "synchronized intercept now requires {:.2f} m/s, above "
                    "the {:.2f} m/s pedestrian limit".format(
                        replan.required_pedestrian_speed,
                        args.pedestrian_speed,
                    ),
                )
            candidate_crossing_distance = (
                (replan.target_x - state.crossing_origin[0])
                * state.crossing_direction[0]
                + (replan.target_y - state.crossing_origin[1])
                * state.crossing_direction[1]
            )
            current_crossing_progress = (
                (float(pedestrian_location.x) - state.crossing_origin[0])
                * state.crossing_direction[0]
                + (float(pedestrian_location.y) - state.crossing_origin[1])
                * state.crossing_direction[1]
            )
            if (
                candidate_crossing_distance
                < current_crossing_progress
                - float(args.intercept_arrival_tolerance)
            ):
                return ActiveUpdateResult(
                    "respawn",
                    "updated L2 moved behind the committed pedestrian path",
                )
            prediction_time = replan.time_seconds
            state.crossing_endpoint = (replan.target_x, replan.target_y)
            state.crossing_distance = max(0.0, candidate_crossing_distance)
            state.commanded_pedestrian_speed = max(
                0.0,
                replan.required_pedestrian_speed,
            )
            longitudinal_acceleration = replan.longitudinal_acceleration
        else:
            contact_line_value = ego_contact_reference_line_value(
                ego_location,
                ego_velocity,
                state.crossing_origin,
                state.ego_path_direction,
                state.front_contact_offset,
            )
            if contact_line_value >= 0.0:
                stop_walker(state.actor)
                near_miss_reason = qualified_passed_near_miss_reason(
                    state,
                    ego,
                    ego_location,
                    ego_velocity,
                    pedestrian_location,
                    args,
                )
                if near_miss_reason is not None:
                    return ActiveUpdateResult("hold", near_miss_reason)
                return ActiveUpdateResult(
                    "respawn",
                    "ego impact reference passed L2 before pedestrian arrival; "
                    "recycling to prevent a side impact",
                )
            current_ego_direction = normalized_xy(
                float(ego_velocity.x),
                float(ego_velocity.y),
            )
            if current_ego_direction is not None:
                perpendicular_residual = abs(
                    state.crossing_direction[0] * current_ego_direction[0]
                    + state.crossing_direction[1] * current_ego_direction[1]
                )
                perpendicular_error = math.degrees(
                    math.asin(max(0.0, min(1.0, perpendicular_residual)))
                )
                if perpendicular_error > float(
                    args.active_perpendicular_tolerance
                ):
                    stop_walker(state.actor)
                    return ActiveUpdateResult(
                        "respawn",
                        "ego trajectory turned {:.1f} degrees away from the "
                        "locked perpendicular intercept and no future "
                        "intersection remains".format(perpendicular_error),
                    )
            if state.commanded_pedestrian_speed is None:
                state.commanded_pedestrian_speed = float(args.min_pedestrian_speed)

    traveled_x = float(pedestrian_location.x) - state.crossing_origin[0]
    traveled_y = float(pedestrian_location.y) - state.crossing_origin[1]
    crossing_progress = (
        traveled_x * state.crossing_direction[0]
        + traveled_y * state.crossing_direction[1]
    )
    remaining_distance = math.hypot(
        state.crossing_endpoint[0] - float(pedestrian_location.x),
        state.crossing_endpoint[1] - float(pedestrian_location.y),
    )
    if remaining_distance <= float(args.intercept_arrival_tolerance):
        if not state.intercept_reached:
            LOG.info(
                "Pedestrian #%d id=%d reached L2=(%.2f, %.2f); waiting for "
                "collision or qualified near miss",
                state.index,
                state.actor.id,
                state.crossing_endpoint[0],
                state.crossing_endpoint[1],
            )
        state.intercept_reached = True
        stop_walker(state.actor)
    else:
        command_speed = min(
            float(args.pedestrian_speed),
            max(0.0, float(state.commanded_pedestrian_speed or 0.0)),
            max(0.0, 0.8 * remaining_distance * float(args.update_hz)),
        )
        command_succeeded = walk_toward(
            state.actor,
            state.crossing_endpoint[0],
            state.crossing_endpoint[1],
            command_speed,
            args.walker_control_speed_scale,
        )
        if command_succeeded:
            state.motion_command_failures = 0
        else:
            state.motion_command_failures += 1
            LOG.warning(
                "Pedestrian #%d id=%s motion command failed (%d/%d)",
                state.index,
                getattr(state.actor, "id", "unknown"),
                state.motion_command_failures,
                args.max_motion_command_failures,
            )
            if state.motion_command_failures >= args.max_motion_command_failures:
                return ActiveUpdateResult(
                    "respawn",
                    "motion commands repeatedly failed",
                )

        expected_progress = max(
            0.02,
            min(
                float(args.motion_stall_min_progress),
                command_speed * float(args.motion_stall_timeout) * 0.5,
            ),
        )
        stall_detected = False
        if state.scripted_recovery_active:
            stall_detected = True
        elif state.last_progress is None:
            state.last_progress = crossing_progress
            state.last_progress_time = float(simulation_time)
        elif crossing_progress >= state.last_progress + expected_progress:
            state.last_progress = crossing_progress
            state.last_progress_time = float(simulation_time)
        elif state.last_progress_time is None:
            state.last_progress_time = float(simulation_time)
        elif simulation_time - state.last_progress_time >= args.motion_stall_timeout:
            stall_detected = True

        if stall_detected:
            if args.stall_recovery_step <= 0.0:
                return ActiveUpdateResult(
                    "respawn",
                    "motion stalled with scripted recovery disabled",
                )
            if state.scripted_recovery_started_at is None:
                state.scripted_recovery_started_at = float(simulation_time)
            elif (
                simulation_time - state.scripted_recovery_started_at
                >= args.max_scripted_recovery_time
            ):
                return ActiveUpdateResult(
                    "respawn",
                    "scripted stall-recovery time budget expired",
                )
            if not state.scripted_recovery_active:
                state.scripted_recovery_active = True
                LOG.warning(
                    "Pedestrian #%d id=%d motion stalled at %.2f/%.2f m; "
                    "entering bounded scripted recovery",
                    state.index,
                    state.actor.id,
                    crossing_progress,
                    state.crossing_distance,
                )
            recovery_step = min(
                args.stall_recovery_step,
                command_speed * motion_elapsed,
                remaining_distance,
            )
            if recovery_step > 1.0e-6:
                recovery_result = apply_bounded_stall_recovery(
                    state.actor,
                    state.crossing_endpoint[0],
                    state.crossing_endpoint[1],
                    recovery_step,
                    command_speed,
                    args.walker_control_speed_scale,
                    vehicles=vehicles,
                )
                if recovery_result is None:
                    return ActiveUpdateResult(
                        "respawn",
                        "motion stalled and bounded recovery failed",
                    )
                applied_step, blocking_vehicle_id = recovery_result
                state.stall_recovery_count += 1
                state.last_progress = max(
                    crossing_progress,
                    crossing_progress + applied_step,
                )
                state.last_progress_time = float(simulation_time)
                if (
                    state.stall_recovery_count == 1
                    or state.stall_recovery_count % 20 == 0
                ):
                    LOG.warning(
                        "Pedestrian #%d id=%d scripted recovery progress "
                        "%.2f/%.2f m requested_step=%.2f m applied_step=%.2f m "
                        "blocked_by_vehicle=%s count=%d",
                        state.index,
                        state.actor.id,
                        crossing_progress + applied_step,
                        state.crossing_distance,
                        recovery_step,
                        applied_step,
                        "none" if blocking_vehicle_id is None else blocking_vehicle_id,
                        state.stall_recovery_count,
                    )

    draw_intercept_plan(
        world,
        state,
        ego_location,
        ego_velocity,
        acceleration_xy,
        prediction_time,
        pedestrian_location,
        simulation_time,
        args,
    )
    near_miss_reason = observe_near_miss(
        state,
        ego,
        ego_location,
        ego_velocity,
        pedestrian_location,
        longitudinal_acceleration,
        args,
    )
    if near_miss_reason is not None:
        return ActiveUpdateResult("hold", near_miss_reason)
    return None


def candidate_decision(
    state: PedestrianState,
    ego,
    args: argparse.Namespace,
) -> Optional[TriggerDecision]:
    if state.state != STATE_WAITING or not state_actor_alive(state):
        return None
    try:
        ego_transform = ego.get_transform()
        ego_velocity = ego.get_velocity()
        ego_acceleration = ego.get_acceleration()
        pedestrian_location = state.actor.get_location()
    except (AttributeError, RuntimeError):
        return None
    clearance = actor_collision_clearance(ego, state.actor, ego_velocity)
    front_contact_offset = None
    if args.impact_target == "front":
        front_contact_offset = front_impact_contact_offset(
            ego,
            state.actor,
            ego_velocity,
            args.front_impact_margin,
        )
        if front_contact_offset is None:
            return None
    return evaluate_trigger(
        ego_transform=ego_transform,
        ego_velocity=ego_velocity,
        ego_acceleration=ego_acceleration,
        pedestrian_location=pedestrian_location,
        minimum_pedestrian_speed=args.min_pedestrian_speed,
        maximum_pedestrian_speed=args.pedestrian_speed,
        min_ego_speed=args.min_ego_speed,
        min_closing_speed=args.min_closing_speed,
        trigger_distance=args.trigger_distance,
        max_approach_angle_degrees=args.max_approach_angle,
        max_lateral_offset=args.max_lateral_offset,
        reaction_time=args.reaction_time,
        max_brake_deceleration=args.max_brake_deceleration,
        braking_margin=args.braking_margin,
        minimum_intercept_time=args.min_intercept_time,
        max_intercept_time=args.max_intercept_time,
        prediction_acceleration_limit=args.prediction_acceleration_limit,
        collision_clearance=clearance,
        front_contact_offset=front_contact_offset,
    )


def activate_pedestrian(
    state: PedestrianState,
    ego,
    decision: TriggerDecision,
    simulation_time: float,
    args: argparse.Namespace,
) -> bool:
    try:
        pedestrian_location = state.actor.get_location()
    except (AttributeError, RuntimeError):
        return False
    crossing_direction = (
        decision.intercept.pedestrian_direction_x,
        decision.intercept.pedestrian_direction_y,
    )
    crossing_distance = decision.intercept.pedestrian_distance
    crossing_endpoint = (
        decision.intercept.target_x,
        decision.intercept.target_y,
    )
    state.state = STATE_ACTIVE
    state.active_since = simulation_time
    state.active_ego_id = int(ego.id)
    state.last_separation = decision.separation
    state.crossing_origin = (
        float(pedestrian_location.x),
        float(pedestrian_location.y),
    )
    state.crossing_direction = crossing_direction
    state.crossing_endpoint = crossing_endpoint
    state.crossing_distance = crossing_distance
    state.ego_path_direction = (
        decision.intercept.tangent_x,
        decision.intercept.tangent_y,
    )
    state.front_contact_offset = decision.intercept.front_contact_offset
    state.commanded_pedestrian_speed = (
        decision.intercept.required_pedestrian_speed
    )
    state.filtered_ego_acceleration = (
        decision.intercept.acceleration_x,
        decision.intercept.acceleration_y,
    )
    state.last_ego_intercept_signed_distance = decision.ego_travel
    state.minimum_ego_surface_gap = None
    state.hard_brake_seen = False
    state.intercept_reached = False
    state.pending_near_miss_reason = None
    state.last_progress = 0.0
    state.last_progress_time = float(simulation_time)
    state.motion_command_failures = 0
    state.stall_recovery_count = 0
    state.scripted_recovery_active = False
    state.scripted_recovery_started_at = None
    state.last_motion_update_time = float(simulation_time)
    if not face_walker_toward(
        state.actor,
        crossing_endpoint[0],
        crossing_endpoint[1],
    ):
        LOG.warning(
            "Pedestrian #%d id=%d could not align its body with the crossing "
            "path; continuing with directional controls",
            state.index,
            state.actor.id,
        )
    if not walk_toward(
        state.actor,
        crossing_endpoint[0],
        crossing_endpoint[1],
        decision.intercept.required_pedestrian_speed,
        args.walker_control_speed_scale,
    ):
        LOG.warning(
            "Pedestrian #%d id=%d activation motion command failed; "
            "remaining WAITING",
            state.index,
            state.actor.id,
        )
        stop_walker(state.actor)
        state.state = STATE_WAITING
        reset_activation_fields(state)
        return False
    LOG.info(
        "Pedestrian #%d id=%d state=ACTIVE ego_id=%d "
        "timed_pedestrian_speed=%.2f m/s max_pedestrian_speed=%.2f m/s "
        "animation_speed=%.2f m/s "
        "ego_speed=%.2f m/s "
        "separation=%.2f m closing=%.2f m/s lateral=%.2f m angle=%.1f deg "
        "intercept=%.2f s ego_acceleration=(%.2f, %.2f) m/s2 "
        "ego_travel=%.2f m effective_travel=%.2f m "
        "stopping_distance=%.2f m L2=(%.2f, %.2f) "
        "impact_target=%s front_contact_offset=%.2f m "
        "pedestrian_direction=(%.3f, %.3f) "
        "ego_tangent=(%.3f, %.3f) crossing_distance=%.2f m",
        state.index,
        state.actor.id,
        ego.id,
        decision.intercept.required_pedestrian_speed,
        args.pedestrian_speed,
        walker_animation_speed(
            decision.intercept.required_pedestrian_speed,
            args.walker_control_speed_scale,
        ),
        decision.ego_speed,
        decision.separation,
        decision.closing_speed,
        decision.lateral_offset,
        decision.approach_angle_degrees,
        decision.intercept.time_seconds,
        decision.intercept.acceleration_x,
        decision.intercept.acceleration_y,
        decision.ego_travel,
        decision.effective_travel,
        decision.stopping_distance,
        decision.intercept.target_x,
        decision.intercept.target_y,
        args.impact_target,
        decision.intercept.front_contact_offset,
        crossing_direction[0],
        crossing_direction[1],
        decision.intercept.tangent_x,
        decision.intercept.tangent_y,
        crossing_distance,
    )
    return True


def destroy_actor(actor, actor_kind: str) -> bool:
    if actor is None:
        return True
    try:
        if not actor.is_alive:
            return True
        if actor.destroy():
            LOG.info("Destroyed %s id=%d", actor_kind, actor.id)
            return True
    except RuntimeError as exc:
        LOG.warning("Unable to destroy %s id=%s: %s", actor_kind, actor.id, exc)
        return False
    LOG.warning("CARLA did not confirm destruction of %s id=%s", actor_kind, actor.id)
    return False


def passively_wait_for_network_profile_snapshot(
    world: carla.World,
    context: str,
    timeout_seconds: float = NETWORK_PROFILE_PASSIVE_WAIT_TIMEOUT_SECONDS,
) -> bool:
    """Wait for one externally driven snapshot without assuming clock ownership."""
    try:
        snapshot = world.wait_for_tick(float(timeout_seconds))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        LOG.debug(
            "Passive network-profile wait timed out during %s: %s",
            context,
            exc,
        )
        return False
    if snapshot is None:
        LOG.debug(
            "Passive network-profile wait returned no snapshot during %s",
            context,
        )
        return False
    return True


def registered_network_profile_actor(
    world: carla.World,
    actor_id: int,
    expected_role_name: Optional[str] = None,
):
    """Resolve one live reserved actor from CARLA's current actor registry."""
    try:
        actor = world.get_actor(int(actor_id))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "unable to resolve network-profile metadata actor {}: {}".format(
                actor_id,
                exc,
            )
        ) from exc
    if actor is None:
        return None
    try:
        if not actor.is_alive:
            return None
        if str(actor.type_id) != NETWORK_PROFILE_BLUEPRINT_ID:
            raise RuntimeError(
                "actor {} type changed to {!r}; refusing to operate on a "
                "non-metadata actor".format(actor_id, actor.type_id)
            )
        role_name = str(actor.attributes.get("role_name", ""))
        if expected_role_name is not None:
            if role_name != str(expected_role_name):
                raise RuntimeError(
                    "actor {} role changed from {!r} to {!r}; refusing to "
                    "destroy a potentially unrelated actor".format(
                        actor_id,
                        expected_role_name,
                        role_name,
                    )
                )
        elif not role_name.startswith(NETWORK_PROFILE_ROLE_PREFIX):
            raise RuntimeError(
                "actor {} no longer carries spawn_blocker network-profile "
                "metadata".format(actor_id)
            )
        if sensor_is_listening(actor):
            raise RuntimeError(
                "network-profile metadata actor {} is unexpectedly "
                "listening".format(actor_id)
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            "unable to validate network-profile metadata actor {}: {}".format(
                actor_id,
                exc,
            )
        ) from exc
    return actor


def destroy_registered_network_profile_actor(
    world: carla.World,
    actor_id: int,
    expected_role_name: str,
    actor_kind: str,
    attempts: int = NETWORK_PROFILE_REGISTRY_RETRY_ATTEMPTS,
) -> bool:
    """Destroy by registry ID and confirm absence on a later passive snapshot."""
    actor_id = int(actor_id)
    attempts = max(1, int(attempts))
    last_error = None

    for attempt in range(attempts):
        try:
            actor = registered_network_profile_actor(
                world,
                actor_id,
                expected_role_name,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            break

        if actor is not None:
            try:
                if not actor.destroy():
                    last_error = "CARLA did not acknowledge the destroy request"
            except RuntimeError as exc:
                last_error = str(exc)

        # An immediate None/dead result is not enough for a just-spawned actor:
        # require absence after a subsequently observed server snapshot.
        if not passively_wait_for_network_profile_snapshot(
            world,
            "{} id={} confirmation {}/{}".format(
                actor_kind,
                actor_id,
                attempt + 1,
                attempts,
            ),
        ):
            continue
        try:
            survivor = registered_network_profile_actor(
                world,
                actor_id,
                expected_role_name,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            break
        if survivor is None:
            LOG.info("Destroyed %s id=%d", actor_kind, actor_id)
            return True

    LOG.warning(
        "Unable to confirm destruction of %s id=%d after %d passive "
        "snapshot attempt(s)%s",
        actor_kind,
        actor_id,
        attempts,
        " (last error: {})".format(last_error) if last_error else "",
    )
    return False


def configure_network_profile_blueprint(world: carla.World, role_name: str):
    """Configure one invisible, unlistened GNSS metadata actor blueprint."""
    try:
        blueprint = world.get_blueprint_library().find(
            NETWORK_PROFILE_BLUEPRINT_ID
        )
    except (IndexError, RuntimeError) as exc:
        raise RuntimeError(
            "required network-profile metadata blueprint {!r} is "
            "unavailable".format(NETWORK_PROFILE_BLUEPRINT_ID)
        ) from exc
    if not blueprint.has_attribute("role_name"):
        raise RuntimeError(
            "network-profile metadata blueprint {!r} has no role_name "
            "attribute".format(NETWORK_PROFILE_BLUEPRINT_ID)
        )
    blueprint.set_attribute("role_name", str(role_name))
    if blueprint.has_attribute("sensor_tick"):
        blueprint.set_attribute(
            "sensor_tick",
            str(NETWORK_PROFILE_SENSOR_TICK_SECONDS),
        )
    return blueprint


def spawn_network_profile_metadata_actor(
    world: carla.World,
    role_name: str,
    x_coord: float,
    y_coord: float,
):
    """Spawn one parentless profile actor without registering a callback."""
    blueprint = configure_network_profile_blueprint(world, role_name)
    transform = carla.Transform(
        carla.Location(x=float(x_coord), y=float(y_coord), z=0.0),
        carla.Rotation(),
    )
    try:
        actor = world.spawn_actor(blueprint, transform)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "unable to spawn network-profile metadata role {!r}: {}".format(
                role_name,
                exc,
            )
        ) from exc
    if actor is None:
        raise RuntimeError(
            "network-profile metadata spawn returned no actor for role "
            "{!r}".format(role_name)
        )
    try:
        if sensor_is_listening(actor):
            raise RuntimeError(
                "network-profile metadata actor id={} unexpectedly started "
                "listening".format(actor.id)
            )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        destroy_actor(actor, "invalid network-profile metadata actor")
        raise
    return actor


def network_profile_matches_requested_configuration(
    profile,
    zones: Sequence[NetworkDegradationZone],
    start_active_sensors: bool,
) -> bool:
    """Return whether a discovered committed profile is safe to reuse."""
    if profile.start_active_sensors != bool(start_active_sensors):
        return False
    if len(profile.zones) != len(zones):
        return False
    for expected, actual in zip(zones, profile.zones):
        if (
            expected.index != actual.index
            or abs(expected.x - actual.x) > 1.0e-3
            or abs(expected.y - actual.y) > 1.0e-3
            or abs(expected.radius - actual.radius) > 1.0e-6
        ):
            return False
    return True


def resolve_network_profile_actor(world: carla.World, actor_id: int):
    """Resolve and validate one existing unlistened profile actor handle."""
    try:
        actor = world.get_actor(int(actor_id))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "unable to resolve network-profile metadata actor {}: {}".format(
                actor_id,
                exc,
            )
        ) from exc
    if actor is None:
        raise RuntimeError(
            "network-profile metadata actor {} disappeared during startup".format(
                actor_id,
            )
        )
    try:
        if not actor.is_alive:
            raise RuntimeError(
                "network-profile metadata actor {} is no longer alive".format(
                    actor_id,
                )
            )
        role_name = str(actor.attributes.get("role_name", ""))
        if not role_name.startswith(NETWORK_PROFILE_ROLE_PREFIX):
            raise RuntimeError(
                "actor {} no longer carries spawn_blocker network-profile "
                "metadata".format(actor_id)
            )
        if sensor_is_listening(actor):
            raise RuntimeError(
                "network-profile metadata actor {} is unexpectedly "
                "listening".format(actor_id)
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            "unable to validate network-profile metadata actor {}: {}".format(
                actor_id,
                exc,
            )
        ) from exc
    return actor


def matching_zone_only_network_profile_residue(
    world: carla.World,
    conflicts: Sequence[Tuple[int, str]],
    requested_zones: Sequence[NetworkDegradationZone],
) -> Optional[str]:
    """Return the token for a valid requested-zone subset with no manifest."""
    if not conflicts or len(conflicts) > len(requested_zones):
        return None

    requested_by_index = {zone.index: zone for zone in requested_zones}
    metadata_by_index = {}
    tokens = set()
    for actor_id, role_name in conflicts:
        try:
            if parse_manifest_role(role_name) is not None:
                return None
            metadata = parse_zone_role(role_name)
        except NetworkProfileError:
            return None
        if metadata is None or metadata.index in metadata_by_index:
            return None
        expected = requested_by_index.get(metadata.index)
        if expected is None:
            return None
        try:
            actor = registered_network_profile_actor(
                world,
                actor_id,
                role_name,
            )
            if actor is None:
                return None
            location = actor.get_transform().location
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        metadata_by_index[metadata.index] = (
            metadata,
            float(location.x),
            float(location.y),
        )
        tokens.add(metadata.session_token)

    if len(tokens) != 1:
        return None
    for zone_index, record in metadata_by_index.items():
        expected = requested_by_index[zone_index]
        metadata, actual_x, actual_y = record
        if (
            abs(metadata.radius - expected.radius) > 1.0e-6
            or abs(actual_x - expected.x) > 1.0e-3
            or abs(actual_y - expected.y) > 1.0e-3
        ):
            return None
    return next(iter(tokens))


def remove_stable_matching_zone_only_residue(
    world: carla.World,
    conflicts: Sequence[Tuple[int, str]],
    requested_zones: Sequence[NetworkDegradationZone],
) -> bool:
    """Remove only a stable orphan matching this launch's requested zones."""
    candidate_conflicts = tuple(sorted(conflicts))
    session_token = matching_zone_only_network_profile_residue(
        world,
        candidate_conflicts,
        requested_zones,
    )
    if session_token is None:
        return False

    confirmations = 0
    observations = 0
    wait_attempts = 0
    started_at = time.monotonic()
    candidate_stable_since = started_at
    deadline = started_at + NETWORK_PROFILE_STALE_MAX_OBSERVATION_SECONDS
    while (
        wait_attempts < NETWORK_PROFILE_STALE_MAX_SNAPSHOT_OBSERVATIONS
        and time.monotonic() < deadline
    ):
        remaining_seconds = max(0.0, deadline - time.monotonic())
        wait_attempts += 1
        if not passively_wait_for_network_profile_snapshot(
            world,
            "zone-only residue stability observation {}/{}".format(
                wait_attempts,
                NETWORK_PROFILE_STALE_MAX_SNAPSHOT_OBSERVATIONS,
            ),
            timeout_seconds=min(
                NETWORK_PROFILE_PASSIVE_WAIT_TIMEOUT_SECONDS,
                remaining_seconds,
            ),
        ):
            continue
        observations += 1
        observed_at = time.monotonic()
        try:
            current_conflicts = tuple(sorted(find_profile_actor_conflicts(world)))
        except NetworkProfileError:
            return False
        current_token = matching_zone_only_network_profile_residue(
            world,
            current_conflicts,
            requested_zones,
        )
        if current_token != session_token:
            # A manifest, malformed/mixed actor, unrelated zone, or complete
            # disappearance means this is not a stable orphan we may own.
            return False
        if current_conflicts == candidate_conflicts:
            confirmations += 1
        else:
            # Registry propagation may expose another valid zone on a later
            # snapshot. Adopt that safe subset and restart the stability count.
            candidate_conflicts = current_conflicts
            confirmations = 1
            candidate_stable_since = observed_at
        stable_seconds = observed_at - candidate_stable_since
        if (
            confirmations >= NETWORK_PROFILE_STALE_CONFIRMATION_TICKS
            and stable_seconds >= NETWORK_PROFILE_STALE_MIN_STABLE_SECONDS
        ):
            break

    stable_seconds = time.monotonic() - candidate_stable_since
    if (
        confirmations < NETWORK_PROFILE_STALE_CONFIRMATION_TICKS
        or stable_seconds < NETWORK_PROFILE_STALE_MIN_STABLE_SECONDS
    ):
        return False

    # One last immediate identity check closes the ordinary publication window
    # as tightly as CARLA's actor API permits before beginning cleanup.
    try:
        current_conflicts = tuple(sorted(find_profile_actor_conflicts(world)))
    except NetworkProfileError:
        return False
    if (
        current_conflicts != candidate_conflicts
        or matching_zone_only_network_profile_residue(
            world,
            current_conflicts,
            requested_zones,
        )
        != session_token
    ):
        return False

    LOG.warning(
        "Automatically removing %d stable zone-only network-profile "
        "metadata actor(s) from incomplete token=%s; no manifest appeared "
        "while the actor set remained unchanged for %.2f s across %d "
        "externally driven snapshot(s)",
        len(candidate_conflicts),
        session_token,
        stable_seconds,
        confirmations,
    )
    failed_ids = []
    for actor_id, role_name in reversed(candidate_conflicts):
        if not destroy_registered_network_profile_actor(
            world,
            actor_id,
            role_name,
            "orphaned network-profile zone metadata actor",
        ):
            failed_ids.append(actor_id)
    if failed_ids:
        raise RuntimeError(
            "unable to remove stable incomplete network-profile zone "
            "actor(s) {}".format(
                ", ".join(str(actor_id) for actor_id in failed_ids)
            )
        )

    try:
        survivors = find_profile_actor_conflicts(world)
    except NetworkProfileError as exc:
        raise RuntimeError(str(exc)) from exc
    if survivors:
        raise RuntimeError(
            "network-profile metadata changed during automatic incomplete-"
            "profile cleanup; remaining actor IDs are {}".format(
                ", ".join(str(actor_id) for actor_id, _role in survivors)
            )
        )
    LOG.info(
        "Removed %d stable incomplete network-profile zone metadata actor(s) "
        "without requiring --replace-existing-network-profile",
        len(candidate_conflicts),
    )
    return True


def discover_network_profile_after_passive_registration(
    world: carla.World,
    session_token: str,
    attempts: int = NETWORK_PROFILE_REGISTRY_RETRY_ATTEMPTS,
):
    """Discover this session after bounded, externally driven registry updates."""
    attempts = max(1, int(attempts))
    last_observation = "no profile was visible"
    for attempt in range(attempts):
        try:
            discovered = discover_network_degradation_profile(world, strict=True)
        except NetworkProfileError as exc:
            discovered = None
            last_observation = str(exc)
        if discovered is not None:
            if discovered.session_token == session_token:
                return discovered
            last_observation = (
                "a different committed profile token {} was visible".format(
                    discovered.session_token
                )
            )
        if attempt + 1 >= attempts:
            break
        passively_wait_for_network_profile_snapshot(
            world,
            "new manifest registration {}/{}".format(
                attempt + 1,
                attempts - 1,
            ),
        )
    raise RuntimeError(
        "profile token {} was not committed after {} registry observation(s); "
        "last observation: {}".format(
            session_token,
            attempts,
            last_observation,
        )
    )


def reuse_compatible_network_profile(
    world: carla.World,
    zones: Sequence[NetworkDegradationZone],
    start_active_sensors: bool,
) -> Optional[PublishedNetworkProfileActors]:
    """Borrow an identical committed profile without assuming ownership."""
    try:
        discovered = discover_network_degradation_profile(world, strict=True)
    except NetworkProfileError as exc:
        raise RuntimeError(
            "pre-existing network-profile metadata is incomplete or "
            "ambiguous: {}".format(exc)
        ) from exc
    if discovered is None:
        return None
    if not network_profile_matches_requested_configuration(
        discovered,
        zones,
        start_active_sensors,
    ):
        return None

    manifest_actor = resolve_network_profile_actor(
        world,
        discovered.manifest_actor_id,
    )
    zone_actors = tuple(
        resolve_network_profile_actor(world, actor_id)
        for actor_id in discovered.zone_actor_ids
    )
    publication = PublishedNetworkProfileActors(
        session_token=discovered.session_token,
        zones=tuple(discovered.zones),
        zone_actors=zone_actors,
        manifest_actor=manifest_actor,
        start_active_sensors=bool(discovered.start_active_sensors),
        owns_actors=False,
    )
    LOG.info(
        "Reusing compatible pre-existing CARLA-world network profile "
        "token=%s manifest=%d zone_actor_ids=%s active_sensor_streams=%s; "
        "metadata actors remain unlistened and are not owned by this process",
        publication.session_token,
        publication.manifest_actor.id,
        (
            ",".join(str(actor.id) for actor in publication.zone_actors)
            if publication.zone_actors
            else "none"
        ),
        "permitted for displayed-active IDs"
        if publication.start_active_sensors
        else "off (default)",
    )
    return publication


def replace_existing_network_profile_metadata(
    world: carla.World,
    conflicts: Sequence[Tuple[int, str]],
) -> None:
    """Explicitly remove only reserved profile actors, manifest first."""
    identities = [
        (int(actor_id), str(role_name))
        for actor_id, role_name in conflicts
    ]
    manifests = [
        record
        for record in identities
        if record[1].startswith(NETWORK_PROFILE_MANIFEST_PREFIX)
    ]
    remaining = [
        record
        for record in identities
        if not record[1].startswith(NETWORK_PROFILE_MANIFEST_PREFIX)
    ]

    failed_manifest_ids = []
    for actor_id, role_name in manifests:
        if not destroy_registered_network_profile_actor(
            world,
            actor_id,
            role_name,
            "stale network-profile manifest metadata actor",
        ):
            failed_manifest_ids.append(actor_id)
    if failed_manifest_ids:
        raise RuntimeError(
            "unable to remove pre-existing network-profile manifest actor(s) "
            "{}; preserving zone metadata to avoid leaving a committed "
            "partial profile".format(
                ", ".join(str(actor_id) for actor_id in failed_manifest_ids)
            )
        )

    failed_actor_ids = []
    for actor_id, role_name in reversed(remaining):
        if not destroy_registered_network_profile_actor(
            world,
            actor_id,
            role_name,
            "stale network-profile metadata actor",
        ):
            failed_actor_ids.append(actor_id)
    if failed_actor_ids:
        raise RuntimeError(
            "unable to remove pre-existing network-profile metadata actor(s) "
            "{}".format(
                ", ".join(str(actor_id) for actor_id in failed_actor_ids)
            )
        )

    try:
        survivors = find_profile_actor_conflicts(world)
    except NetworkProfileError as exc:
        raise RuntimeError(str(exc)) from exc
    if survivors:
        raise RuntimeError(
            "network-profile metadata changed during replacement; remaining "
            "actor IDs are {}".format(
                ", ".join(str(actor_id) for actor_id, _role in survivors)
            )
        )
    LOG.info(
        "Removed %d explicitly replaceable pre-existing network-profile "
        "metadata actor(s)",
        len(identities),
    )


def destroy_published_network_profile(
    publication: Optional[PublishedNetworkProfileActors],
    world: Optional[carla.World] = None,
) -> None:
    """Invalidate the profile first, then remove its unlistened zone actors."""
    if publication is None:
        return
    if not publication.owns_actors:
        LOG.info(
            "Leaving reused network profile token=%s intact because this "
            "process does not own its metadata actors",
            publication.session_token,
        )
        return
    if world is None:
        LOG.error(
            "Cannot safely remove owned network profile token=%s without the "
            "current CARLA world registry; preserving its metadata actors",
            publication.session_token,
        )
        return
    # The manifest is the commit marker. Removing it first makes any remaining
    # zone actor set invalid to strict readers during passive teardown.
    manifest_role = build_manifest_role(
        publication.session_token,
        len(publication.zones),
        publication.start_active_sensors,
    )
    manifest_removed = destroy_registered_network_profile_actor(
        world,
        publication.manifest_actor.id,
        manifest_role,
        "network-profile manifest metadata actor",
    )
    if not manifest_removed:
        # Keep the still-committed profile complete. Deleting its zone actors
        # after a failed manifest deletion would leave a manifest that claims a
        # zone count CARLA no longer contains. A later launch will report the
        # exact retained actor IDs for explicit recovery.
        LOG.error(
            "Network-profile manifest id=%s could not be removed; preserving "
            "%d zone metadata actor(s) to avoid a partial committed profile",
            getattr(publication.manifest_actor, "id", "unknown"),
            len(publication.zone_actors),
        )
        return
    for zone, actor in reversed(
        tuple(zip(publication.zones, publication.zone_actors))
    ):
        destroy_registered_network_profile_actor(
            world,
            actor.id,
            build_zone_role(
                publication.session_token,
                zone.index,
                zone.radius,
            ),
            "network-profile zone metadata actor",
        )


def rollback_network_profile_session(
    world: carla.World,
    session_token: str,
    manifest_role: str,
    zone_roles: Sequence[str],
    manifest_actor,
    zone_actors: Sequence[object],
) -> bool:
    """Invalidate and remove one failed publication using registry identities."""
    expected_roles = {str(manifest_role)} | {str(role) for role in zone_roles}
    known_by_role = {}
    if manifest_actor is not None:
        known_by_role.setdefault(str(manifest_role), set()).add(
            int(manifest_actor.id)
        )
    for role_name, actor in zip(zone_roles, zone_actors):
        known_by_role.setdefault(str(role_name), set()).add(int(actor.id))

    # A spawn RPC can complete server-side before its returned handle becomes
    # discoverable, or can raise after the server accepted it. Observe later
    # snapshots and collect the exact random-token roles from the live registry.
    for attempt in range(NETWORK_PROFILE_REGISTRY_RETRY_ATTEMPTS):
        passively_wait_for_network_profile_snapshot(
            world,
            "failed profile token={} registry reconciliation {}/{}".format(
                session_token,
                attempt + 1,
                NETWORK_PROFILE_REGISTRY_RETRY_ATTEMPTS,
            ),
        )
        try:
            conflicts = find_profile_actor_conflicts(world)
        except NetworkProfileError as exc:
            LOG.warning(
                "Unable to enumerate failed network-profile token=%s during "
                "rollback: %s",
                session_token,
                exc,
            )
            continue
        for actor_id, role_name in conflicts:
            if role_name in expected_roles:
                known_by_role.setdefault(role_name, set()).add(int(actor_id))
        if expected_roles.issubset(set(known_by_role)):
            break

    manifest_ids = sorted(known_by_role.get(str(manifest_role), ()))
    failed_manifest_ids = []
    for actor_id in manifest_ids:
        if not destroy_registered_network_profile_actor(
            world,
            actor_id,
            manifest_role,
            "partial network-profile manifest metadata actor",
        ):
            failed_manifest_ids.append(actor_id)
    if failed_manifest_ids:
        LOG.error(
            "Partial network-profile manifest actor(s) %s could not be "
            "confirmed absent; preserving zone metadata for token=%s to "
            "avoid corrupting a possibly committed profile",
            ", ".join(str(actor_id) for actor_id in failed_manifest_ids),
            session_token,
        )
        return False

    failed_zone_ids = []
    for role_name in reversed(tuple(zone_roles)):
        for actor_id in sorted(known_by_role.get(str(role_name), ()), reverse=True):
            if not destroy_registered_network_profile_actor(
                world,
                actor_id,
                role_name,
                "partial network-profile zone metadata actor",
            ):
                failed_zone_ids.append(actor_id)

    try:
        survivors = tuple(
            (actor_id, role_name)
            for actor_id, role_name in find_profile_actor_conflicts(world)
            if role_name in expected_roles
        )
    except NetworkProfileError as exc:
        LOG.warning(
            "Unable to confirm rollback of network-profile token=%s: %s",
            session_token,
            exc,
        )
        return False
    if failed_zone_ids or survivors:
        LOG.warning(
            "Rollback of uncommitted network-profile token=%s left zone "
            "actor IDs %s; a later normal launch can automatically recover "
            "an unchanged matching zone-only residue",
            session_token,
            ", ".join(
                str(actor_id)
                for actor_id in sorted(
                    set(failed_zone_ids)
                    | {actor_id for actor_id, _role_name in survivors}
                )
            ),
        )
        return False
    return True


def publish_network_degradation_profile(
    world: carla.World,
    zones: Sequence[NetworkDegradationZone],
    start_active_sensors: bool,
    replace_existing: bool = False,
) -> PublishedNetworkProfileActors:
    """Transactionally publish one complete shared profile into CARLA."""
    normalized_zones = []
    seen_indices = set()
    for configured_zone in zones:
        zone = normalize_zone(
            configured_zone.index,
            configured_zone.x,
            configured_zone.y,
            configured_zone.radius,
        )
        if zone.index in seen_indices:
            raise RuntimeError(
                "duplicate configured network-degradation zone index {}".format(
                    zone.index
                )
            )
        seen_indices.add(zone.index)
        normalized_zones.append(zone)
    normalized_zones.sort(key=lambda item: item.index)

    try:
        conflicts = find_profile_actor_conflicts(world)
    except NetworkProfileError as exc:
        raise RuntimeError(str(exc)) from exc
    if conflicts and not replace_existing:
        try:
            removed_incomplete_profile = (
                remove_stable_matching_zone_only_residue(
                    world,
                    conflicts,
                    normalized_zones,
                )
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "unable to recover incomplete network-profile metadata: "
                "{}".format(exc)
            ) from exc
        if removed_incomplete_profile:
            try:
                conflicts = find_profile_actor_conflicts(world)
            except NetworkProfileError as exc:
                raise RuntimeError(str(exc)) from exc
    if conflicts:
        conflict_text = ", ".join(
            "actor={} role={!r}".format(actor_id, role_name)
            for actor_id, role_name in conflicts[:6]
        )
        if len(conflicts) > 6:
            conflict_text += ", ..."
        if replace_existing:
            LOG.warning(
                "Explicitly replacing %d pre-existing spawn_blocker "
                "network-profile metadata actor(s): %s",
                len(conflicts),
                conflict_text,
            )
            replace_existing_network_profile_metadata(world, conflicts)
        else:
            try:
                reused = reuse_compatible_network_profile(
                    world,
                    normalized_zones,
                    start_active_sensors,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "found {} pre-existing spawn_blocker network-profile "
                    "metadata actor(s), but they cannot be safely reused: "
                    "{}. Another publisher may still own them. If they are "
                    "known to be stale, retry with "
                    "--replace-existing-network-profile".format(
                        len(conflicts),
                        exc,
                    )
                ) from exc
            if reused is not None:
                return reused
            try:
                remaining_conflicts = find_profile_actor_conflicts(world)
            except NetworkProfileError as exc:
                raise RuntimeError(str(exc)) from exc
            if remaining_conflicts:
                raise RuntimeError(
                    "found {} incompatible pre-existing spawn_blocker "
                    "network-profile metadata actor(s): {}. Another publisher "
                    "may still own them. If they are known to be stale, retry "
                    "with --replace-existing-network-profile".format(
                        len(conflicts),
                        conflict_text,
                    )
                )
            LOG.info(
                "Pre-existing network-profile metadata disappeared during "
                "startup; publishing the requested profile"
            )

    session_token = new_session_token()
    zone_actors = []
    manifest_actor = None
    zone_roles = tuple(
        build_zone_role(session_token, zone.index, zone.radius)
        for zone in normalized_zones
    )
    manifest_role = build_manifest_role(
        session_token,
        len(normalized_zones),
        start_active_sensors,
    )
    try:
        # Zone actors are deliberately created before the manifest. Strict
        # readers will reject/ignore this incomplete state until the final
        # manifest actor commits the expected count and session token.
        for zone, zone_role in zip(normalized_zones, zone_roles):
            zone_actor = spawn_network_profile_metadata_actor(
                world,
                zone_role,
                zone.x,
                zone.y,
            )
            zone_actors.append(zone_actor)
        manifest_actor = spawn_network_profile_metadata_actor(
            world,
            manifest_role,
            0.0,
            0.0,
        )

        discovered = discover_network_profile_after_passive_registration(
            world,
            session_token,
        )
        if discovered.manifest_actor_id != int(manifest_actor.id):
            raise RuntimeError(
                "published network-profile manifest actor mismatch: "
                "expected {}, found {}".format(
                    manifest_actor.id,
                    discovered.manifest_actor_id,
                )
            )
        expected_zone_actor_ids = tuple(int(actor.id) for actor in zone_actors)
        if tuple(sorted(discovered.zone_actor_ids)) != tuple(
            sorted(expected_zone_actor_ids)
        ):
            raise RuntimeError(
                "published network-profile zone actor mismatch: expected {}, "
                "found {}".format(
                    expected_zone_actor_ids,
                    tuple(sorted(discovered.zone_actor_ids)),
                )
            )
        if discovered.start_active_sensors != bool(start_active_sensors):
            raise RuntimeError(
                "published network-profile sensor-stream flag mismatch"
            )
        if len(discovered.zones) != len(normalized_zones):
            raise RuntimeError(
                "published network-profile zone count mismatch"
            )
        for expected, actual in zip(normalized_zones, discovered.zones):
            if (
                expected.index != actual.index
                or abs(expected.x - actual.x) > 1.0e-3
                or abs(expected.y - actual.y) > 1.0e-3
                or abs(expected.radius - actual.radius) > 1.0e-6
            ):
                raise RuntimeError(
                    "published network-profile zone verification mismatch"
                )

        # Retain current registry handles rather than spawn-return proxies.
        # This also validates the exact expected roles and non-listening state.
        manifest_actor = registered_network_profile_actor(
            world,
            discovered.manifest_actor_id,
            manifest_role,
        )
        if manifest_actor is None:
            raise RuntimeError(
                "published network-profile manifest disappeared after "
                "verification"
            )
        resolved_zone_actors = []
        for actor_id, zone_role in zip(expected_zone_actor_ids, zone_roles):
            actor = registered_network_profile_actor(
                world,
                actor_id,
                zone_role,
            )
            if actor is None:
                raise RuntimeError(
                    "published network-profile zone actor {} disappeared "
                    "after verification".format(actor_id)
                )
            resolved_zone_actors.append(actor)
        zone_actors = resolved_zone_actors
    except BaseException as exc:
        # Re-resolve the random-token roles from the current registry. The
        # manifest is always confirmed absent before any zone is removed.
        rollback_network_profile_session(
            world,
            session_token,
            manifest_role,
            zone_roles,
            manifest_actor,
            zone_actors,
        )
        if isinstance(
            exc,
            (NetworkProfileError, RuntimeError, TypeError, ValueError),
        ):
            raise RuntimeError(
                "unable to publish shared network-degradation profile: {}".format(
                    exc
                )
            ) from exc
        # KeyboardInterrupt/SystemExit must still abort, but only after rolling
        # back any zone actors created before the manifest commit point.
        raise

    publication = PublishedNetworkProfileActors(
        session_token=session_token,
        zones=tuple(normalized_zones),
        zone_actors=tuple(zone_actors),
        manifest_actor=manifest_actor,
        start_active_sensors=bool(start_active_sensors),
        owns_actors=True,
    )
    LOG.info(
        "Published CARLA-world network profile token=%s manifest=%d zones=%s "
        "active_sensor_streams=%s; metadata GNSS actors are unlistened",
        publication.session_token,
        publication.manifest_actor.id,
        (
            ", ".join(
                "{}=({:.3f}, {:.3f}, r={:.1f} m) actor={}".format(
                    zone.label,
                    zone.x,
                    zone.y,
                    zone.radius,
                    actor.id,
                )
                for zone, actor in zip(
                    publication.zones,
                    publication.zone_actors,
                )
            )
            if publication.zones
            else "disabled"
        ),
        "permitted for displayed-active IDs"
        if publication.start_active_sensors
        else "off (default)",
    )
    return publication


def reset_activation_fields(state: PedestrianState) -> None:
    state.active_since = None
    state.active_ego_id = None
    state.last_separation = None
    state.crossing_origin = None
    state.crossing_direction = None
    state.crossing_endpoint = None
    state.crossing_distance = None
    state.ego_path_direction = None
    state.front_contact_offset = 0.0
    state.commanded_pedestrian_speed = None
    state.filtered_ego_acceleration = None
    state.last_ego_intercept_signed_distance = None
    state.minimum_ego_surface_gap = None
    state.hard_brake_seen = False
    state.intercept_reached = False
    state.pending_near_miss_reason = None
    state.last_progress = None
    state.last_progress_time = None
    state.motion_command_failures = 0
    state.stall_recovery_count = 0
    state.scripted_recovery_active = False
    state.scripted_recovery_started_at = None
    state.last_motion_update_time = None
    state.last_debug_draw_time = None
    state.hold_until = None
    state.hold_reason = None


def retire_state_actors(
    state: PedestrianState,
    registry: CollisionRegistry,
) -> None:
    walker = state.actor
    walker_id = getattr(walker, "id", None)
    if walker_id is not None:
        registry.unregister(int(walker_id))

    sensor = state.sensor
    sensor_id = getattr(sensor, "id", None)
    if sensor is not None:
        try:
            if sensor.is_alive:
                sensor.stop()
        except (AttributeError, RuntimeError):
            pass
        destroy_actor(sensor, "collision sensor")
    state.sensor = None

    stop_walker(walker)
    destroy_actor(walker, "pedestrian blocker")
    state.actor = None
    state.retired_sensor_id = (
        None if sensor_id is None else int(sensor_id)
    )
    state.retired_actor_id = (
        None if walker_id is None else int(walker_id)
    )


def begin_post_event_hold(
    state: PedestrianState,
    simulation_time: float,
    reason: str,
    args: argparse.Namespace,
) -> None:
    """Keep the pedestrian visible and stationary before it is recycled."""
    if state.state == STATE_RESPAWN_PENDING:
        return
    if state.state == STATE_HOLDING:
        if reason.startswith("vehicle contact") and not str(
            state.hold_reason or ""
        ).startswith("vehicle contact"):
            state.hold_reason = reason
            state.hold_until = (
                float(simulation_time) + float(args.post_event_hold)
            )
            LOG.info(
                "Pedestrian #%d id=%s HOLDING outcome upgraded to %s",
                state.index,
                getattr(state.actor, "id", None),
                reason,
            )
        stop_walker(state.actor)
        return
    state.state = STATE_HOLDING
    state.hold_reason = reason
    state.hold_until = float(simulation_time) + float(args.post_event_hold)
    stop_walker(state.actor)
    try:
        location = state.actor.get_location()
        location_text = "({:.2f}, {:.2f})".format(location.x, location.y)
    except (AttributeError, RuntimeError):
        location_text = "unavailable"
    LOG.info(
        "Pedestrian #%d generation=%d id=%s state=HOLDING location=%s "
        "for %.2f s after %s",
        state.index,
        state.generation,
        getattr(state.actor, "id", None),
        location_text,
        args.post_event_hold,
        reason,
    )


def update_holding_pedestrian(
    state: PedestrianState,
    simulation_time: float,
) -> Optional[str]:
    if state.state != STATE_HOLDING:
        return None
    if not state_actor_alive(state):
        return "held pedestrian actor was removed outside this client"
    stop_walker(state.actor)
    if state.hold_until is None or simulation_time + 1.0e-9 >= state.hold_until:
        return "post-event hold completed after {}".format(
            state.hold_reason or "collision/near miss"
        )
    return None


def request_respawn(
    state: PedestrianState,
    simulation_time: float,
    reason: str,
    registry: CollisionRegistry,
    args: argparse.Namespace,
    delay_override: Optional[float] = None,
) -> None:
    if state.state == STATE_RESPAWN_PENDING:
        return
    old_actor_id = getattr(state.actor, "id", None)
    LOG.info(
        "Pedestrian #%d generation=%d id=%s reset requested: %s",
        state.index,
        state.generation,
        old_actor_id,
        reason,
    )
    retire_state_actors(state, registry)
    reset_activation_fields(state)
    state.state = STATE_RESPAWN_PENDING
    delay = args.respawn_delay if delay_override is None else delay_override
    state.respawn_due = float(simulation_time) + max(0.0, float(delay))


def retired_actor_is_absent(
    world: carla.World,
    actor_id: Optional[int],
    actor_kind: str,
) -> bool:
    if actor_id is None:
        return True
    try:
        actor = world.get_actor(int(actor_id))
    except RuntimeError:
        return False
    if actor is None:
        return True
    try:
        if not actor.is_alive:
            return True
        # A wrapper rediscovered by ID has no live subscription owned through
        # that proxy. Calling stop() can therefore emit a misleading CARLA
        # warning; direct destruction is correct for both collision and
        # unsubscribed front-sensor cleanup confirmation.
        destroy_actor(actor, actor_kind)
    except (AttributeError, RuntimeError):
        pass
    # Confirm removal from a later passive snapshot before reusing the home
    # transform; destroy() completion can lag the client proxy by one tick.
    return False


def attempt_pending_respawn(
    state: PedestrianState,
    world: carla.World,
    carla_map: carla.Map,
    navigation: NavigationSampler,
    simulation_time: float,
    registry: CollisionRegistry,
    args: argparse.Namespace,
) -> bool:
    if state.state != STATE_RESPAWN_PENDING:
        return False
    if state.respawn_due is not None and simulation_time + 1.0e-9 < state.respawn_due:
        return False

    sensor_absent = retired_actor_is_absent(
        world,
        state.retired_sensor_id,
        "collision sensor",
    )
    walker_absent = retired_actor_is_absent(
        world,
        state.retired_actor_id,
        "pedestrian blocker",
    )
    if not sensor_absent or not walker_absent:
        state.respawn_due = simulation_time + args.respawn_retry_interval
        return False
    state.retired_sensor_id = None
    state.retired_actor_id = None

    desired_location = target_transform(state.target, args.z_offset).location
    try:
        occupying_vehicle_id = vehicle_near_location(
            world,
            desired_location,
            args.respawn_clearance,
        )
    except RuntimeError as exc:
        state.respawn_due = simulation_time + args.respawn_retry_interval
        LOG.warning(
            "Pedestrian #%d deferring respawn for %.2f s: %s",
            state.index,
            args.respawn_retry_interval,
            exc,
        )
        return False
    if occupying_vehicle_id is not None:
        state.respawn_due = simulation_time + args.respawn_retry_interval
        LOG.info(
            "Pedestrian #%d waiting to respawn: vehicle id=%d footprint is "
            "within %.2f m XY of the original target",
            state.index,
            occupying_vehicle_id,
            args.respawn_clearance,
        )
        return False
    try:
        occupying_actor = blocking_actor_at_location(
            world,
            desired_location,
            DEFAULT_SPAWN_OCCUPANCY_CLEARANCE_M,
        )
    except RuntimeError as exc:
        state.respawn_due = simulation_time + args.respawn_retry_interval
        LOG.warning(
            "Pedestrian #%d deferring respawn for %.2f s: %s",
            state.index,
            args.respawn_retry_interval,
            exc,
        )
        return False
    if occupying_actor is not None:
        state.respawn_due = simulation_time + args.respawn_retry_interval
        LOG.info(
            "Pedestrian #%d waiting to respawn: actor id=%d type=%s occupies "
            "the original target",
            state.index,
            occupying_actor.id,
            occupying_actor.type_id,
        )
        return False

    try:
        walker = spawn_pedestrian(
            world,
            carla_map,
            state.target,
            state.index,
            args,
            navigation,
        )
    except (RuntimeError, ValueError) as exc:
        state.respawn_due = simulation_time + args.respawn_retry_interval
        LOG.warning(
            "Pedestrian #%d respawn at the original transform failed; "
            "retrying in %.2f s: %s",
            state.index,
            args.respawn_retry_interval,
            exc,
        )
        return False

    state.actor = walker
    state.sensor = spawn_collision_sensor(world, walker, registry)
    reset_activation_fields(state)
    state.generation += 1
    state.state = STATE_WAITING
    state.respawn_due = None
    LOG.info(
        "Pedestrian #%d generation=%d id=%d respawned WAITING at "
        "original target=(%.3f, %.3f, %.3f)",
        state.index,
        state.generation,
        walker.id,
        state.target.x,
        state.target.y,
        state.target.z + args.z_offset,
    )
    return True


def reactive_vehicle_actor_alive(state: ReactiveVehicleState) -> bool:
    try:
        return state.actor is not None and bool(state.actor.is_alive)
    except (AttributeError, RuntimeError):
        return False


def stop_reactive_vehicle(vehicle) -> None:
    """Best-effort stop that also clears scripted constant velocity."""
    if vehicle is None:
        return
    try:
        vehicle.disable_constant_velocity()
    except (AttributeError, RuntimeError):
        pass
    try:
        vehicle.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
    except (AttributeError, RuntimeError):
        pass
    try:
        vehicle.set_target_angular_velocity(
            carla.Vector3D(x=0.0, y=0.0, z=0.0)
        )
    except (AttributeError, RuntimeError):
        pass
    try:
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        control.hand_brake = True
        control.reverse = False
        vehicle.apply_control(control)
    except (AttributeError, RuntimeError):
        pass


def release_reactive_vehicle_brake(vehicle) -> bool:
    try:
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0
        control.hand_brake = False
        control.reverse = False
        vehicle.apply_control(control)
        return True
    except (AttributeError, RuntimeError):
        return False


def reactive_vehicle_planar_speed(vehicle) -> float:
    if vehicle is None:
        return 0.0
    try:
        velocity = vehicle.get_velocity()
        return math.hypot(float(velocity.x), float(velocity.y))
    except (AttributeError, RuntimeError):
        return 0.0


def coast_reactive_vehicle_after_contact(vehicle) -> bool:
    """Release scripted drive without erasing the physical impact impulse."""
    if vehicle is None:
        return False
    try:
        vehicle.disable_constant_velocity()
    except (AttributeError, RuntimeError):
        pass
    try:
        vehicle.set_simulate_physics(True)
        vehicle.set_collisions(True)
    except (AttributeError, RuntimeError):
        pass
    try:
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0
        control.hand_brake = False
        control.reverse = False
        vehicle.apply_control(control)
        return True
    except (AttributeError, RuntimeError):
        return False


def apply_reactive_vehicle_service_brake(
    vehicle,
    engage_parking_brake: bool = False,
) -> bool:
    """Brake through CARLA physics without setting velocity to zero."""
    if vehicle is None:
        return False
    try:
        vehicle.disable_constant_velocity()
    except (AttributeError, RuntimeError):
        pass
    try:
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        control.hand_brake = bool(engage_parking_brake)
        control.reverse = False
        vehicle.apply_control(control)
        return True
    except (AttributeError, RuntimeError):
        return False


def freeze_reactive_vehicle_at_home(
    state: ReactiveVehicleState,
    args: argparse.Namespace,
) -> bool:
    if not reactive_vehicle_actor_alive(state):
        return False
    stop_reactive_vehicle(state.actor)
    try:
        state.actor.set_simulate_physics(False)
        state.actor.set_transform(
            vehicle_target_transform(state.target, args.vehicle_z_offset)
        )
        state.route_segment_index = 0
        state.route_progress = 0.0
        state.route_projection_gap = 0.0
        return True
    except (AttributeError, RuntimeError):
        return False


def reactive_vehicle_motion_geometry(
    state: ReactiveVehicleState,
) -> Optional[Tuple[float, float, float, float]]:
    """Return lane-route center/front station, cross-track, and speed."""
    if not reactive_vehicle_actor_alive(state):
        return None
    try:
        transform = state.actor.get_transform()
        velocity = state.actor.get_velocity()
    except (AttributeError, RuntimeError):
        return None
    segment_count = len(state.route_locations) - 1
    if segment_count <= 0:
        return None
    search_start = max(0, int(state.route_segment_index) - 3)
    search_end = max(search_start, int(state.route_segment_index))
    forward_station_limit = float(state.route_progress) + 20.0
    while (
        search_end < segment_count - 1
        and float(state.route_distances[search_end + 1])
        <= forward_station_limit
    ):
        search_end += 1
    projection = project_xy_onto_route(
        state.route_locations,
        state.route_distances,
        float(transform.location.x),
        float(transform.location.y),
        search_start,
        search_end,
    )
    if projection is None:
        return None
    center_progress = max(float(state.route_progress), projection.station)
    pose = route_pose_at_station(
        state.route_locations,
        state.route_distances,
        center_progress,
    )
    if pose is None:
        return None
    _route_location, tangent, route_segment_index = pose
    state.route_progress = center_progress
    state.route_segment_index = max(
        int(state.route_segment_index),
        int(route_segment_index),
    )
    state.route_projection_gap = projection.gap
    cross_track = projection.signed_cross_track
    forward_speed = max(
        0.0,
        float(velocity.x) * tangent[0]
        + float(velocity.y) * tangent[1],
    )
    current_support = actor_leading_support(
        state.actor,
        tangent,
    )
    if current_support is not None:
        state.front_support = current_support
    front_progress = center_progress + max(0.0, state.front_support)
    return center_progress, front_progress, cross_track, forward_speed


def normalize_angle_degrees(angle_degrees: float) -> float:
    return (float(angle_degrees) + 180.0) % 360.0 - 180.0


def reactive_vehicle_steer(
    state: ReactiveVehicleState,
    args: argparse.Namespace,
) -> float:
    try:
        transform = state.actor.get_transform()
    except (AttributeError, RuntimeError):
        return 0.0
    try:
        velocity = state.actor.get_velocity()
        current_speed = math.hypot(float(velocity.x), float(velocity.y))
    except (AttributeError, RuntimeError):
        current_speed = 0.0
    lookahead = float(args.reactive_route_lookahead) + min(
        REACTIVE_ROUTE_DYNAMIC_LOOKAHEAD_MAX_M,
        0.25 * current_speed,
    )
    target_pose = route_pose_at_station(
        state.route_locations,
        state.route_distances,
        state.route_progress + lookahead,
    )
    if target_pose is None:
        return 0.0
    target_location, _target_tangent, _target_segment = target_pose
    delta_x = float(target_location.x - transform.location.x)
    delta_y = float(target_location.y - transform.location.y)
    target_distance = math.hypot(delta_x, delta_y)
    if target_distance <= 1.0e-6:
        return 0.0
    target_yaw = math.degrees(math.atan2(delta_y, delta_x))
    heading_error = math.radians(
        normalize_angle_degrees(target_yaw - float(transform.rotation.yaw))
    )
    try:
        half_length = float(state.actor.bounding_box.extent.x)
    except (AttributeError, RuntimeError):
        half_length = 2.5
    wheelbase = max(2.5, min(8.0, 1.10 * half_length))
    steer = math.atan2(
        2.0 * wheelbase * math.sin(heading_error),
        max(1.0, target_distance),
    )
    limit = float(args.reactive_route_steer_limit)
    return max(-limit, min(limit, steer))


def apply_reactive_vehicle_command(
    state: ReactiveVehicleState,
    speed: float,
    acceleration: float,
    args: argparse.Namespace,
) -> bool:
    if not reactive_vehicle_actor_alive(state):
        return False
    speed_value = max(
        0.0,
        min(
            float(args.reactive_vehicle_speed_max),
            float(state.route_speed_limit),
            float(speed),
        ),
    )
    steer = reactive_vehicle_steer(state, args)
    try:
        if args.reactive_vehicle_control == "constant-velocity":
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            limit = float(args.reactive_route_steer_limit)
            control.steer = max(-1.0, min(1.0, steer / limit))
            state.actor.apply_control(control)
            state.actor.enable_constant_velocity(
                carla.Vector3D(x=speed_value, y=0.0, z=0.0)
            )
        else:
            state.actor.disable_constant_velocity()
            control = carla.VehicleAckermannControl()
            control.steer = float(steer)
            control.steer_speed = 0.0
            control.speed = speed_value
            control.acceleration = float(acceleration)
            control.jerk = 0.0
            state.actor.apply_ackermann_control(control)
        state.commanded_speed = speed_value
        return True
    except (AttributeError, RuntimeError):
        return False


def reactive_vehicle_contact_fallback(
    state: ReactiveVehicleState,
    pedestrian,
) -> Optional[str]:
    if pedestrian is None or not reactive_vehicle_actor_alive(state):
        return None
    try:
        if int(pedestrian.id) != int(state.active_target_id):
            return None
    except (AttributeError, TypeError):
        return None
    vehicle_geometry = actor_box_geometry(state.actor)
    pedestrian_geometry = actor_box_geometry(pedestrian)
    if vehicle_geometry is not None and pedestrian_geometry is not None:
        vehicle_footprint, vehicle_min_z, vehicle_max_z = vehicle_geometry
        pedestrian_footprint, pedestrian_min_z, pedestrian_max_z = (
            pedestrian_geometry
        )
        exact_overlap = vertical_intervals_overlap(
            vehicle_min_z,
            vehicle_max_z,
            pedestrian_min_z,
            pedestrian_max_z,
            DEFAULT_COLLISION_VERTICAL_CLEARANCE_M,
        ) and convex_footprints_overlap(
            vehicle_footprint,
            pedestrian_footprint,
        )
        if exact_overlap:
            return "oriented bounding-box overlap"
        # A valid separating-axis result is authoritative. Circumscribed radii
        # are much too conservative for a long bus and would report contacts
        # several metres outside its actual footprint.
        return None
    if state.sensor is not None:
        return None
    try:
        vehicle_location = state.actor.get_location()
        pedestrian_location = pedestrian.get_location()
        if (
            abs(float(vehicle_location.z - pedestrian_location.z))
            > DEFAULT_COLLISION_CENTER_VERTICAL_TOLERANCE_M
        ):
            return None
        vehicle_radius = actor_planar_bounding_radius(state.actor)
        pedestrian_radius = actor_planar_bounding_radius(pedestrian)
        if vehicle_radius is None or pedestrian_radius is None:
            return None
        if planar_distance(vehicle_location, pedestrian_location) <= (
            vehicle_radius + pedestrian_radius
        ):
            return "bounding-radius fallback"
    except (AttributeError, RuntimeError):
        return None
    return None


def segment_intersects_axis_aligned_box(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    minimum_x: float,
    maximum_x: float,
    minimum_y: float,
    maximum_y: float,
) -> bool:
    """Liang-Barsky style intersection for one finite XY segment."""
    lower = 0.0
    upper = 1.0
    for start_value, delta, bound_min, bound_max in (
        (start_x, end_x - start_x, minimum_x, maximum_x),
        (start_y, end_y - start_y, minimum_y, maximum_y),
    ):
        if abs(delta) <= 1.0e-12:
            if start_value < bound_min or start_value > bound_max:
                return False
            continue
        first = (bound_min - start_value) / delta
        second = (bound_max - start_value) / delta
        if first > second:
            first, second = second, first
        lower = max(lower, first)
        upper = min(upper, second)
        if lower > upper:
            return False
    return True


def update_reactive_contact_pose_history(
    state: ReactiveVehicleState,
    pedestrian,
    simulation_time: float,
) -> Optional[
    Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
        float,
    ]
]:
    """Store actor poses and return the previous/current pair when usable."""
    try:
        vehicle_location = state.actor.get_location()
        pedestrian_location = pedestrian.get_location()
    except (AttributeError, RuntimeError):
        return None
    current_vehicle = (
        float(vehicle_location.x),
        float(vehicle_location.y),
        float(vehicle_location.z),
    )
    current_pedestrian = (
        float(pedestrian_location.x),
        float(pedestrian_location.y),
        float(pedestrian_location.z),
    )
    previous_time = state.previous_contact_sample_time
    previous_vehicle = state.previous_vehicle_location
    previous_pedestrian = state.previous_pedestrian_location
    state.previous_contact_sample_time = float(simulation_time)
    state.previous_vehicle_location = current_vehicle
    state.previous_pedestrian_location = current_pedestrian
    if (
        previous_time is None
        or previous_vehicle is None
        or previous_pedestrian is None
    ):
        return None
    elapsed = float(simulation_time) - float(previous_time)
    if elapsed <= 0.0 or not math.isfinite(elapsed):
        return None
    return (
        previous_vehicle,
        previous_pedestrian,
        current_vehicle,
        current_pedestrian,
        elapsed,
    )


def reactive_vehicle_swept_contact_fallback(
    state: ReactiveVehicleState,
    pedestrian,
    simulation_time: float,
    maximum_interval: float = DEFAULT_REACTIVE_CONTACT_SWEEP_MAX_INTERVAL_SECONDS,
) -> Optional[str]:
    """Detect a fast walker sweeping through the vehicle between updates.

    This does not synthesize an impulse.  It only prevents actor teardown while
    CARLA's asynchronous collision event/physics response catches up.
    """
    sample = update_reactive_contact_pose_history(
        state,
        pedestrian,
        simulation_time,
    )
    if sample is None or sample[4] > float(maximum_interval):
        return None
    (
        previous_vehicle,
        previous_pedestrian,
        current_vehicle,
        current_pedestrian,
        _elapsed,
    ) = sample
    vehicle_geometry = actor_box_geometry(state.actor)
    pedestrian_geometry = actor_box_geometry(pedestrian)
    if vehicle_geometry is None or pedestrian_geometry is None:
        return None
    vehicle_footprint, vehicle_min_z, vehicle_max_z = vehicle_geometry
    _pedestrian_footprint, pedestrian_min_z, pedestrian_max_z = (
        pedestrian_geometry
    )
    if not vertical_intervals_overlap(
        vehicle_min_z,
        vehicle_max_z,
        pedestrian_min_z,
        pedestrian_max_z,
        DEFAULT_COLLISION_VERTICAL_CLEARANCE_M,
    ):
        return None
    pose = route_pose_at_station(
        state.route_locations,
        state.route_distances,
        state.route_progress,
    )
    if pose is None:
        return None
    tangent_x, tangent_y = pose[1]
    right_x, right_y = -tangent_y, tangent_x
    vehicle_center_x = current_vehicle[0]
    vehicle_center_y = current_vehicle[1]
    longitudinal_values = [
        (x_coord - vehicle_center_x) * tangent_x
        + (y_coord - vehicle_center_y) * tangent_y
        for x_coord, y_coord in vehicle_footprint
    ]
    lateral_values = [
        (x_coord - vehicle_center_x) * right_x
        + (y_coord - vehicle_center_y) * right_y
        for x_coord, y_coord in vehicle_footprint
    ]
    pedestrian_radius = actor_planar_bounding_radius(pedestrian)
    if pedestrian_radius is None:
        return None
    # Walker assets are compact; cap a malformed/custom box so the safety
    # fallback cannot claim a collision metres outside the actual vehicle.
    pedestrian_radius = max(0.05, min(1.0, float(pedestrian_radius)))

    def relative_coordinates(
        vehicle_xyz: Tuple[float, float, float],
        pedestrian_xyz: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        relative_x = pedestrian_xyz[0] - vehicle_xyz[0]
        relative_y = pedestrian_xyz[1] - vehicle_xyz[1]
        return (
            relative_x * tangent_x + relative_y * tangent_y,
            relative_x * right_x + relative_y * right_y,
        )

    start_relative = relative_coordinates(
        previous_vehicle,
        previous_pedestrian,
    )
    end_relative = relative_coordinates(
        current_vehicle,
        current_pedestrian,
    )
    if segment_intersects_axis_aligned_box(
        start_relative[0],
        start_relative[1],
        end_relative[0],
        end_relative[1],
        min(longitudinal_values) - pedestrian_radius,
        max(longitudinal_values) + pedestrian_radius,
        min(lateral_values) - pedestrian_radius,
        max(lateral_values) + pedestrian_radius,
    ):
        return "swept relative-motion overlap"
    return None


def pedestrian_distance_from_attack_point(
    pedestrian,
    state: ReactiveVehicleState,
) -> Optional[float]:
    if pedestrian is None:
        return None
    try:
        location = pedestrian.get_location()
    except (AttributeError, RuntimeError):
        return None
    target_x = state.attack_x if state.armed else state.rearm_attack_x
    target_y = state.attack_y if state.armed else state.rearm_attack_y
    return math.hypot(
        float(location.x) - target_x,
        float(location.y) - target_y,
    )


def pedestrian_passed_attack_point(
    pedestrian,
    state: ReactiveVehicleState,
    pass_distance: float,
) -> bool:
    if pedestrian is None or state.pedestrian_direction is None:
        return False
    try:
        location = pedestrian.get_location()
    except (AttributeError, RuntimeError):
        return False
    signed_progress = (
        (float(location.x) - state.attack_x) * state.pedestrian_direction[0]
        + (float(location.y) - state.attack_y) * state.pedestrian_direction[1]
    )
    return signed_progress >= max(0.0, float(pass_distance))


def reset_reactive_vehicle_activation_fields(
    state: ReactiveVehicleState,
) -> None:
    state.attack_x = float(state.nominal_attack_x)
    state.attack_y = float(state.nominal_attack_y)
    state.attack_distance = float(state.nominal_attack_distance)
    state.direction_x = float(state.nominal_direction_x)
    state.direction_y = float(state.nominal_direction_y)
    state.attack_yaw = math.degrees(
        math.atan2(state.direction_y, state.direction_x)
    )
    state.active_target_id = None
    state.pedestrian_direction = None
    state.pedestrian_contact_support = 0.0
    state.front_support = 0.0
    state.active_since = None
    state.aligning_since = None
    state.runout_since = None
    state.runout_start_front_progress = None
    state.contact_since = None
    state.contact_settle_until = None
    state.contact_hold_until = None
    state.contact_settle_wall_until = None
    state.contact_hold_wall_until = None
    state.contact_brake_applied = False
    state.impact_speed = 0.0
    state.impact_impulse = 0.0
    state.last_progress = None
    state.last_progress_time = None
    state.last_status_log_time = None
    state.commanded_speed = 0.0
    state.contact_detected = False
    state.outcome = None
    state.route_segment_index = 0
    state.route_progress = 0.0
    state.route_projection_gap = 0.0
    state.predicted_contact_time = None
    state.best_effort_launch = False
    state.committed = False
    state.approach_hold_since = None
    state.previous_contact_sample_time = None
    state.previous_vehicle_location = None
    state.previous_pedestrian_location = None
    state.transient_failure_count = 0
    state.transient_failure_reason = None
    state.transient_failure_since = None
    state.transient_failure_wall_since = None


def update_reactive_vehicle_encounter_target(
    state: ReactiveVehicleState,
    pedestrian,
    approach: PedestrianTargetApproach,
    simulation_time: float,
) -> None:
    """Latch or refine one crossing target while it is still uncommitted."""
    if math.isfinite(float(approach.route_station)) and float(
        approach.route_station
    ) > 1.0e-6:
        state.attack_x = float(approach.target_x)
        state.attack_y = float(approach.target_y)
        state.attack_distance = float(approach.route_station)
        state.direction_x = float(approach.route_tangent_x)
        state.direction_y = float(approach.route_tangent_y)
    state.attack_yaw = math.degrees(
        math.atan2(state.direction_y, state.direction_x)
    )
    state.rearm_attack_x = state.attack_x
    state.rearm_attack_y = state.attack_y
    state.pedestrian_direction = (
        approach.direction_x,
        approach.direction_y,
    )
    state.predicted_contact_time = (
        float(simulation_time) + max(0.0, float(approach.eta_seconds))
    )
    pedestrian_support = actor_leading_support(
        pedestrian,
        (-state.direction_x, -state.direction_y),
    )
    if pedestrian_support is None:
        pedestrian_support = actor_planar_bounding_radius(pedestrian)
    state.pedestrian_contact_support = max(
        0.0,
        0.0 if pedestrian_support is None else float(pedestrian_support),
    )


def begin_reactive_vehicle_alignment(
    state: ReactiveVehicleState,
    pedestrian,
    approach: PedestrianTargetApproach,
    simulation_time: float,
    args: argparse.Namespace,
    best_effort_launch: bool = False,
) -> bool:
    if not reactive_vehicle_actor_alive(state):
        return False
    stop_reactive_vehicle(state.actor)
    alignment_transform = carla.Transform(
        carla.Location(
            x=float(state.target.x),
            y=float(state.target.y),
            z=(
                float(state.target.z)
                + float(args.vehicle_z_offset)
                + float(args.reactive_activation_z_lift)
            ),
        ),
        carla.Rotation(yaw=float(state.target.yaw)),
    )
    try:
        state.actor.set_simulate_physics(False)
        state.actor.set_transform(alignment_transform)
    except (AttributeError, RuntimeError):
        return False
    try:
        pedestrian.set_simulate_physics(True)
        pedestrian.set_collisions(True)
    except (AttributeError, RuntimeError) as exc:
        LOG.warning(
            "Unable to explicitly enable ego-pedestrian collision physics "
            "for actor id=%s: %s",
            getattr(pedestrian, "id", None),
            exc,
        )
    try:
        if str(pedestrian.attributes.get("is_invincible", "false")).lower() in (
            "1",
            "true",
            "yes",
        ):
            LOG.warning(
                "Ego pedestrian id=%d is_invincible=true; CARLA will not "
                "produce the expected physical death/ragdoll response",
                pedestrian.id,
            )
    except (AttributeError, RuntimeError):
        pass
    reset_reactive_vehicle_activation_fields(state)
    update_reactive_vehicle_encounter_target(
        state,
        pedestrian,
        approach,
        simulation_time,
    )
    state.state = REACTIVE_STATE_ALIGNING
    state.armed = False
    state.aligning_since = float(simulation_time)
    state.active_target_id = int(pedestrian.id)
    state.best_effort_launch = bool(best_effort_launch)
    update_reactive_contact_pose_history(
        state,
        pedestrian,
        simulation_time,
    )
    LOG.info(
        "Reactive vehicle #%d generation=%d id=%d state=ALIGNING "
        "target_pedestrian_id=%d attack=(%.3f, %.3f) "
        "pedestrian_distance=%.2f m ETA=%.2f s closing=%.2f m/s "
        "cross_track_miss=%.2f m crossing_angle=%.1f deg "
        "route_station=%.2f m attack_yaw=%.2f deg best_effort=%s "
        "pedestrian_contact_support=%.2f m",
        state.index,
        state.generation,
        state.actor.id,
        pedestrian.id,
        state.attack_x,
        state.attack_y,
        approach.distance,
        approach.eta_seconds,
        approach.closing_speed,
        approach.cross_track_miss,
        approach.crossing_angle_degrees,
        state.attack_distance,
        state.attack_yaw,
        state.best_effort_launch,
        state.pedestrian_contact_support,
    )
    return True


def cancel_reactive_vehicle_alignment(
    state: ReactiveVehicleState,
    reason: str,
    args: argparse.Namespace,
) -> None:
    LOG.info(
        "Reactive vehicle #%d id=%s alignment cancelled: %s",
        state.index,
        getattr(state.actor, "id", None),
        reason,
    )
    freeze_reactive_vehicle_at_home(state, args)
    reset_reactive_vehicle_activation_fields(state)
    state.state = REACTIVE_STATE_WAITING
    state.armed = True


def begin_reactive_vehicle_runout(
    state: ReactiveVehicleState,
    simulation_time: float,
    reason: str,
    front_progress: Optional[float] = None,
) -> None:
    if state.state == REACTIVE_STATE_RUNOUT:
        return
    state.state = REACTIVE_STATE_RUNOUT
    state.runout_since = float(simulation_time)
    state.runout_start_front_progress = max(
        float(state.attack_distance),
        float(
            state.attack_distance
            if front_progress is None
            else front_progress
        ),
    )
    state.contact_detected = False
    state.outcome = reason
    LOG.info(
        "Reactive vehicle #%d generation=%d id=%s state=RUNOUT "
        "outcome=%s start_front_station=%.2f m",
        state.index,
        state.generation,
        getattr(state.actor, "id", None),
        reason,
        state.runout_start_front_progress,
    )


def begin_reactive_vehicle_contact_settle(
    state: ReactiveVehicleState,
    simulation_time: float,
    reason: str,
    args: argparse.Namespace,
    contact: Optional[ReactiveVehicleContact] = None,
    target_contact: bool = True,
) -> None:
    """Preserve collision physics, then hold the vehicle before retirement."""
    wall_time = time.monotonic()
    if state.state in (
        REACTIVE_STATE_CONTACT_SETTLING,
        REACTIVE_STATE_CONTACT_HOLDING,
    ):
        if contact is not None:
            state.impact_impulse = max(
                state.impact_impulse,
                float(contact.impulse_magnitude),
            )
        if target_contact and not state.contact_detected:
            # An incidental prop/road event can precede the true pedestrian
            # event by one callback/update. The intended target must upgrade
            # that provisional hold and receive its own full physics window.
            state.state = REACTIVE_STATE_CONTACT_SETTLING
            state.contact_since = float(simulation_time)
            state.contact_settle_until = (
                float(simulation_time)
                + float(args.reactive_impact_settle_time)
            )
            state.contact_hold_until = (
                state.contact_settle_until
                + float(args.reactive_contact_hold_time)
            )
            state.contact_settle_wall_until = (
                wall_time + float(args.reactive_impact_settle_time)
            )
            state.contact_hold_wall_until = (
                state.contact_settle_wall_until
                + float(args.reactive_contact_hold_time)
            )
            state.contact_brake_applied = False
            state.contact_detected = True
            state.outcome = reason
            state.impact_speed = reactive_vehicle_planar_speed(state.actor)
            coast_reactive_vehicle_after_contact(state.actor)
            LOG.info(
                "Reactive vehicle #%d id=%s collision hold upgraded to "
                "target pedestrian contact: impact_speed=%.2f m/s "
                "normal_impulse=%.2f frame=%s",
                state.index,
                getattr(state.actor, "id", None),
                state.impact_speed,
                state.impact_impulse,
                "fallback" if contact is None else contact.frame,
            )
        return
    state.state = REACTIVE_STATE_CONTACT_SETTLING
    state.contact_since = float(simulation_time)
    state.contact_settle_until = (
        float(simulation_time) + float(args.reactive_impact_settle_time)
    )
    state.contact_hold_until = (
        state.contact_settle_until + float(args.reactive_contact_hold_time)
    )
    state.contact_settle_wall_until = (
        wall_time + float(args.reactive_impact_settle_time)
    )
    state.contact_hold_wall_until = (
        state.contact_settle_wall_until
        + float(args.reactive_contact_hold_time)
    )
    state.contact_brake_applied = False
    state.contact_detected = bool(target_contact)
    state.outcome = reason
    state.impact_speed = reactive_vehicle_planar_speed(state.actor)
    state.impact_impulse = (
        0.0 if contact is None else float(contact.impulse_magnitude)
    )
    if not coast_reactive_vehicle_after_contact(state.actor):
        LOG.warning(
            "Reactive vehicle #%d id=%s could not switch to neutral coast "
            "after collision; leaving physics enabled for settlement",
            state.index,
            getattr(state.actor, "id", None),
        )
    LOG.info(
        "Reactive vehicle #%d generation=%d id=%s "
        "state=CONTACT_SETTLING target_contact=%s impact_speed=%.2f m/s "
        "normal_impulse=%.2f frame=%s settle=%.2f s hold=%.2f s outcome=%s",
        state.index,
        state.generation,
        getattr(state.actor, "id", None),
        bool(target_contact),
        state.impact_speed,
        state.impact_impulse,
        "fallback" if contact is None else contact.frame,
        args.reactive_impact_settle_time,
        args.reactive_contact_hold_time,
        reason,
    )


def update_reactive_vehicle_contact_hold(
    state: ReactiveVehicleState,
    simulation_time: float,
    registry: ReactiveVehicleCollisionRegistry,
    args: argparse.Namespace,
) -> None:
    wall_time = time.monotonic()
    if state.state == REACTIVE_STATE_CONTACT_SETTLING:
        if (
            (
                state.contact_settle_until is not None
                and simulation_time + 1.0e-9 < state.contact_settle_until
            )
            or (
                state.contact_settle_wall_until is not None
                and wall_time + 1.0e-9 < state.contact_settle_wall_until
            )
        ):
            return
        state.state = REACTIVE_STATE_CONTACT_HOLDING
        current_speed = reactive_vehicle_planar_speed(state.actor)
        state.contact_brake_applied = apply_reactive_vehicle_service_brake(
            state.actor,
            engage_parking_brake=(
                current_speed <= float(args.reactive_contact_stop_speed)
            ),
        )
        LOG.info(
            "Reactive vehicle #%d id=%s state=CONTACT_HOLDING "
            "speed=%.2f m/s visible_until=%.3f",
            state.index,
            getattr(state.actor, "id", None),
            current_speed,
            (
                simulation_time
                if state.contact_hold_until is None
                else state.contact_hold_until
            ),
        )

    if state.state != REACTIVE_STATE_CONTACT_HOLDING:
        return
    current_speed = reactive_vehicle_planar_speed(state.actor)
    state.contact_brake_applied = apply_reactive_vehicle_service_brake(
        state.actor,
        engage_parking_brake=(
            current_speed <= float(args.reactive_contact_stop_speed)
        ),
    )
    if (
        (
            state.contact_hold_until is not None
            and simulation_time + 1.0e-9 < state.contact_hold_until
        )
        or (
            state.contact_hold_wall_until is not None
            and wall_time + 1.0e-9 < state.contact_hold_wall_until
        )
    ):
        return
    request_reactive_vehicle_respawn(
        state,
        simulation_time,
        "completed physical collision settle/visible hold after {}".format(
            state.outcome or "contact"
        ),
        registry,
        args,
    )


def retire_reactive_vehicle_actors(
    state: ReactiveVehicleState,
    registry: ReactiveVehicleCollisionRegistry,
) -> None:
    vehicle = state.actor
    vehicle_id = getattr(vehicle, "id", None)
    if vehicle_id is not None:
        registry.unregister(int(vehicle_id))

    # Stop the moving parent before tearing down any of its children.
    stop_reactive_vehicle(vehicle)

    sensor = state.sensor
    sensor_id = getattr(sensor, "id", None)
    if sensor is not None:
        try:
            if sensor.is_alive:
                sensor.stop()
        except (AttributeError, RuntimeError):
            pass
        destroy_actor(sensor, "reactive vehicle collision sensor")
    state.sensor = None

    destroy_actor(vehicle, "reactive vehicle blocker")
    state.actor = None
    state.retired_sensor_id = None if sensor_id is None else int(sensor_id)
    state.retired_actor_id = None if vehicle_id is None else int(vehicle_id)


def request_reactive_vehicle_respawn(
    state: ReactiveVehicleState,
    simulation_time: float,
    reason: str,
    registry: ReactiveVehicleCollisionRegistry,
    args: argparse.Namespace,
) -> None:
    if state.state == REACTIVE_STATE_RESPAWN_PENDING:
        return
    LOG.info(
        "Reactive vehicle #%d generation=%d id=%s reset requested after %s",
        state.index,
        state.generation,
        getattr(state.actor, "id", None),
        reason,
    )
    retire_reactive_vehicle_actors(state, registry)
    reset_reactive_vehicle_activation_fields(state)
    state.state = REACTIVE_STATE_RESPAWN_PENDING
    state.armed = False
    state.respawn_due = (
        float(simulation_time) + float(args.reactive_respawn_delay)
    )


def attempt_reactive_vehicle_respawn(
    state: ReactiveVehicleState,
    world: carla.World,
    simulation_time: float,
    registry: ReactiveVehicleCollisionRegistry,
    args: argparse.Namespace,
) -> bool:
    if state.state != REACTIVE_STATE_RESPAWN_PENDING:
        return False
    if state.respawn_due is not None and simulation_time + 1.0e-9 < state.respawn_due:
        return False
    sensor_absent = retired_actor_is_absent(
        world,
        state.retired_sensor_id,
        "reactive vehicle collision sensor",
    )
    vehicle_absent = retired_actor_is_absent(
        world,
        state.retired_actor_id,
        "reactive vehicle blocker",
    )
    if not sensor_absent or not vehicle_absent:
        state.respawn_due = (
            simulation_time + args.reactive_respawn_retry
        )
        return False
    state.retired_sensor_id = None
    state.retired_actor_id = None

    home_location = vehicle_target_transform(
        state.target,
        args.vehicle_z_offset,
    ).location
    try:
        occupying_actor = blocking_actor_at_location(
            world,
            home_location,
            args.reactive_respawn_clearance,
            ignored_ids=state.home_clearance_ignored_actor_ids,
        )
    except RuntimeError as exc:
        state.respawn_due = simulation_time + args.reactive_respawn_retry
        LOG.warning(
            "Reactive vehicle #%d deferring respawn for %.2f s: %s",
            state.index,
            args.reactive_respawn_retry,
            exc,
        )
        return False
    if occupying_actor is not None:
        state.respawn_due = simulation_time + args.reactive_respawn_retry
        LOG.info(
            "Reactive vehicle #%d waiting to respawn: actor id=%d type=%s "
            "occupies its home within %.2f m",
            state.index,
            occupying_actor.id,
            occupying_actor.type_id,
            args.reactive_respawn_clearance,
        )
        return False

    try:
        vehicle = spawn_reactive_vehicle(
            world,
            state.target,
            state.index,
            args,
        )
    except ValueError as exc:
        vehicle = None
        LOG.warning("Reactive vehicle respawn blueprint error: %s", exc)
    if vehicle is None:
        state.respawn_due = simulation_time + args.reactive_respawn_retry
        return False
    state.actor = vehicle
    if not freeze_reactive_vehicle_at_home(state, args):
        vehicle_id = int(vehicle.id)
        stop_reactive_vehicle(vehicle)
        destroy_actor(vehicle, "partially spawned reactive vehicle")
        state.actor = None
        state.retired_actor_id = vehicle_id
        state.respawn_due = simulation_time + args.reactive_respawn_retry
        return False
    state.sensor = spawn_reactive_vehicle_collision_sensor(
        world,
        vehicle,
        registry,
    )
    reset_reactive_vehicle_activation_fields(state)
    state.generation += 1
    state.state = REACTIVE_STATE_WAITING
    state.armed = False
    state.respawn_due = None
    LOG.info(
        "Reactive vehicle #%d generation=%d id=%d respawned WAITING at "
        "home=(%.3f, %.3f, %.3f, yaw=%.2f); rearming after the ego "
        "pedestrian leaves %.2f m",
        state.index,
        state.generation,
        vehicle.id,
        state.target.x,
        state.target.y,
        state.target.z + args.vehicle_z_offset,
        state.target.yaw,
        args.reactive_rearm_distance,
    )
    return True


def reactive_contact_matches_target(
    contact: ReactiveVehicleContact,
    state: ReactiveVehicleState,
    args: argparse.Namespace,
) -> bool:
    """Match the latched walker despite a delayed callback/Y actor swap."""
    if state.active_target_id is not None:
        try:
            if int(contact.actor_id) == int(state.active_target_id):
                return True
        except (TypeError, ValueError):
            pass
    return (
        str(contact.type_id).startswith("walker.pedestrian.")
        and bool(args.ego_pedestrian_role_name)
        and str(contact.role_name) == str(args.ego_pedestrian_role_name)
    )


def defer_or_request_reactive_vehicle_respawn(
    state: ReactiveVehicleState,
    simulation_time: float,
    reason: str,
    registry: ReactiveVehicleCollisionRegistry,
    args: argparse.Namespace,
) -> bool:
    """Give an asynchronous collision callback a few updates to arrive.

    Returns True while teardown is deferred and False once a respawn was
    requested.  A single post-impact RPC/projection failure must never erase
    the vehicle in the same rendered frame as contact.
    """
    state.transient_failure_count += 1
    state.transient_failure_reason = str(reason)
    wall_time = time.monotonic()
    if state.transient_failure_since is None:
        state.transient_failure_since = float(simulation_time)
    if state.transient_failure_wall_since is None:
        state.transient_failure_wall_since = wall_time
    simulation_grace_elapsed = (
        float(simulation_time) - state.transient_failure_since
        >= DEFAULT_REACTIVE_CONTACT_SWEEP_MAX_INTERVAL_SECONDS
    )
    wall_grace_elapsed = (
        wall_time - state.transient_failure_wall_since
        >= DEFAULT_REACTIVE_CONTACT_SWEEP_MAX_INTERVAL_SECONDS
    )
    if (
        state.transient_failure_count < DEFAULT_REACTIVE_TRANSIENT_FAILURE_LIMIT
        or not simulation_grace_elapsed
        or not wall_grace_elapsed
    ):
        LOG.warning(
            "Reactive vehicle #%d id=%s deferring teardown after transient "
            "failure count=%d (minimum=%d, grace=%.2f s): %s",
            state.index,
            getattr(state.actor, "id", None),
            state.transient_failure_count,
            DEFAULT_REACTIVE_TRANSIENT_FAILURE_LIMIT,
            DEFAULT_REACTIVE_CONTACT_SWEEP_MAX_INTERVAL_SECONDS,
            reason,
        )
        return True
    request_reactive_vehicle_respawn(
        state,
        simulation_time,
        "{} (repeated {} updates)".format(
            reason,
            state.transient_failure_count,
        ),
        registry,
        args,
    )
    return False


def clear_reactive_vehicle_transient_failure(
    state: ReactiveVehicleState,
) -> None:
    state.transient_failure_count = 0
    state.transient_failure_reason = None
    state.transient_failure_since = None
    state.transient_failure_wall_since = None


def update_reactive_vehicle(
    state: ReactiveVehicleState,
    world: carla.World,
    pedestrian,
    simulation_time: float,
    registry: ReactiveVehicleCollisionRegistry,
    args: argparse.Namespace,
) -> None:
    """Advance the crosswalk-attack state machine on the passive main loop."""
    if state.state == REACTIVE_STATE_RESPAWN_PENDING:
        attempt_reactive_vehicle_respawn(
            state,
            world,
            simulation_time,
            registry,
            args,
        )
        return
    if not reactive_vehicle_actor_alive(state):
        request_reactive_vehicle_respawn(
            state,
            simulation_time,
            "vehicle actor was removed outside this client",
            registry,
            args,
        )
        return

    if state.sensor is not None:
        try:
            sensor_alive = bool(state.sensor.is_alive)
        except (AttributeError, RuntimeError):
            sensor_alive = False
        if not sensor_alive:
            registry.unregister(int(state.actor.id))
            LOG.warning(
                "Reactive vehicle #%d id=%d collision sensor was removed; "
                "using geometry-only contact detection until respawn",
                state.index,
                state.actor.id,
            )
            state.sensor = None

    contacts = registry.consume_contacts(int(state.actor.id))
    matching_contact = None
    if state.active_target_id is not None:
        matching_contact = next(
            (
                contact
                for contact in contacts
                if reactive_contact_matches_target(contact, state, args)
            ),
            None,
        )
    if state.state in (
        REACTIVE_STATE_CONTACT_SETTLING,
        REACTIVE_STATE_CONTACT_HOLDING,
    ):
        if matching_contact is not None and not state.contact_detected:
            begin_reactive_vehicle_contact_settle(
                state,
                simulation_time,
                "collision sensor contact with ego pedestrian id={}".format(
                    matching_contact.actor_id
                ),
                args,
                contact=matching_contact,
                target_contact=True,
            )
        # Once target contact owns the lifecycle, secondary collision events
        # are expected while bodies separate and must not tear down the actor
        # before the camera has rendered the physical response.
        update_reactive_vehicle_contact_hold(
            state,
            simulation_time,
            registry,
            args,
        )
        return

    if matching_contact is not None and state.state in (
        REACTIVE_STATE_ACTIVE,
        REACTIVE_STATE_RUNOUT,
    ):
        begin_reactive_vehicle_contact_settle(
            state,
            simulation_time,
            "collision sensor contact with ego pedestrian id={}".format(
                matching_contact.actor_id
            ),
            args,
            contact=matching_contact,
            target_contact=True,
        )
        # Target contact dominates any simultaneous environment/prop event.
        return
    unexpected_contact = next(
        (
            contact
            for contact in contacts
            if not reactive_contact_matches_target(contact, state, args)
        ),
        None,
    )
    if (
        unexpected_contact is not None
        and state.state in (REACTIVE_STATE_ACTIVE, REACTIVE_STATE_RUNOUT)
    ):
        begin_reactive_vehicle_contact_settle(
            state,
            simulation_time,
            "unexpected route collision with id={} type={}".format(
                unexpected_contact.actor_id,
                unexpected_contact.type_id,
            ),
            args,
            contact=unexpected_contact,
            target_contact=False,
        )
        return

    pedestrian_matches = False
    if pedestrian is not None and state.active_target_id is not None:
        try:
            pedestrian_matches = (
                int(pedestrian.id) == int(state.active_target_id)
                and bool(pedestrian.is_alive)
            )
        except (AttributeError, RuntimeError):
            pedestrian_matches = False
    if (
        pedestrian_matches
        and state.state in (REACTIVE_STATE_ACTIVE, REACTIVE_STATE_RUNOUT)
    ):
        contact_source = reactive_vehicle_contact_fallback(state, pedestrian)
        if contact_source is None:
            contact_source = reactive_vehicle_swept_contact_fallback(
                state,
                pedestrian,
                simulation_time,
            )
        else:
            # Keep the history current even when exact overlap wins so a
            # subsequent callback/update cannot bridge an old stale sample.
            update_reactive_contact_pose_history(
                state,
                pedestrian,
                simulation_time,
            )
        if contact_source is not None:
            begin_reactive_vehicle_contact_settle(
                state,
                simulation_time,
                "ego-pedestrian contact detected by {}".format(contact_source),
                args,
                contact=None,
                target_contact=True,
            )
            return

    if state.state == REACTIVE_STATE_WAITING:
        if not state.armed:
            distance = pedestrian_distance_from_attack_point(pedestrian, state)
            if pedestrian is None or (
                distance is not None
                and distance >= float(args.reactive_rearm_distance)
            ):
                state.armed = True
                LOG.info(
                    "Reactive vehicle #%d id=%d is ARMED for a fresh "
                    "ego-pedestrian approach",
                    state.index,
                    state.actor.id,
                )
            return
        geometry = reactive_vehicle_motion_geometry(state)
        if geometry is None:
            return
        _, front_progress, _, _ = geometry
        approach = pedestrian_approach_for_actor(
            pedestrian,
            args,
            state=state,
            route_crossing=True,
            minimum_route_station=front_progress + 0.05,
        )
        if approach is None:
            return
        pedestrian_support = actor_leading_support(
            pedestrian,
            (-approach.route_tangent_x, -approach.route_tangent_y),
        )
        if pedestrian_support is None:
            pedestrian_support = actor_planar_bounding_radius(pedestrian)
        contact_station = max(
            0.0,
            approach.route_station
            - max(
                0.0,
                0.0 if pedestrian_support is None else pedestrian_support,
            ),
        )
        front_remaining = max(
            0.0,
            contact_station - front_progress,
        )
        effective_max_speed = state.route_speed_limit
        if args.reactive_vehicle_control == "constant-velocity":
            minimum_time = front_remaining / effective_max_speed
        else:
            minimum_time = minimum_vehicle_travel_time(
                front_remaining,
                0.0,
                effective_max_speed,
                args.reactive_vehicle_max_acceleration,
            )
        available_time = approach.eta_seconds - args.reactive_impact_lead
        best_effort_launch = (
            approach.eta_seconds < args.reactive_min_pedestrian_eta
            or minimum_time > available_time + 1.0e-9
        )
        if best_effort_launch and not args.reactive_best_effort_launch:
            if (
                state.last_status_log_time is None
                or simulation_time - state.last_status_log_time >= 1.0
            ):
                LOG.info(
                    "Reactive vehicle #%d cannot safely reach the live "
                    "crossing on this approach: minimum_vehicle_time=%.2f s "
                    "available=%.2f s; waiting for the next crossing",
                    state.index,
                    minimum_time,
                    available_time,
                )
                state.last_status_log_time = simulation_time
            return
        if best_effort_launch and (
            state.last_status_log_time is None
            or simulation_time - state.last_status_log_time >= 1.0
        ):
            LOG.warning(
                "Reactive vehicle #%d launching best-effort at the curve-safe "
                "limit: pedestrian crossing station=%.2f m ETA=%.2f s but "
                "minimum vehicle time is %.2f s; physical contact cannot be "
                "guaranteed",
                state.index,
                approach.route_station,
                approach.eta_seconds,
                minimum_time,
            )
            state.last_status_log_time = simulation_time
        if not begin_reactive_vehicle_alignment(
            state,
            pedestrian,
            approach,
            simulation_time,
            args,
            best_effort_launch=best_effort_launch,
        ):
            request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "failed to align the attack vehicle",
                registry,
                args,
            )
        return

    if state.state == REACTIVE_STATE_ALIGNING:
        if not pedestrian_matches:
            cancel_reactive_vehicle_alignment(
                state,
                "latched ego pedestrian disappeared or respawned",
                args,
            )
            return
        geometry = reactive_vehicle_motion_geometry(state)
        if geometry is None:
            request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "vehicle geometry unavailable after alignment",
                registry,
                args,
            )
            return
        _, front_progress, _, current_speed = geometry
        route_approach = pedestrian_approach_for_actor(
            pedestrian,
            args,
            state=state,
            route_crossing=True,
            minimum_route_station=front_progress + 0.05,
        )
        if route_approach is not None:
            update_reactive_vehicle_encounter_target(
                state,
                pedestrian,
                route_approach,
                simulation_time,
            )
            approach = route_approach
        else:
            approach = pedestrian_approach_for_actor(
                pedestrian,
                args,
                state=state,
                tracking=True,
            )
            if approach is not None:
                update_reactive_vehicle_encounter_target(
                    state,
                    pedestrian,
                    approach,
                    simulation_time,
                )
        if approach is None:
            aligning_elapsed = (
                0.0
                if state.aligning_since is None
                else simulation_time - state.aligning_since
            )
            if aligning_elapsed >= args.reactive_approach_hold_timeout:
                cancel_reactive_vehicle_alignment(
                    state,
                    "live pedestrian approach remained unavailable while staged",
                    args,
                )
            elif (
                state.last_status_log_time is None
                or simulation_time - state.last_status_log_time >= 1.0
            ):
                LOG.info(
                    "Reactive vehicle #%d id=%d remains STAGED at road home: "
                    "waiting for a fresh pedestrian ETA",
                    state.index,
                    state.actor.id,
                )
                state.last_status_log_time = simulation_time
            return
        pedestrian_eta = approach.eta_seconds
        contact_station = max(
            0.0,
            state.attack_distance - state.pedestrian_contact_support,
        )
        front_remaining = max(0.0, contact_station - front_progress)
        effective_max_speed = state.route_speed_limit
        timing_ready, minimum_time, available_time = (
            reactive_vehicle_timing_window(
                front_remaining,
                current_speed,
                pedestrian_eta,
                effective_max_speed,
                args,
            )
        )
        newly_infeasible = (
            approach.eta_seconds < args.reactive_min_pedestrian_eta
        ) or minimum_time > (
            available_time
        ) + 0.10
        if newly_infeasible:
            if args.reactive_best_effort_launch:
                state.best_effort_launch = True
            else:
                cancel_reactive_vehicle_alignment(
                    state,
                    "pedestrian ETA became infeasible during the alignment tick",
                    args,
                )
                return
        elif not timing_ready:
            aligning_elapsed = (
                0.0
                if state.aligning_since is None
                else simulation_time - state.aligning_since
            )
            if aligning_elapsed >= args.reactive_approach_hold_timeout:
                cancel_reactive_vehicle_alignment(
                    state,
                    "staged launch window did not arrive before timeout",
                    args,
                )
            elif (
                state.last_status_log_time is None
                or simulation_time - state.last_status_log_time >= 1.0
            ):
                LOG.info(
                    "Reactive vehicle #%d id=%d remains STAGED at road home: "
                    "pedestrian_ETA=%.2f s vehicle_minimum=%.2f s "
                    "launch_margin=%.2f s",
                    state.index,
                    state.actor.id,
                    pedestrian_eta,
                    minimum_time,
                    args.reactive_launch_margin,
                )
                state.last_status_log_time = simulation_time
            return
        try:
            state.actor.set_simulate_physics(True)
            state.actor.set_collisions(True)
        except (AttributeError, RuntimeError):
            request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "failed to enable vehicle physics",
                registry,
                args,
            )
            return
        if not release_reactive_vehicle_brake(state.actor):
            request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "failed to release vehicle parking brake",
                registry,
                args,
            )
            return
        command = synchronized_vehicle_command(
            front_remaining,
            current_speed,
            pedestrian_eta,
            args.reactive_impact_lead,
            args.reactive_vehicle_speed_min,
            effective_max_speed,
            args.reactive_vehicle_max_acceleration,
            args.reactive_vehicle_max_deceleration,
            args.reactive_commit_distance,
            min(args.reactive_commit_speed, effective_max_speed),
            args.reactive_vehicle_control,
            commit_ready=False,
        )
        state.state = REACTIVE_STATE_ACTIVE
        state.active_since = float(simulation_time)
        state.approach_hold_since = None
        state.committed = False
        state.last_progress = front_progress
        state.last_progress_time = float(simulation_time)
        if not apply_reactive_vehicle_command(
            state,
            command.speed,
            command.acceleration,
            args,
        ):
            request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "initial motion command failed",
                registry,
                args,
            )
            return
        LOG.info(
            "Reactive vehicle #%d generation=%d id=%d state=ACTIVE "
            "control=%s front_remaining=%.2f m pedestrian_ETA=%.2f s "
            "command_speed=%.2f m/s command_acceleration=%.2f m/s2",
            state.index,
            state.generation,
            state.actor.id,
            args.reactive_vehicle_control,
            front_remaining,
            pedestrian_eta,
            command.speed,
            command.acceleration,
        )
        return

    geometry = reactive_vehicle_motion_geometry(state)
    if geometry is None:
        defer_or_request_reactive_vehicle_respawn(
            state,
            simulation_time,
            "active vehicle geometry became unavailable",
            registry,
            args,
        )
        return
    _, front_progress, cross_track, current_speed = geometry
    if abs(cross_track) > args.reactive_route_deviation_limit:
        defer_or_request_reactive_vehicle_respawn(
            state,
            simulation_time,
            "vehicle left its Driving-lane route by {:.2f} m (limit {:.2f} m)".format(
                abs(cross_track),
                args.reactive_route_deviation_limit,
            ),
            registry,
            args,
        )
        return
    if (
        state.state == REACTIVE_STATE_ACTIVE
        and not state.committed
        and pedestrian_matches
        and (
            state.attack_distance
            - state.pedestrian_contact_support
            - front_progress
        )
        > args.reactive_commit_distance
    ):
        refined_approach = pedestrian_approach_for_actor(
            pedestrian,
            args,
            state=state,
            route_crossing=True,
            minimum_route_station=front_progress + 0.05,
        )
        if refined_approach is not None:
            previous_station = state.attack_distance
            update_reactive_vehicle_encounter_target(
                state,
                pedestrian,
                refined_approach,
                simulation_time,
            )
            if abs(state.attack_distance - previous_station) >= 0.50:
                LOG.info(
                    "Reactive vehicle #%d refined live crossing station "
                    "from %.2f m to %.2f m before commit",
                    state.index,
                    previous_station,
                    state.attack_distance,
                )
    contact_station = max(
        0.0,
        state.attack_distance - state.pedestrian_contact_support,
    )
    front_remaining = contact_station - front_progress

    if state.state == REACTIVE_STATE_ACTIVE:
        approach = (
            pedestrian_approach_for_actor(
                pedestrian,
                args,
                state=state,
                tracking=True,
            )
            if pedestrian_matches
            else None
        )
        timing_ready = False
        if approach is not None:
            if not state.committed:
                update_reactive_vehicle_encounter_target(
                    state,
                    pedestrian,
                    approach,
                    simulation_time,
                )
                contact_station = max(
                    0.0,
                    state.attack_distance - state.pedestrian_contact_support,
                )
                front_remaining = contact_station - front_progress
            timing_ready, _minimum_time, _available_time = (
                reactive_vehicle_timing_window(
                    front_remaining,
                    current_speed,
                    approach.eta_seconds,
                    state.route_speed_limit,
                    args,
                )
            )
            if (
                not state.committed
                and reactive_vehicle_commit_ready(
                    front_remaining,
                    current_speed,
                    approach.eta_seconds,
                    state.route_speed_limit,
                    args,
                )
            ):
                state.committed = True
                state.approach_hold_since = None
                state.active_since = float(simulation_time)
                LOG.info(
                    "Reactive vehicle #%d id=%d final approach COMMITTED: "
                    "front_remaining=%.2f m pedestrian_ETA=%.2f s",
                    state.index,
                    state.actor.id,
                    front_remaining,
                    approach.eta_seconds,
                )

        should_hold_for_pedestrian = (
            not state.committed
            and (
                approach is None
                or not timing_ready
                or front_progress >= state.attack_distance
            )
        )
        if (
            should_hold_for_pedestrian
            and state.approach_hold_since is None
        ):
            state.approach_hold_since = float(simulation_time)
            LOG.info(
                "Reactive vehicle #%d id=%d state=ACTIVE_PRECOMMIT_HOLD "
                "front_remaining=%.2f m pedestrian_ETA=%s hold_line=%.2f m",
                state.index,
                state.actor.id,
                front_remaining,
                "unavailable"
                if approach is None
                else "{:.2f} s".format(approach.eta_seconds),
                args.reactive_approach_hold_distance,
            )

        if pedestrian_matches and pedestrian_passed_attack_point(
            pedestrian,
            state,
            args.reactive_target_pass_distance,
        ):
            begin_reactive_vehicle_runout(
                state,
                simulation_time,
                "ego pedestrian passed the attack point before contact",
                front_progress=front_progress,
            )
        elif not pedestrian_matches:
            begin_reactive_vehicle_runout(
                state,
                simulation_time,
                "latched ego pedestrian disappeared or respawned",
                front_progress=front_progress,
            )
        elif state.committed and front_progress >= state.attack_distance:
            begin_reactive_vehicle_runout(
                state,
                simulation_time,
                "committed vehicle front passed the attack point without "
                "confirmed contact",
                front_progress=front_progress,
            )
        elif (
            state.approach_hold_since is not None
            and simulation_time - state.approach_hold_since
            >= args.reactive_approach_hold_timeout
        ):
            begin_reactive_vehicle_runout(
                state,
                simulation_time,
                "pre-commit pedestrian wait timed out",
                front_progress=front_progress,
            )
        elif (
            state.approach_hold_since is None
            and state.active_since is not None
            and simulation_time - state.active_since
            >= args.reactive_active_timeout
        ):
            begin_reactive_vehicle_runout(
                state,
                simulation_time,
                "synchronized approach timed out",
                front_progress=front_progress,
            )

    if state.state == REACTIVE_STATE_ACTIVE:
        if should_hold_for_pedestrian:
            command = reactive_vehicle_approach_hold_command(
                front_remaining,
                current_speed,
                args.reactive_approach_hold_distance,
                args.reactive_vehicle_max_deceleration,
            )
            desired_speed = command.speed
            desired_acceleration = command.acceleration
            eta_for_log = None if approach is None else approach.eta_seconds
        elif state.committed and approach is None:
            # A fresh ETA justified the final approach.  If the pedestrian
            # changes direction after that no-return decision, continue the
            # lane-following miss rather than stopping on the crosswalk.
            desired_speed = max(0.0, state.commanded_speed)
            if (
                front_remaining <= args.reactive_commit_distance
                and args.reactive_commit_distance > 1.0e-6
            ):
                commit_progress = max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - front_remaining / args.reactive_commit_distance,
                    ),
                )
                desired_speed = max(
                    desired_speed,
                    commit_progress
                    * min(
                        max(
                            args.reactive_vehicle_speed_min,
                            args.reactive_commit_speed,
                        ),
                        state.route_speed_limit,
                    ),
                )
            desired_speed = min(desired_speed, state.route_speed_limit)
            desired_acceleration = bounded_acceleration_toward_speed(
                current_speed,
                desired_speed,
                args.reactive_vehicle_max_acceleration,
                args.reactive_vehicle_max_deceleration,
            )
            eta_for_log = None
        else:
            if state.approach_hold_since is not None:
                LOG.info(
                    "Reactive vehicle #%d id=%d released pre-commit hold: "
                    "pedestrian_ETA=%.2f s",
                    state.index,
                    state.actor.id,
                    approach.eta_seconds,
                )
                state.approach_hold_since = None
                state.active_since = float(simulation_time)
            command = synchronized_vehicle_command(
                front_remaining,
                current_speed,
                approach.eta_seconds,
                args.reactive_impact_lead,
                args.reactive_vehicle_speed_min,
                state.route_speed_limit,
                args.reactive_vehicle_max_acceleration,
                args.reactive_vehicle_max_deceleration,
                args.reactive_commit_distance,
                min(args.reactive_commit_speed, state.route_speed_limit),
                args.reactive_vehicle_control,
                commit_ready=state.committed,
            )
            desired_speed = command.speed
            desired_acceleration = command.acceleration
            eta_for_log = approach.eta_seconds
        if not apply_reactive_vehicle_command(
            state,
            desired_speed,
            desired_acceleration,
            args,
        ):
            defer_or_request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "active vehicle motion command failed",
                registry,
                args,
            )
            return
        clear_reactive_vehicle_transient_failure(state)
        if (
            state.last_status_log_time is None
            or simulation_time - state.last_status_log_time >= 1.0
        ):
            LOG.info(
                "Reactive vehicle #%d id=%d ACTIVE front_remaining=%.2f m "
                "speed=%.2f m/s command=%.2f m/s pedestrian_ETA=%s "
                "committed=%s precommit_hold=%s cross_track=%.2f m",
                state.index,
                state.actor.id,
                front_remaining,
                current_speed,
                desired_speed,
                "unavailable" if eta_for_log is None else "{:.2f} s".format(
                    eta_for_log
                ),
                state.committed,
                state.approach_hold_since is not None,
                cross_track,
            )
            state.last_status_log_time = simulation_time
        return

    if state.state == REACTIVE_STATE_RUNOUT:
        runout_start = (
            state.attack_distance
            if state.runout_start_front_progress is None
            else state.runout_start_front_progress
        )
        runout_goal = runout_start + args.reactive_post_distance
        if front_progress >= runout_goal:
            request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "completed {:.2f} m post-target drive-through after {}".format(
                    args.reactive_post_distance,
                    state.outcome or "encounter",
                ),
                registry,
                args,
            )
            return
        if (
            state.runout_since is not None
            and simulation_time - state.runout_since
            >= args.reactive_runout_timeout
        ):
            request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "post-target drive-through timeout after {}".format(
                    state.outcome or "encounter"
                ),
                registry,
                args,
            )
            return
        if not apply_reactive_vehicle_command(
            state,
            min(args.reactive_runout_speed, state.route_speed_limit),
            bounded_acceleration_toward_speed(
                current_speed,
                min(args.reactive_runout_speed, state.route_speed_limit),
                args.reactive_vehicle_max_acceleration,
                args.reactive_vehicle_max_deceleration,
            ),
            args,
        ):
            defer_or_request_reactive_vehicle_respawn(
                state,
                simulation_time,
                "runout motion command failed",
                registry,
                args,
            )
            return
        clear_reactive_vehicle_transient_failure(state)


def snapshot_time(snapshot) -> float:
    try:
        return float(snapshot.timestamp.elapsed_seconds)
    except (AttributeError, TypeError, ValueError):
        return time.monotonic()


def rebase_state_timers_after_clock_rewind(
    states: Sequence[PedestrianState],
    previous_simulation_time: float,
    simulation_time: float,
) -> None:
    """Shift timers to a rewound CARLA clock while preserving durations."""
    clock_shift = float(simulation_time - previous_simulation_time)

    def shifted(timer_value: Optional[float]) -> Optional[float]:
        if timer_value is None:
            return None
        try:
            numeric_value = float(timer_value)
        except (TypeError, ValueError):
            return float(simulation_time)
        if not math.isfinite(numeric_value):
            return float(simulation_time)
        return numeric_value + clock_shift

    for state in states:
        if state.state == STATE_ACTIVE:
            state.active_since = shifted(state.active_since)
            state.last_progress_time = shifted(state.last_progress_time)
            state.last_motion_update_time = shifted(
                state.last_motion_update_time
            )
            state.scripted_recovery_started_at = shifted(
                state.scripted_recovery_started_at
            )
            state.last_debug_draw_time = shifted(state.last_debug_draw_time)
        elif state.state == STATE_HOLDING:
            state.hold_until = shifted(state.hold_until)
        elif state.state == STATE_RESPAWN_PENDING:
            state.respawn_due = shifted(state.respawn_due)


def rebase_reactive_vehicle_timers_after_clock_rewind(
    state: Optional[ReactiveVehicleState],
    previous_simulation_time: float,
    simulation_time: float,
) -> None:
    if state is None:
        return
    clock_shift = float(simulation_time - previous_simulation_time)

    def shifted(timer_value: Optional[float]) -> Optional[float]:
        if timer_value is None:
            return None
        try:
            numeric_value = float(timer_value)
        except (TypeError, ValueError):
            return float(simulation_time)
        if not math.isfinite(numeric_value):
            return float(simulation_time)
        return numeric_value + clock_shift

    state.aligning_since = shifted(state.aligning_since)
    state.active_since = shifted(state.active_since)
    state.runout_since = shifted(state.runout_since)
    state.contact_since = shifted(state.contact_since)
    state.contact_settle_until = shifted(state.contact_settle_until)
    state.contact_hold_until = shifted(state.contact_hold_until)
    state.predicted_contact_time = shifted(state.predicted_contact_time)
    state.approach_hold_since = shifted(state.approach_hold_since)
    state.previous_contact_sample_time = shifted(
        state.previous_contact_sample_time
    )
    state.transient_failure_since = shifted(state.transient_failure_since)
    state.last_progress_time = shifted(state.last_progress_time)
    state.last_status_log_time = shifted(state.last_status_log_time)
    state.respawn_due = shifted(state.respawn_due)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    world = None
    blocker_vehicles: List[object] = []
    network_profile_publication: Optional[PublishedNetworkProfileActors] = None
    reactive_vehicle_state: Optional[ReactiveVehicleState] = None
    states: List[PedestrianState] = []
    registry = CollisionRegistry()
    reactive_vehicle_registry = ReactiveVehicleCollisionRegistry()
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
        carla_map = world.get_map()
        settings = world.get_settings()
        deterministic_ragdolls = getattr(
            settings,
            "deterministic_ragdolls",
            None,
        )
        LOG.info(
            "Connected to %s at %s:%d synchronous_mode=%s "
            "deterministic_ragdolls=%s",
            carla_map.name,
            args.host,
            args.port,
            settings.synchronous_mode,
            deterministic_ragdolls,
        )
        if settings.synchronous_mode:
            LOG.info(
                "Passive client: the world is already synchronous, so its "
                "existing clock owner must advance ticks; this script never "
                "calls world.tick() or changes settings"
            )
        else:
            LOG.info(
                "Passive client: the asynchronous CARLA server advances "
                "simulation ticks automatically; no other CARLA client is "
                "required, and this script never calls world.tick() or "
                "changes settings"
            )
        if deterministic_ragdolls is True:
            LOG.warning(
                "WorldSettings.deterministic_ragdolls=True uses the less "
                "physical deterministic pedestrian death animation. Set it "
                "False in the world clock/settings owner before this demo if "
                "a physical ragdoll response is required. This passive client "
                "will not mutate world settings."
            )

        using_built_in_locations = (
            not args.no_vehicle_blockers
            and args.vehicle_locations is None
        ) or (
            not args.no_pedestrian_blockers
            and args.pedestrian_locations is None
            and not args.from_spectator
        )
        if using_built_in_locations and not carla_map.name.endswith("Town10HD_Opt"):
            LOG.warning(
                "Built-in blocker transforms were captured in Town10HD_Opt, "
                "but the connected map is %s",
                carla_map.name,
            )

        LOG.info(
            "Spatial-map RGB/radar inventory is virtual: this client will not "
            "load a traffic-light route or spawn camera/radar actors on "
            "vehicles, pedestrians, or traffic-light poles"
        )
        if args.start_active_spatial_map_sensors:
            LOG.warning(
                "Deprecated --start-active-spatial-map-sensors was ignored; "
                "the shared profile will publish sensor streaming disabled"
            )

        network_profile_publication = publish_network_degradation_profile(
            world,
            args.network_degradation_zones,
            False,
            args.replace_existing_network_profile,
        )

        vehicle_targets = resolve_vehicle_targets(args)
        reactive_enabled = (
            bool(vehicle_targets)
            and not args.no_reactive_vehicle
        )
        if (
            reactive_enabled
            and args.reactive_vehicle_index > len(vehicle_targets)
        ):
            raise ValueError(
                "--reactive-vehicle-index {} is unavailable: only {} vehicle "
                "location(s) were configured; add locations, choose another "
                "index, or pass --no-reactive-vehicle".format(
                    args.reactive_vehicle_index,
                    len(vehicle_targets),
                )
            )
        if reactive_enabled:
            selected_reactive_target = vehicle_targets[
                args.reactive_vehicle_index - 1
            ]
            if selected_reactive_target.blueprint_id is not None:
                raise ValueError(
                    "--reactive-vehicle-index {} selects a dedicated static "
                    "vehicle target ({!r}); choose another index or replace "
                    "the baked-in locations with --vehicle-location".format(
                        args.reactive_vehicle_index,
                        selected_reactive_target.blueprint_id,
                    )
                )
        for index, target in enumerate(vehicle_targets, 1):
            if reactive_enabled and index == args.reactive_vehicle_index:
                route_plan = build_reactive_vehicle_route_plan(
                    carla_map,
                    target,
                    args,
                )
                reactive_vehicle_state = make_reactive_vehicle_state(
                    index,
                    route_plan,
                    None,
                    args,
                )
                vehicle = spawn_reactive_vehicle(
                    world,
                    reactive_vehicle_state.target,
                    index,
                    args,
                )
                reactive_vehicle_state.actor = vehicle
                if vehicle is not None:
                    reactive_vehicle_state.generation = 1
                    reactive_vehicle_state.state = REACTIVE_STATE_WAITING
                    reactive_vehicle_state.respawn_due = None
                    if not freeze_reactive_vehicle_at_home(
                        reactive_vehicle_state,
                        args,
                    ):
                        vehicle_id = int(vehicle.id)
                        stop_reactive_vehicle(vehicle)
                        destroy_actor(
                            vehicle,
                            "partially initialized reactive vehicle",
                        )
                        reactive_vehicle_state.actor = None
                        reactive_vehicle_state.generation = 0
                        reactive_vehicle_state.retired_actor_id = vehicle_id
                        reactive_vehicle_state.state = (
                            REACTIVE_STATE_RESPAWN_PENDING
                        )
                        reactive_vehicle_state.respawn_due = 0.0
                    else:
                        reactive_vehicle_state.sensor = (
                            spawn_reactive_vehicle_collision_sensor(
                                world,
                                vehicle,
                                reactive_vehicle_registry,
                            )
                        )
                        LOG.info(
                            "Reactive vehicle #%d id=%d type=%s state=WAITING "
                            "road_home=(%.3f, %.3f, %.3f, yaw=%.2f, "
                            "road=%d lane=%d, snap=%.2f m) "
                            "attack=(%.3f, %.3f, road=%d lane=%d, "
                            "lateral_offset=%.2f m) route_distance=%.2f m "
                            "route_length=%.2f m curve=%.4f 1/m "
                            "curve_speed_cap=%.2f m/s attack_yaw=%.2f source=%s",
                            index,
                            vehicle.id,
                            vehicle.type_id,
                            reactive_vehicle_state.target.x,
                            reactive_vehicle_state.target.y,
                            reactive_vehicle_state.target.z
                            + args.vehicle_z_offset,
                            reactive_vehicle_state.target.yaw,
                            reactive_vehicle_state.home_road_id,
                            reactive_vehicle_state.home_lane_id,
                            reactive_vehicle_state.home_snap_distance,
                            reactive_vehicle_state.attack_x,
                            reactive_vehicle_state.attack_y,
                            reactive_vehicle_state.attack_road_id,
                            reactive_vehicle_state.attack_lane_id,
                            reactive_vehicle_state.attack_lateral_offset,
                            reactive_vehicle_state.attack_distance,
                            reactive_vehicle_state.route_length,
                            reactive_vehicle_state.maximum_route_curvature,
                            reactive_vehicle_state.route_speed_limit,
                            reactive_vehicle_state.attack_yaw,
                            reactive_vehicle_state.target.source,
                        )
                continue
            spawn_result = spawn_static_vehicle(world, target, index, args)
            if spawn_result is None:
                continue
            vehicle = spawn_result.actor
            blocker_vehicles.append(vehicle)
            vehicle.set_simulate_physics(False)
            if (
                reactive_vehicle_state is not None
                and reactive_vehicle_state.index
                == DEFAULT_REACTIVE_VEHICLE_INDEX
                and (
                    reactive_vehicle_state.requested_target.source
                    == "built-in-capture"
                )
                and index == DEFAULT_ADDITIONAL_PATROL_INDEX
                and target.blueprint_id == DEFAULT_ADDITIONAL_PATROL_BLUEPRINT
            ):
                reactive_vehicle_state.home_clearance_ignored_actor_ids = (
                    int(vehicle.id),
                )
            LOG.info(
                "Static vehicle #%d id=%d type=%s state=BLOCKING "
                "target=(%.3f, %.3f, %.3f, yaw=%.2f) source=%s "
                "placement=%s",
                index,
                vehicle.id,
                vehicle.type_id,
                spawn_result.transform.location.x,
                spawn_result.transform.location.y,
                spawn_result.transform.location.z,
                spawn_result.transform.rotation.yaw,
                target.source,
                spawn_result.placement_source,
            )

        navigation = NavigationSampler(world, args.nav_samples)
        targets = resolve_pedestrian_targets(world, args, navigation)
        for index, target in enumerate(targets, 1):
            try:
                walker = spawn_pedestrian(
                    world,
                    carla_map,
                    target,
                    index,
                    args,
                    navigation,
                )
            except RuntimeError as exc:
                LOG.warning(
                    "Skipping pedestrian #%d target=(%.3f, %.3f, %.3f): %s",
                    index,
                    target.x,
                    target.y,
                    target.z + args.z_offset,
                    exc,
                )
                continue
            state = PedestrianState(
                index=index,
                actor=walker,
                target=target,
            )
            states.append(state)
            state.sensor = spawn_collision_sensor(world, walker, registry)
            LOG.info(
                "Pedestrian #%d id=%d state=WAITING target=(%.3f, %.3f, %.3f) "
                "source=%s",
                index,
                walker.id,
                target.x,
                target.y,
                target.z + args.z_offset,
                target.source,
            )

        if (
            not blocker_vehicles
            and reactive_vehicle_state is None
            and not states
        ):
            LOG.error("No blocker actors could be spawned")
            return 1
        if states:
            LOG.info(
                "Ready with %d static vehicle(s), reactive_vehicle=%s, and "
                "%d armed pedestrian(s); "
                "ego lookup role_name=%r actor_id=%s",
                len(blocker_vehicles),
                "enabled" if reactive_vehicle_state is not None else "disabled",
                len(states),
                args.ego_role_name,
                "auto" if args.ego_actor_id is None else args.ego_actor_id,
            )
            LOG.info(
                "Pedestrian resilience: stall_timeout=%.2f s "
                "min_progress=%.2f m scripted_step_cap=%.2f m/update "
                "scripted_budget=%.2f s max_command_failures=%d",
                args.motion_stall_timeout,
                args.motion_stall_min_progress,
                args.stall_recovery_step,
                args.max_scripted_recovery_time,
                args.max_motion_command_failures,
            )
            LOG.info(
                "Intercept model: acceleration-aware perpendicular line, "
                "impact_target=%s front_impact_margin=%.2f m "
                "pedestrian_speed_range=%.2f..%.2f m/s horizon=%.2f..%.2f s "
                "active_perpendicular_tolerance=%.1f deg hard_brake=%.2f m/s2 "
                "near_miss_gap=%.2f m hold=%.2f s debug=%s",
                args.impact_target,
                args.front_impact_margin,
                args.min_pedestrian_speed,
                args.pedestrian_speed,
                args.min_intercept_time,
                args.max_intercept_time,
                args.active_perpendicular_tolerance,
                args.hard_brake_deceleration,
                args.near_miss_distance,
                args.post_event_hold,
                args.intercept_debug,
            )
        else:
            LOG.info(
                "Ready with %d static vehicle blocker(s), reactive_vehicle=%s, "
                "and no blocker pedestrians",
                len(blocker_vehicles),
                "enabled" if reactive_vehicle_state is not None else "disabled",
            )
        if reactive_vehicle_state is not None:
            if args.reactive_vehicle_control == "constant-velocity":
                LOG.warning(
                    "constant-velocity reactive control overrides normal "
                    "longitudinal dynamics and is unsuitable for realistic "
                    "collision rendering; use the default ackermann mode"
                )
            LOG.info(
                "Reactive crosswalk model: ego_pedestrian role_name=%r "
                "actor_id=%s nominal_target=(%.3f, %.3f) trigger=%.2f m "
                "route_window=+/-%.2f m min_crossing_angle=%.1f deg "
                "best_effort=%s "
                "ETA=%.2f..%.2f s vehicle_speed=%.2f..%.2f m/s "
                "configured_max=%.2f m/s control=%s post_distance=%.2f m "
                "launch_margin=%.2f s precommit_hold=%.2f m/%.2f s "
                "impact_settle=%.2f s contact_hold=%.2f s",
                args.ego_pedestrian_role_name,
                (
                    "auto"
                    if args.ego_pedestrian_actor_id is None
                    else args.ego_pedestrian_actor_id
                ),
                reactive_vehicle_state.attack_x,
                reactive_vehicle_state.attack_y,
                args.reactive_trigger_distance,
                args.reactive_intercept_route_window,
                args.reactive_min_crossing_angle,
                args.reactive_best_effort_launch,
                args.reactive_min_pedestrian_eta,
                args.reactive_max_pedestrian_eta,
                args.reactive_vehicle_speed_min,
                reactive_vehicle_state.route_speed_limit,
                args.reactive_vehicle_speed_max,
                args.reactive_vehicle_control,
                args.reactive_post_distance,
                args.reactive_launch_margin,
                args.reactive_approach_hold_distance,
                args.reactive_approach_hold_timeout,
                args.reactive_impact_settle_time,
                args.reactive_contact_hold_time,
            )
        update_period = 1.0 / args.update_hz
        last_update_time = None
        last_ego_status = None
        last_ego_pedestrian_status = None
        last_tick_warning = 0.0

        while True:
            try:
                snapshot = world.wait_for_tick(args.tick_timeout)
            except RuntimeError as exc:
                now = time.monotonic()
                if now - last_tick_warning >= 5.0:
                    if settings.synchronous_mode:
                        LOG.warning(
                            "Waiting for the existing CARLA synchronous-clock "
                            "owner: %s",
                            exc,
                        )
                    else:
                        LOG.warning(
                            "Timed out waiting for the asynchronous CARLA "
                            "server's next snapshot: %s",
                            exc,
                        )
                    last_tick_warning = now
                continue
            simulation_time = snapshot_time(snapshot)
            if last_update_time is not None:
                update_delta = simulation_time - last_update_time
                if update_delta < -1.0e-6:
                    LOG.warning(
                        "CARLA elapsed time moved backward from %.3f to %.3f; "
                        "rebasing blocker timers",
                        last_update_time,
                        simulation_time,
                    )
                    rebase_state_timers_after_clock_rewind(
                        states,
                        last_update_time,
                        simulation_time,
                    )
                    rebase_reactive_vehicle_timers_after_clock_rewind(
                        reactive_vehicle_state,
                        last_update_time,
                        simulation_time,
                    )
                elif update_delta + 1.0e-9 < update_period:
                    continue
            last_update_time = simulation_time

            if reactive_vehicle_state is not None:
                ego_pedestrian, ego_pedestrian_status = find_ego_pedestrian(
                    world,
                    args.ego_pedestrian_role_name,
                    args.ego_pedestrian_actor_id,
                )
                if ego_pedestrian_status != last_ego_pedestrian_status:
                    LOG.info(
                        "Ego-pedestrian discovery: %s",
                        ego_pedestrian_status,
                    )
                    last_ego_pedestrian_status = ego_pedestrian_status
                if (
                    ego_pedestrian is None
                    and reactive_vehicle_state.active_target_id is not None
                ):
                    try:
                        latched_actor = world.get_actor(
                            int(reactive_vehicle_state.active_target_id)
                        )
                        if (
                            latched_actor is not None
                            and latched_actor.is_alive
                            and str(latched_actor.type_id).startswith(
                                "walker.pedestrian."
                            )
                        ):
                            ego_pedestrian = latched_actor
                    except (AttributeError, RuntimeError):
                        pass
                update_reactive_vehicle(
                    reactive_vehicle_state,
                    world,
                    ego_pedestrian,
                    simulation_time,
                    reactive_vehicle_registry,
                    args,
                )

            if not states:
                continue

            # Pending replacements are attempted before processing new events,
            # so every reset waits for at least one subsequent world snapshot.
            for state in states:
                attempt_pending_respawn(
                    state,
                    world,
                    carla_map,
                    navigation,
                    simulation_time,
                    registry,
                    args,
                )

            # Collision callbacks only enqueue IDs. Actor lifecycle changes stay
            # on this main loop and therefore never run on a sensor thread.
            try:
                frame_vehicles = world.get_actors().filter("vehicle.*")
            except RuntimeError:
                frame_vehicles = None
            for state in states:
                if state.state == STATE_RESPAWN_PENDING:
                    continue
                if not state_actor_alive(state):
                    request_respawn(
                        state,
                        simulation_time,
                        "walker actor was removed outside this client",
                        registry,
                        args,
                    )
                    continue
                vehicle_id = registry.consume_vehicle_hit(int(state.actor.id))
                contact_source = "collision sensor"
                if vehicle_id is None:
                    contact = nearby_vehicle_contact(
                        world,
                        state,
                        args.collision_distance,
                        allow_center_distance=state.sensor is None,
                        vehicles=frame_vehicles,
                    )
                    if contact is not None:
                        vehicle_id, contact_source = contact
                if vehicle_id is not None:
                    begin_post_event_hold(
                        state,
                        simulation_time,
                        "vehicle contact id={} detected by {}".format(
                            vehicle_id,
                            contact_source,
                        ),
                        args,
                    )

            ego, ego_status = find_ego_vehicle(
                world,
                args.ego_role_name,
                args.ego_actor_id,
            )
            if ego_status != last_ego_status:
                LOG.info("Ego discovery: %s", ego_status)
                last_ego_status = ego_status

            for state in states:
                if state.state == STATE_ACTIVE:
                    update_result = update_active_pedestrian(
                        state,
                        ego,
                        simulation_time,
                        args,
                        vehicles=frame_vehicles,
                        world=world,
                    )
                    if update_result is not None and update_result.action == "hold":
                        begin_post_event_hold(
                            state,
                            simulation_time,
                            update_result.reason,
                            args,
                        )
                    elif update_result is not None:
                        request_respawn(
                            state,
                            simulation_time,
                            update_result.reason,
                            registry,
                            args,
                        )

            for state in states:
                hold_completion = update_holding_pedestrian(
                    state,
                    simulation_time,
                )
                if hold_completion is not None:
                    request_respawn(
                        state,
                        simulation_time,
                        hold_completion,
                        registry,
                        args,
                        delay_override=0.0,
                    )

            if ego is None:
                continue

            active_count = sum(
                1
                for state in states
                if state.state in (STATE_ACTIVE, STATE_HOLDING)
            )
            available_slots = max(0, args.max_active_pedestrians - active_count)
            if available_slots <= 0:
                continue
            candidates = []
            for state in states:
                decision = candidate_decision(state, ego, args)
                if decision is not None:
                    candidates.append(
                        (decision.intercept.time_seconds, state, decision)
                    )
            candidates.sort(key=lambda item: item[0])
            activated_count = 0
            for _, state, decision in candidates:
                if activated_count >= available_slots:
                    break
                if activate_pedestrian(
                    state,
                    ego,
                    decision,
                    simulation_time,
                    args,
                ):
                    activated_count += 1

    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    except (RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1
    finally:
        # Invalidate the cross-client profile before any long actor teardown.
        # Its unlistened GNSS manifest is always destroyed before its zones.
        destroy_published_network_profile(network_profile_publication, world)
        network_profile_publication = None
        # Stop the moving attack vehicle before any potentially blocking
        # sensor/actor teardown so it cannot keep its last command on exit.
        if reactive_vehicle_state is not None:
            if (
                reactive_vehicle_state.actor is not None
                or reactive_vehicle_state.sensor is not None
            ):
                retire_reactive_vehicle_actors(
                    reactive_vehicle_state,
                    reactive_vehicle_registry,
                )
            if world is not None:
                retired_actor_is_absent(
                    world,
                    reactive_vehicle_state.retired_sensor_id,
                    "reactive vehicle collision sensor",
                )
                retired_actor_is_absent(
                    world,
                    reactive_vehicle_state.retired_actor_id,
                    "reactive vehicle blocker",
                )
        for state in reversed(states):
            if (
                state.actor is not None
                or state.sensor is not None
            ):
                retire_state_actors(state, registry)
        if world is not None:
            for state in reversed(states):
                retired_actor_is_absent(
                    world,
                    state.retired_sensor_id,
                    "collision sensor",
                )
            for state in reversed(states):
                retired_actor_is_absent(
                    world,
                    state.retired_actor_id,
                    "pedestrian blocker",
                )
        for vehicle in reversed(blocker_vehicles):
            destroy_actor(vehicle, "static vehicle blocker")
        blocker_vehicles.clear()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
