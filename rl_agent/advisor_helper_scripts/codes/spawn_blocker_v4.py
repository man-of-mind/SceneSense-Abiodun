#!/usr/bin/env python3

"""
Spawn static vehicle and reactive pedestrian blockers for a CARLA test scene.

This client is designed to run beside manual_control_ar_v5.py. It remains a
passive world client: it never loads a map, changes world settings, changes the
Traffic Manager, or calls world.tick(). The CARLA server or manual-control
client must advance the simulation.

With no blocker-location arguments, the script spawns the three captured static
bus blockers and two captured pedestrian blockers in Town10HD. Repeating
--vehicle-location or --pedestrian-location replaces that category's baked-in
locations and supports any number of blockers. --no-vehicle-blockers and
--no-pedestrian-blockers disable a category explicitly.

Pedestrians may also use a ground-projected location captured from CARLA's
free-floating spectator camera. If direct spawning fails while the requested
target remains clear, the script tries a nearby Sidewalk waypoint and then the
nearest sampled navigation locations before relocating the walker to the
requested transform. Occupied initial targets are skipped, and occupied
respawn targets are retried on later passive snapshots.
If the preferred pedestrian blueprint is unavailable, the script logs the
problem and selects the first available walker.pedestrian.* ID alphabetically.

The ego vehicle is discovered by role_name (default: hero, matching
manual_control_ar_v5.py). A waiting pedestrian activates only when the ego is
moving toward it and an acceleration-aware, perpendicular intercept lies inside
the configured reaction-plus-hard-braking stopping-distance envelope. The
constant-acceleration prediction uses the ego's current location, velocity, and
acceleration. It selects L2 where the line from the pedestrian to L2 is
perpendicular to the predicted ego tangent, then computes the pedestrian speed
needed for both actors to reach L2 at the same predicted time. The configured
``--pedestrian-speed`` is now the maximum allowed physical speed.

After activation the straight pedestrian crossing line stays fixed. Every
update uses fresh ego velocity and acceleration to predict the ego's next
intersection with that line and retimes the pedestrian without turning it into
a moving-target chase. If steering changes the predicted tangent beyond
``--active-perpendicular-tolerance``, the encounter is recycled instead of
claiming a synchronized perpendicular collision that is no longer feasible.
It is likewise recycled if fresh prediction requires more than the configured
maximum pedestrian speed or moves L2 behind the committed walker.
Short-lived CARLA debug lines show the predicted ego trajectory, perpendicular
pedestrian path, and current L2 by default; use
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

Version 4 retains v3's dynamic interception and recurring respawn model while
adapting the reusable resilience mechanisms from scenesense_scenario_harness.py:
activation yaw alignment, tight endpoint completion, repeated directional
commands, and a deterministic fallback after verified motion stalls. It also
adds vertically gated projected-bounding-box checks, occupied-target protection,
partial-command rollback, and CARLA clock-rewind recovery. This revision adds
acceleration-aware perpendicular interception, active ETA retiming, near-miss
classification, debug geometry, and a visible post-event hold state.

Examples:
    # Spawn the baked-in three buses and two pedestrians.
    python3 spawn_blocker_v4.py

    # Replace only the pedestrian defaults; keep the three default buses.
    python3 spawn_blocker_v4.py \\
        --pedestrian-blueprint walker.pedestrian.0015 \\
        --pedestrian-location 70.0 61.0 0.2 -90 \\
        --ego-role-name hero

    # Spawn only one spectator-derived pedestrian.
    python3 spawn_blocker_v4.py \\
        --no-vehicle-blockers --from-spectator \\
        --ego-role-name manual_ar_ego
"""

import argparse
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import carla


LOG = logging.getLogger("spawn_mixed_blockers")

DEFAULT_VEHICLE_BLUEPRINT = "vehicle.fuso.mitsubishi"
DEFAULT_PEDESTRIAN_BLUEPRINT = "walker.pedestrian.0015"
DEFAULT_EGO_ROLE_NAME = "hero"
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

DEFAULT_PEDESTRIAN_SPEED_MPS = 8.0
DEFAULT_MIN_PEDESTRIAN_SPEED_MPS = 0.5
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

# Ground targets captured from the Town10HD_Opt spectator and recorded in
# blocker_locations_v1.json. Category-specific Z offsets are applied once when
# CARLA transforms are constructed.
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
        -13.132160186767578,
        28.438270568847656,
        -0.13800303637981415,
        -3.5454695224761963,
    ),
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

VEHICLE_ROLE_PREFIX = "static_blocker_v4"
PEDESTRIAN_ROLE_PREFIX = "pedestrian_blocker_v4"
STATE_WAITING = "WAITING"
STATE_ACTIVE = "ACTIVE"
STATE_HOLDING = "HOLDING"
STATE_RESPAWN_PENDING = "RESPAWN_PENDING"


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
        help="exact vehicle.* blueprint (default: %(default)s)",
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
        help="manual_control_ar_v5.py --rolename value (default: %(default)s)",
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
    if not args.host:
        parser.error("--host must not be empty")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
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


def planar_distance(first: carla.Location, second: carla.Location) -> float:
    return math.hypot(float(first.x - second.x), float(first.y - second.y))


def normalized_xy(x_coord: float, y_coord: float) -> Optional[Tuple[float, float]]:
    magnitude = math.hypot(float(x_coord), float(y_coord))
    if magnitude <= 1e-9:
        return None
    return float(x_coord) / magnitude, float(y_coord) / magnitude


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
) -> Optional[InterceptSolution]:
    """Find the first acceleration-aware perpendicular XY intercept.

    With ``X(t) = E + V*t + 0.5*A*t^2``, a line from pedestrian P
    to X(t) is perpendicular to the predicted ego tangent when
    ``(X(t) - P) dot (V + A*t) == 0``. The root is found by bounded
    scanning and bisection so the implementation has no NumPy dependency.
    """
    velocity_x = float(ego_velocity.x)
    velocity_y = float(ego_velocity.y)
    initial_speed = math.hypot(velocity_x, velocity_y)
    initial_direction = normalized_xy(velocity_x, velocity_y)
    if initial_direction is None or initial_speed <= 1.0e-6:
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
        return (
            (predicted_x - float(pedestrian_location.x)) * tangent_x
            + (predicted_y - float(pedestrian_location.y)) * tangent_y
        )

    # A negative derivative of squared distance means the predicted ego is
    # approaching the pedestrian's perpendicular projection.
    previous_time = 0.0
    previous_value = difference_x * velocity_x + difference_y * velocity_y
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

            target_x, target_y = predicted_ego_xy(
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
) -> Optional[LineInterceptUpdate]:
    """Predict the ego's next intersection with a locked pedestrian line."""
    normal_x, normal_y = ego_path_direction
    constant = (
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
    if abs(quadratic_acceleration) <= epsilon:
        if linear <= epsilon:
            return None
        roots.append(-constant / linear)
    else:
        discriminant = linear * linear - 2.0 * quadratic_acceleration * constant
        if discriminant < 0.0:
            return None
        square_root = math.sqrt(max(0.0, discriminant))
        roots.extend(
            (
                (-linear - square_root) / quadratic_acceleration,
                (-linear + square_root) / quadratic_acceleration,
            )
        )
    for intercept_time in sorted(
        root for root in roots if 1.0e-4 < root <= float(maximum_time)
    ):
        predicted_normal_speed = linear + quadratic_acceleration * intercept_time
        if predicted_normal_speed <= 0.05:
            continue
        target_x, target_y = predicted_ego_xy(
            ego_location,
            ego_velocity,
            acceleration_xy,
            intercept_time,
        )
        # Remove round-off along the line normal. The remaining point lies on
        # the one straight crossing line locked at activation.
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
        tangent_direction = normalized_xy(
            float(ego_velocity.x) + acceleration_xy[0] * intercept_time,
            float(ego_velocity.y) + acceleration_xy[1] * intercept_time,
        )
        if tangent_direction is None:
            continue
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
    )
    if intercept is None:
        return None

    ego_travel = intercept.ego_travel
    effective_travel = max(0.0, ego_travel - max(0.0, collision_clearance))
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
    return [
        VehicleTarget(
            x=float(values[0]),
            y=float(values[1]),
            z=float(values[2]),
            yaw=float(values[3]),
            source=source,
        )
        for values in locations
    ]


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
):
    role_name = "{}_{}".format(VEHICLE_ROLE_PREFIX, index)
    blueprint = find_vehicle_blueprint(
        world,
        args.vehicle_blueprint,
        role_name,
    )
    transform = vehicle_target_transform(target, args.vehicle_z_offset)
    vehicle = try_spawn_actor(world, blueprint, transform)
    if vehicle is None:
        LOG.warning(
            "Static vehicle #%d failed to spawn at "
            "x=%.3f y=%.3f z=%.3f yaw=%.2f; location may be occupied",
            index,
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.yaw,
        )
        return None
    return vehicle


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
        previous = carla.Location(
            x=float(ego_location.x),
            y=float(ego_location.y),
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
    signed_distance = (
        (target_x - float(ego_location.x)) * path_x
        + (target_y - float(ego_location.y)) * path_y
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
        elif state.commanded_pedestrian_speed is None:
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


def reset_activation_fields(state: PedestrianState) -> None:
    state.active_since = None
    state.active_ego_id = None
    state.last_separation = None
    state.crossing_origin = None
    state.crossing_direction = None
    state.crossing_endpoint = None
    state.crossing_distance = None
    state.ego_path_direction = None
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
        # The original subscribed sensor wrapper was stopped during teardown.
        # A wrapper rediscovered by ID was never subscribed by this process, so
        # calling stop() on it can emit a misleading CARLA warning.
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    world = None
    blocker_vehicles: List[object] = []
    states: List[PedestrianState] = []
    registry = CollisionRegistry()
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
        carla_map = world.get_map()
        settings = world.get_settings()
        LOG.info(
            "Connected to %s at %s:%d synchronous_mode=%s",
            carla_map.name,
            args.host,
            args.port,
            settings.synchronous_mode,
        )
        LOG.info(
            "Passive client: manual_control_ar_v5.py/server must advance ticks; "
            "this script never calls world.tick() or changes settings"
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

        vehicle_targets = resolve_vehicle_targets(args)
        for index, target in enumerate(vehicle_targets, 1):
            vehicle = spawn_static_vehicle(world, target, index, args)
            if vehicle is None:
                continue
            blocker_vehicles.append(vehicle)
            vehicle.set_simulate_physics(False)
            LOG.info(
                "Static vehicle #%d id=%d type=%s state=BLOCKING "
                "target=(%.3f, %.3f, %.3f, yaw=%.2f) source=%s",
                index,
                vehicle.id,
                vehicle.type_id,
                target.x,
                target.y,
                target.z + args.vehicle_z_offset,
                target.yaw,
                target.source,
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
            state = PedestrianState(index=index, actor=walker, target=target)
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

        if not blocker_vehicles and not states:
            LOG.error("No blocker actors could be spawned")
            return 1
        if states:
            LOG.info(
                "Ready with %d static vehicle(s) and %d armed pedestrian(s); "
                "ego lookup role_name=%r actor_id=%s",
                len(blocker_vehicles),
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
                "pedestrian_speed_range=%.2f..%.2f m/s horizon=%.2f..%.2f s "
                "active_perpendicular_tolerance=%.1f deg hard_brake=%.2f m/s2 "
                "near_miss_gap=%.2f m hold=%.2f s debug=%s",
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
                "Ready with %d static vehicle blocker(s) and no pedestrians",
                len(blocker_vehicles),
            )
        update_period = 1.0 / args.update_hz
        last_update_time = None
        last_ego_status = None
        last_tick_warning = 0.0

        while True:
            try:
                snapshot = world.wait_for_tick(args.tick_timeout)
            except RuntimeError as exc:
                now = time.monotonic()
                if now - last_tick_warning >= 5.0:
                    LOG.warning(
                        "Waiting for the CARLA clock master: %s",
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
                        "rebasing pedestrian timers",
                        last_update_time,
                        simulation_time,
                    )
                    rebase_state_timers_after_clock_rewind(
                        states,
                        last_update_time,
                        simulation_time,
                    )
                elif update_delta + 1.0e-9 < update_period:
                    continue
            last_update_time = simulation_time

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
        for state in reversed(states):
            walker_id = getattr(state.actor, "id", None)
            if walker_id is not None:
                registry.unregister(int(walker_id))
            sensor = state.sensor
            if sensor is None:
                continue
            try:
                if sensor.is_alive:
                    sensor.stop()
            except RuntimeError:
                pass
            destroy_actor(sensor, "collision sensor")
            state.sensor = None
        if world is not None:
            for state in reversed(states):
                retired_actor_is_absent(
                    world,
                    state.retired_sensor_id,
                    "collision sensor",
                )
        for state in reversed(states):
            stop_walker(state.actor)
            destroy_actor(state.actor, "pedestrian blocker")
            state.actor = None
        if world is not None:
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
