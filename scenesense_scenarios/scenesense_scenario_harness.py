#!/usr/bin/env python3
"""Repeatable CARLA scenario harness for SceneSense.

Step 1 only: spawn and document controlled scenes. No model inference,
training, sensors, or RL logic lives here yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def _bootstrap_carla():
    try:
        import carla as imported_carla

        return imported_carla
    except ModuleNotFoundError:
        pass

    script_path = Path(__file__).resolve()
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    search_roots: List[Path] = []
    for depth in (7, 6, 5, 4, 3, 2, 1, 0):
        if len(script_path.parents) > depth:
            search_roots.append(script_path.parents[depth])

    for root in search_roots:
        for site_packages in root.glob(f"**/lib/{py_version}/site-packages"):
            if not list(site_packages.glob("carla*.so")):
                continue
            sys.path.insert(0, str(site_packages))
            try:
                import carla as imported_carla

                return imported_carla
            except ModuleNotFoundError:
                sys.path.pop(0)

    raise ModuleNotFoundError(
        "Unable to import CARLA. Run inside the CARLA Python environment or "
        "add CARLA's PythonAPI site-packages directory to PYTHONPATH."
    )


carla = None

ABIODUN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ABIODUN_DIR / "metrics_logs" / "scenesense_scenarios"
TRAFFIC_LIGHTS_JSON = ABIODUN_DIR / "traffic_lights_data.json"
SCENESENSE_ROLE_PREFIX = "scenesense_"

SAFE_VEHICLE_BLUEPRINTS = (
    "vehicle.lincoln.mkz",
    "vehicle.lincoln.mkz_2020",
    "vehicle.lincoln.mkz_2017",
    "vehicle.mercedes.coupe_2020",
    "vehicle.dodge.charger_2020",
    "vehicle.audi.a2",
    "vehicle.toyota.prius",
    "vehicle.nissan.micra",
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    default_town: str
    traffic_light_id: str
    anchor_radius_m: float
    background_vehicles: int
    pedestrians: int
    ego_distance_m: float
    occlusion_pair: bool = False
    manual_occlusion_crossing: bool = False
    intersection_truck_occlusion: bool = False
    curbside_occlusion: bool = False
    intersection_occlusion_mode: str = "standard"
    spectator_height_m: float = 32.0


SCENARIOS: Dict[str, ScenarioSpec] = {
    "clear_low_density": ScenarioSpec(
        name="clear_low_density",
        description="Low-density baseline with clear line of sight around traffic light 14.",
        default_town="Town10HD_Opt",
        traffic_light_id="14",
        anchor_radius_m=55.0,
        background_vehicles=4,
        pedestrians=4,
        ego_distance_m=22.0,
    ),
    "crowded_intersection": ScenarioSpec(
        name="crowded_intersection",
        description="Higher-density intersection scene around the same anchor.",
        default_town="Town10HD_Opt",
        traffic_light_id="14",
        anchor_radius_m=70.0,
        background_vehicles=22,
        pedestrians=34,
        ego_distance_m=24.0,
    ),
    "occlusion_static": ScenarioSpec(
        name="occlusion_static",
        description="Static occlusion candidate: occluder closer to anchor, target farther along similar bearing.",
        default_town="Town10HD_Opt",
        traffic_light_id="14",
        anchor_radius_m=65.0,
        background_vehicles=10,
        pedestrians=16,
        ego_distance_m=22.0,
        occlusion_pair=True,
    ),
    "occlusion_crossing_ego": ScenarioSpec(
        name="occlusion_crossing_ego",
        description="Ego-facing blind-spot setup: parked occluder hides a target pedestrian near the ego route.",
        default_town="Town10HD_Opt",
        traffic_light_id="14",
        anchor_radius_m=80.0,
        background_vehicles=0,
        pedestrians=0,
        ego_distance_m=42.0,
        manual_occlusion_crossing=True,
        spectator_height_m=26.0,
    ),
    "intersection_truck_pedestrian_occlusion": ScenarioSpec(
        name="intersection_truck_pedestrian_occlusion",
        description="Intersection occlusion: parked truck/van hides a crossing pedestrian from the ego view.",
        default_town="Town10HD_Opt",
        traffic_light_id="11",
        anchor_radius_m=90.0,
        background_vehicles=0,
        pedestrians=0,
        ego_distance_m=42.0,
        intersection_truck_occlusion=True,
        intersection_occlusion_mode="occluded_failure",
        spectator_height_m=30.0,
    ),
    "right_turn_truck_pedestrian_occlusion": ScenarioSpec(
        name="right_turn_truck_pedestrian_occlusion",
        description="Right-turn yield failure: stopped truck/van hides a pedestrian near the crosswalk.",
        default_town="Town10HD_Opt",
        traffic_light_id="11",
        anchor_radius_m=90.0,
        background_vehicles=0,
        pedestrians=0,
        ego_distance_m=42.0,
        intersection_truck_occlusion=True,
        intersection_occlusion_mode="right_turn_occluded_failure",
        spectator_height_m=30.0,
    ),
    "visible_crossing_failure": ScenarioSpec(
        name="visible_crossing_failure",
        description="Control failure: ego drives into a visible crossing pedestrian with the same route/timing.",
        default_town="Town10HD_Opt",
        traffic_light_id="11",
        anchor_radius_m=90.0,
        background_vehicles=0,
        pedestrians=0,
        ego_distance_m=42.0,
        intersection_truck_occlusion=True,
        intersection_occlusion_mode="visible_control",
        spectator_height_m=30.0,
    ),
    "curbside_parked_vehicle_pedestrian_occlusion": ScenarioSpec(
        name="curbside_parked_vehicle_pedestrian_occlusion",
        description="Mid-block blind spot: a pedestrian emerges from behind parked curbside vehicles into the ego path.",
        default_town="Town10HD_Opt",
        traffic_light_id="14",
        anchor_radius_m=95.0,
        background_vehicles=0,
        pedestrians=0,
        ego_distance_m=42.0,
        curbside_occlusion=True,
        spectator_height_m=26.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spawn repeatable SceneSense CARLA scenarios and save metadata."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument(
        "--scenario",
        default="clear_low_density",
        choices=sorted(SCENARIOS),
        help="Scenario name.",
    )
    parser.add_argument("--list", action="store_true", help="List scenarios and exit.")
    parser.add_argument(
        "--list-vehicles",
        action="store_true",
        help=(
            "Connect to CARLA, print all vehicle.* blueprint IDs in the loaded world "
            "(grouped into trucks/vans/large vs cars/sedans), then exit. Use this to "
            "discover which trucks are actually available in CARLA 0.10 before passing "
            "them to --curbside-occluder-blueprint."
        ),
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic spawn seed.")
    parser.add_argument(
        "--town",
        default="",
        help="Town to load when --load-town is set. Defaults to the scenario town.",
    )
    parser.add_argument(
        "--load-town",
        action="store_true",
        help="Load the requested/scenario town before spawning. Otherwise attach to current world.",
    )
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port.")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="How long to hold/tick the scene. Use 0 to hold until Ctrl+C.",
    )
    parser.add_argument(
        "--async-world",
        action="store_true",
        help="Do not switch CARLA into synchronous fixed-step mode.",
    )
    parser.add_argument(
        "--fixed-delta-s",
        type=float,
        default=0.05,
        help="Fixed delta seconds when synchronous mode is enabled.",
    )
    parser.add_argument(
        "--traffic-light-id",
        default="",
        help="Override scenario anchor traffic-light id.",
    )
    parser.add_argument(
        "--anchor-source",
        choices=("traffic_light", "spawn_point"),
        default="traffic_light",
        help="Use a traffic-light anchor or a map spawn-point anchor.",
    )
    parser.add_argument(
        "--anchor-spawn-index",
        type=int,
        default=0,
        help="Spawn-point index to use when --anchor-source spawn_point is set.",
    )
    parser.add_argument(
        "--ego-spawn-index",
        type=int,
        default=-1,
        help="Force a specific ego spawn-point index for route-based special scenarios.",
    )
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=-1,
        help="Override background vehicle count.",
    )
    parser.add_argument(
        "--pedestrian-count",
        type=int,
        default=-1,
        help="Override pedestrian count.",
    )
    parser.add_argument(
        "--move-pedestrians",
        action="store_true",
        help="Attach walker controllers and move pedestrians toward random navigation targets.",
    )
    parser.add_argument(
        "--background-autopilot",
        action="store_true",
        help="Enable Traffic Manager autopilot for background vehicles.",
    )
    parser.add_argument(
        "--ego-autopilot",
        action="store_true",
        help="Enable Traffic Manager autopilot for the ego vehicle.",
    )
    parser.add_argument(
        "--ego-sensors",
        action="store_true",
        help="Attach front RGB camera and radar sensors to the ego vehicle.",
    )
    parser.add_argument(
        "--ego-camera-preview",
        action="store_true",
        help="Show the ego front RGB camera in an OpenCV window when --ego-sensors is enabled.",
    )
    parser.add_argument("--ego-camera-width", type=int, default=768, help="Ego RGB camera width.")
    parser.add_argument("--ego-camera-height", type=int, default=432, help="Ego RGB camera height.")
    parser.add_argument("--ego-camera-fov", type=float, default=100.0, help="Ego RGB camera FoV.")
    parser.add_argument("--ego-radar-range", type=float, default=80.0, help="Ego radar range in meters.")
    parser.add_argument("--ego-radar-hfov", type=float, default=80.0, help="Ego radar horizontal FoV.")
    parser.add_argument("--ego-radar-vfov", type=float, default=20.0, help="Ego radar vertical FoV.")
    parser.add_argument(
        "--ego-radar-pps",
        type=int,
        default=1500,
        help="Ego radar points per second.",
    )
    parser.add_argument(
        "--scripted-ego-drive",
        action="store_true",
        help="Apply scripted ego control for occlusion demos.",
    )
    parser.add_argument(
        "--ego-drive-mode",
        choices=("waypoint", "straight"),
        default="waypoint",
        help="Scripted ego drive mode. Waypoint mode follows CARLA lane waypoints.",
    )
    parser.add_argument(
        "--ego-route-choice",
        choices=("straight", "left", "right"),
        default="left",
        help="Preferred waypoint branch for the occlusion-crossing ego route.",
    )
    parser.add_argument(
        "--ego-drive-throttle",
        type=float,
        default=0.28,
        help="Throttle used by --scripted-ego-drive.",
    )
    parser.add_argument(
        "--ego-target-speed",
        type=float,
        default=5.0,
        help="Target ego speed in m/s for waypoint scripted driving.",
    )
    parser.add_argument(
        "--ego-route-lookahead",
        type=float,
        default=7.0,
        help="Waypoint lookahead distance in meters for scripted ego steering.",
    )
    parser.add_argument(
        "--target-crossing",
        action="store_true",
        help="Move the occlusion target pedestrian across the ego route after a delay.",
    )
    parser.add_argument(
        "--target-crossing-delay-s",
        type=float,
        default=2.0,
        help="Delay before the target pedestrian starts crossing.",
    )
    parser.add_argument(
        "--target-crossing-speed",
        type=float,
        default=1.8,
        help="Pedestrian crossing speed in m/s for --target-crossing.",
    )
    parser.add_argument(
        "--target-prewalk",
        action="store_true",
        help="Move the target pedestrian toward the hidden crossing start before the crossing trigger fires.",
    )
    parser.add_argument(
        "--target-prewalk-speed",
        type=float,
        default=1.2,
        help="Pedestrian speed in m/s during the prewalk phase.",
    )
    parser.add_argument(
        "--target-prewalk-mode",
        choices=("animated", "deterministic"),
        default="animated",
        help="Use normal walker animation for prewalk or deterministic transform interpolation.",
    )
    parser.add_argument(
        "--target-crossing-control-speed",
        type=float,
        default=-1.0,
        help=(
            "Override low-level WalkerControl speed for target crossings. "
            "Use a negative value to keep the scenario default."
        ),
    )
    parser.add_argument(
        "--target-crossing-motion-mode",
        choices=("", "walker_control", "deterministic", "exact_transform", "scripted_transform", "ai_controller"),
        default="",
        help=(
            "Override how the target pedestrian crosses. Empty keeps the scenario default. "
            "walker_control uses CARLA WalkerControl animation, deterministic uses target "
            "velocity plus WalkerControl for an animated dart-out, scripted_transform hard-sets "
            "the target along the planned crossing path at the requested speed for collision "
            "timing, exact_transform is a debugging alias for scripted_transform, and "
            "ai_controller uses CARLA's walker AI controller on the navigation mesh. "
            "ai_controller can choose sidewalk routes and is not the default for forced "
            "road-crossing failures."
        ),
    )
    parser.add_argument(
        "--target-crossing-trigger-distance-m",
        type=float,
        default=18.0,
        help=(
            "When the layout has a conflict point, start the target crossing once the ego is "
            "within this distance after --target-crossing-delay-s. Use 0 to use delay only. "
            "Ignored when --target-crossing-trigger-ttc-s is positive."
        ),
    )
    parser.add_argument(
        "--target-crossing-trigger-ttc-s",
        type=float,
        default=0.0,
        help=(
            "If positive, fire the crossing when the ego's time-to-conflict-point (ego "
            "distance / current ego speed) drops below this many seconds. This self-adjusts "
            "to ego speed and pedestrian walk distance so the ego arrives roughly as the "
            "pedestrian reaches the lane. Recommended starting value: pedestrian walk "
            "distance / crossing speed minus a small safety margin (e.g. 0.4 s). When "
            "positive, overrides --target-crossing-trigger-distance-m."
        ),
    )
    parser.add_argument(
        "--target-crossing-trigger-route-lead-m",
        type=float,
        default=0.0,
        help=(
            "If positive and the scenario has a preplanned ego route, start the pedestrian "
            "when the ego reaches this many route meters before the target crossing trigger "
            "point. This is a tick/location-style calibrated trigger: use it with "
            "walker_control to keep natural pedestrian animation while tuning collision "
            "timing. When positive and route projection is available, overrides TTC and "
            "Euclidean distance triggers."
        ),
    )
    parser.add_argument(
        "--target-crossing-trigger-min-ego-speed-mps",
        type=float,
        default=0.0,
        help=(
            "If positive, do not start the target crossing until the ego is moving "
            "at least this fast. This creates a rolling-start trigger for cases "
            "where route-lead matching fires while the ego is still accelerating "
            "from spawn."
        ),
    )
    parser.add_argument(
        "--curbside-conflict-distance-m",
        type=float,
        default=31.0,
        help="Curbside scenario distance from ego spawn to the pedestrian conflict point.",
    )
    parser.add_argument(
        "--curbside-occluder-lateral-offset-m",
        type=float,
        default=3.2,
        help="Curbside scenario lateral offset for the parked occluder queue.",
    )
    parser.add_argument(
        "--curbside-target-start-lateral-offset-m",
        type=float,
        default=3.8,
        help="Curbside scenario pedestrian start lateral offset behind the occluder.",
    )
    parser.add_argument(
        "--curbside-target-end-lateral-offset-m",
        type=float,
        default=-0.6,
        help="Curbside scenario pedestrian crossing endpoint lateral offset.",
    )
    parser.add_argument(
        "--curbside-target-forward-offset-m",
        type=float,
        default=1.2,
        help="Curbside scenario pedestrian start forward offset from the conflict point.",
    )
    parser.add_argument(
        "--curbside-ego-start-forward-m",
        type=float,
        default=0.0,
        help=(
            "Curbside scenario: spawn the ego this many metres forward along the "
            "preplanned route while keeping the occluder/pedestrian layout fixed. "
            "Use small values such as 3-6 m to reduce the ego reaction window "
            "without retuning pedestrian/van placement."
        ),
    )
    parser.add_argument(
        "--curbside-target-prewalk-distance-m",
        type=float,
        default=0.0,
        help="Curbside scenario sidewalk/pre-entry walk distance before the crossing start.",
    )
    parser.add_argument(
        "--curbside-target-prewalk-lateral-offset-m",
        type=float,
        default=-1.0,
        help=(
            "Curbside scenario lateral offset for the prewalk start. "
            "Use a negative value to reuse --curbside-target-start-lateral-offset-m."
        ),
    )
    parser.add_argument(
        "--curbside-heavy-occluder-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer truck/bus/large-van blueprints for the curbside occluder before car fallbacks.",
    )
    parser.add_argument(
        "--helper-vehicle",
        action="store_true",
        help="Spawn an opposite-lane helper/observer vehicle for curbside occlusion evidence.",
    )
    parser.add_argument(
        "--helper-camera-preview",
        action="store_true",
        help="Show an RGB camera preview from the optional helper vehicle.",
    )
    parser.add_argument(
        "--helper-drive",
        action="store_true",
        help="Drive the optional helper vehicle slowly through its own opposite-lane path instead of keeping it static.",
    )
    parser.add_argument(
        "--helper-target-speed",
        type=float,
        default=3.0,
        help="Target helper vehicle speed in m/s when --helper-drive is set.",
    )
    parser.add_argument(
        "--helper-stop-distance-to-conflict-m",
        type=float,
        default=5.0,
        help=(
            "Stop the helper vehicle this many meters before its own pass-through target. "
            "The helper target is beyond the conflict point in the opposite lane. "
            "Set this low (e.g. 1.0) when you want the helper to drive all the way through "
            "and never stop within the camera window."
        ),
    )
    parser.add_argument(
        "--helper-pause-until-crossing",
        action="store_true",
        help=(
            "When set, the helper holds the brake at its spawn position and only starts "
            "driving the moment the pedestrian crossing actually fires. Looks unnatural "
            "(car visibly parked on the road) — prefer using --curbside-helper-spawn-forward-m "
            "to push the helper far enough away that it arrives at the conflict zone "
            "naturally, without pausing."
        ),
    )
    parser.add_argument(
        "--curbside-helper-spawn-forward-m",
        type=float,
        default=35.0,
        help=(
            "Curbside scenario: helper spawn distance past the conflict point on the "
            "opposite lane. Calibrate with --helper-target-speed so that "
            "spawn_distance / speed equals the time you want the helper to arrive at "
            "the conflict point. E.g. at speed 4.5 m/s and spawn 35 m, helper arrives "
            "at t = 7.8 s. Pair with --target-crossing-delay-s so the walker starts "
            "around that same moment, putting the walker in the helper's view."
        ),
    )
    parser.add_argument(
        "--curbside-helper-target-forward-m",
        type=float,
        default=-30.0,
        help=(
            "Curbside scenario: helper's drive-through target distance past the conflict "
            "(negative = beyond the conflict, on the ego-spawn side). Together with "
            "--helper-stop-distance-to-conflict-m this sets where the helper finally "
            "brakes. Make it sufficiently negative that the helper never brakes within "
            "the crossing window."
        ),
    )
    parser.add_argument(
        "--curbside-occluder-z-offset-m",
        type=float,
        default=0.1,
        help=(
            "Curbside scenario: vertical lift (in metres) applied to spawned occluders "
            "above the road surface. The default 0.1 keeps the vehicle just above the "
            "ground to avoid spawn-collision with road mesh. Set higher (e.g. 0.3) only "
            "if the vehicle clips into the road; set lower if it appears to float."
        ),
    )
    parser.add_argument(
        "--curbside-occluder-yaw-offset-deg",
        type=float,
        default=0.0,
        help=(
            "Curbside scenario: additional yaw rotation (in degrees) applied to our "
            "spawned occluder vehicles relative to the ego lane direction. Use this when "
            "the baked map vehicles at the curbside are parked at a different angle "
            "than the lane heading (common at angled-parking spots) and our truck looks "
            "rotated relative to them. Try ±5 to ±15 deg to align visually."
        ),
    )
    parser.add_argument(
        "--curbside-occluder-blueprint",
        type=str,
        default="",
        help=(
            "Curbside scenario: force a specific CARLA vehicle blueprint id for OUR "
            "spawned occluders (slot 0, 1, 2, and extra van). Empty (default) means "
            "use the heavy-vehicle pool. Examples: "
            "'vehicle.carlamotors.carlacola' (box truck), "
            "'vehicle.carlamotors.firetruck' (fire truck), "
            "'vehicle.tesla.cybertruck' (pickup), "
            "'vehicle.sprinter.mercedes' (sprinter van). Useful when you want a "
            "specific truck visually distinct from the baked map van."
        ),
    )
    parser.add_argument(
        "--curbside-occluder-count",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help=(
            "Curbside scenario: number of parked vehicles to spawn (1, 2, or 3). "
            "Use 1 when the map already has baked-in vehicles at the curbside and "
            "you just want to add one of our trucks/vans to flank the pedestrian. "
            "When 1, only the middle slot (the designated truck) spawns at "
            "--curbside-slot-1-forward-m."
        ),
    )
    parser.add_argument(
        "--curbside-slot-1-forward-m",
        type=float,
        default=-1.0,
        help=(
            "Curbside scenario: forward offset for the middle parked vehicle (slot 1, "
            "the designated truck/van). Default -1.0 sits it just before the conflict "
            "point. Adjust to move the truck relative to baked map vehicles when using "
            "--curbside-occluder-count 1."
        ),
    )
    parser.add_argument(
        "--curbside-slot-0-forward-m",
        type=float,
        default=-5.5,
        help=(
            "Curbside scenario: forward offset for the front parked vehicle (slot 0). "
            "Default -5.5 puts it about 5 m behind slot 1 (the truck). Push it further "
            "back (e.g. -12 or -15) to open up parking space between slot 0 and slot 1 "
            "for the extra-van flag — useful when you want the extra van placed BEHIND "
            "the truck (closer to ego's spawn) rather than ahead of it."
        ),
    )
    parser.add_argument(
        "--curbside-extra-occluder-forward-m",
        type=float,
        default=0.0,
        help=(
            "Curbside scenario: when nonzero, spawn an EXTRA parked van at this forward "
            "offset (in route metres relative to the occluder reference point, which "
            "sits 2 m before the conflict). Sign convention follows the ROUTE direction "
            "(positive = further along the route, in ego's direction of travel; "
            "negative = behind, toward ego's spawn). The three default slots sit at "
            "(-5.5, -1.0, +4.5) so the immediate gaps that have parking room are at "
            "approximately +8 to +12 (past slot 2) and -8 to -12 (behind slot 0). "
            "Values between -5 and +4 will usually fail to spawn because they overlap "
            "the default occluders."
        ),
    )
    parser.add_argument("--helper-camera-width", type=int, default=768, help="Helper RGB camera width.")
    parser.add_argument("--helper-camera-height", type=int, default=432, help="Helper RGB camera height.")
    parser.add_argument("--helper-camera-fov", type=float, default=100.0, help="Helper RGB camera FoV.")
    parser.add_argument(
        "--evidence-pack",
        action="store_true",
        help=(
            "Write an evidence folder with actor ground-truth traces and buffered ego/helper RGB frames. "
            "Use this for canonical failure/safety-baseline runs."
        ),
    )
    parser.add_argument(
        "--evidence-window-s",
        type=float,
        default=3.0,
        help="Seconds before/after the first target collision to extract into evidence window CSVs.",
    )
    parser.add_argument(
        "--evidence-camera-buffer-size",
        type=int,
        default=80,
        help="Number of sampled RGB frames to keep per evidence camera.",
    )
    parser.add_argument(
        "--evidence-camera-stride",
        type=int,
        default=2,
        help="Store every Nth RGB frame in the evidence camera buffer.",
    )
    parser.add_argument(
        "--stop-on-target-collision",
        action="store_true",
        help="End the scenario once the ego collision sensor reports a collision with the target actor.",
    )
    parser.add_argument(
        "--post-target-collision-hold-s",
        type=float,
        default=0.0,
        help="When stopping on target collision, keep the scene active for this many extra seconds first.",
    )
    parser.add_argument(
        "--keep-actors",
        action="store_true",
        help="Leave spawned actors in the world on exit.",
    )
    parser.add_argument(
        "--set-spectator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move the CARLA spectator above the scenario anchor.",
    )
    parser.add_argument(
        "--spectator-focus",
        choices=("anchor", "conflict"),
        default="conflict",
        help="Where to point the CARLA spectator when an occlusion conflict point exists.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root for scenario metadata outputs.",
    )
    return parser.parse_args()


def sanitize_token(value: object, default: str = "run") -> str:
    token = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in str(value or "").strip()
    ).strip("_")
    return token or default


def git_status_note() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(ABIODUN_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3.0,
            check=False,
        )
    except Exception as exc:
        return f"git_status_unavailable: {exc}"
    if result.returncode != 0:
        return "not_a_git_repository"
    return result.stdout.strip() or "clean"


def load_static_traffic_light_location(traffic_light_id: str) -> Optional["carla.Location"]:
    if not TRAFFIC_LIGHTS_JSON.exists():
        return None
    try:
        rows = json.loads(TRAFFIC_LIGHTS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for row in rows:
        if str(row.get("id")) != str(traffic_light_id):
            continue
        loc = row.get("location") or {}
        return carla.Location(
            x=float(loc.get("x", 0.0)),
            y=float(loc.get("y", 0.0)),
            z=float(loc.get("z", 0.0)),
        )
    return None


def resolve_spawn_point_anchor(
    world: "carla.World",
    spawn_index: int,
    source: str,
) -> Tuple["carla.Location", Dict[str, object]]:
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available for spawn-point anchor.")
    index = int(spawn_index) % len(spawn_points)
    transform = spawn_points[index]
    loc = transform.location
    return loc, {
        "source": source,
        "traffic_light_id": f"spawn_{index}",
        "spawn_index": int(index),
        "actor_id": None,
        "location": location_to_dict(loc),
        "transform": transform_to_dict(transform),
    }


def resolve_anchor(
    world: "carla.World",
    traffic_light_id: str,
    anchor_source: str = "traffic_light",
    anchor_spawn_index: int = 0,
) -> Tuple["carla.Location", Dict[str, object]]:
    if anchor_source == "spawn_point":
        return resolve_spawn_point_anchor(world, anchor_spawn_index, "spawn_point_anchor")

    live_lights = list(world.get_actors().filter("traffic.traffic_light"))
    for actor in live_lights:
        if str(actor.id) == str(traffic_light_id):
            loc = actor.get_transform().location
            return loc, {
                "source": "live_traffic_light_actor",
                "traffic_light_id": str(traffic_light_id),
                "actor_id": int(actor.id),
                "location": location_to_dict(loc),
            }

    static_loc = load_static_traffic_light_location(traffic_light_id)
    if static_loc is not None:
        return static_loc, {
            "source": "traffic_lights_data.json",
            "traffic_light_id": str(traffic_light_id),
            "actor_id": None,
            "location": location_to_dict(static_loc),
        }

    loc, info = resolve_spawn_point_anchor(world, 0, "spawn_point_fallback")
    info["traffic_light_id"] = str(traffic_light_id)
    return loc, info


def location_to_dict(location: "carla.Location") -> Dict[str, float]:
    return {"x": float(location.x), "y": float(location.y), "z": float(location.z)}


def location_from_dict(row: Dict[str, object]) -> "carla.Location":
    return carla.Location(
        x=float(row["x"]),
        y=float(row["y"]),
        z=float(row["z"]),
    )


def copy_location(location: "carla.Location") -> "carla.Location":
    return carla.Location(x=float(location.x), y=float(location.y), z=float(location.z))


def rotation_to_dict(rotation: "carla.Rotation") -> Dict[str, float]:
    return {
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll),
    }


def transform_to_dict(transform: "carla.Transform") -> Dict[str, object]:
    return {
        "location": location_to_dict(transform.location),
        "rotation": rotation_to_dict(transform.rotation),
    }


def transform_from_dict(row: Dict[str, object]) -> "carla.Transform":
    loc = row["location"]
    rot = row["rotation"]
    return carla.Transform(
        carla.Location(
            x=float(loc["x"]),
            y=float(loc["y"]),
            z=float(loc["z"]),
        ),
        carla.Rotation(
            pitch=float(rot["pitch"]),
            yaw=float(rot["yaw"]),
            roll=float(rot["roll"]),
        ),
    )


def write_bgra_image(record: Dict[str, object], path_stem: Path) -> str:
    """Write a CARLA BGRA image buffer as PNG when possible, otherwise PPM."""
    width = int(record["width"])
    height = int(record["height"])
    raw = bytes(record["raw_data"])
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        frame_bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        png_path = path_stem.with_suffix(".png")
        if cv2.imwrite(str(png_path), frame_bgr):
            return png_path.name
    except Exception:
        pass

    # Portable fallback: PPM stores RGB bytes without external dependencies.
    ppm_path = path_stem.with_suffix(".ppm")
    rgb = bytearray(width * height * 3)
    out_index = 0
    for in_index in range(0, min(len(raw), width * height * 4), 4):
        blue = raw[in_index]
        green = raw[in_index + 1]
        red = raw[in_index + 2]
        rgb[out_index] = red
        rgb[out_index + 1] = green
        rgb[out_index + 2] = blue
        out_index += 3
    with ppm_path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(rgb)
    return ppm_path.name


def vector_bearing_deg(origin: "carla.Location", target: "carla.Location") -> float:
    return math.degrees(math.atan2(target.y - origin.y, target.x - origin.x))


def angular_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def signed_angular_difference_deg(target: float, source: float) -> float:
    return (target - source + 180.0) % 360.0 - 180.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def yaw_unit_vectors(yaw_deg: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    yaw_rad = math.radians(yaw_deg)
    forward = (math.cos(yaw_rad), math.sin(yaw_rad))
    right = (math.cos(yaw_rad + math.pi / 2.0), math.sin(yaw_rad + math.pi / 2.0))
    return forward, right


def offset_location(
    origin: "carla.Location",
    yaw_deg: float,
    forward_m: float,
    right_m: float,
    z_offset_m: float = 0.0,
) -> "carla.Location":
    forward, right = yaw_unit_vectors(yaw_deg)
    return carla.Location(
        x=origin.x + forward[0] * forward_m + right[0] * right_m,
        y=origin.y + forward[1] * forward_m + right[1] * right_m,
        z=origin.z + z_offset_m,
    )


def ego_spawn_score(spawn_point: "carla.Transform", anchor: "carla.Location", target_distance_m: float) -> float:
    distance = spawn_point.location.distance(anchor)
    bearing_to_anchor = vector_bearing_deg(spawn_point.location, anchor)
    heading_error = angular_difference_deg(float(spawn_point.rotation.yaw), bearing_to_anchor)
    return abs(distance - target_distance_m) + heading_error * 0.2


def choose_route_next_waypoint(
    current_wp: "carla.Waypoint",
    next_wps: Sequence["carla.Waypoint"],
    route_choice: str,
) -> "carla.Waypoint":
    if not next_wps:
        return current_wp
    current_yaw = float(current_wp.transform.rotation.yaw)

    def yaw_delta(wp: "carla.Waypoint") -> float:
        return signed_angular_difference_deg(float(wp.transform.rotation.yaw), current_yaw)

    if route_choice == "left":
        return max(next_wps, key=yaw_delta)
    if route_choice == "right":
        return min(next_wps, key=yaw_delta)
    return min(next_wps, key=lambda wp: abs(yaw_delta(wp)))


def generate_route_waypoints(
    start_wp: "carla.Waypoint",
    route_choice: str,
    total_distance_m: float,
    step_m: float = 3.0,
) -> List["carla.Waypoint"]:
    route = [start_wp]
    traveled = 0.0
    current = start_wp
    while traveled < total_distance_m:
        next_wps = current.next(step_m)
        if not next_wps:
            break
        nxt = choose_route_next_waypoint(current, next_wps, route_choice)
        traveled += current.transform.location.distance(nxt.transform.location)
        route.append(nxt)
        current = nxt
    return route


def route_transform_at_distance(
    route: Sequence["carla.Waypoint"],
    distance_m: float,
) -> "carla.Transform":
    if not route:
        raise RuntimeError("Cannot sample an empty route.")
    traveled = 0.0
    previous = route[0]
    for waypoint in route[1:]:
        segment = previous.transform.location.distance(waypoint.transform.location)
        traveled += segment
        if traveled >= distance_m:
            return waypoint.transform
        previous = waypoint
    return route[-1].transform


def route_crosswalk_candidate(
    world_map: "carla.Map",
    route: Sequence["carla.Waypoint"],
    min_distance_m: float,
    max_distance_m: float,
    desired_distance_m: float,
    max_crosswalk_gap_m: float = 8.0,
) -> Optional[Tuple[float, "carla.Transform", float]]:
    if not hasattr(world_map, "get_crosswalks"):
        return None
    try:
        crosswalk_points = list(world_map.get_crosswalks())
    except Exception:
        return None
    if not route or not crosswalk_points:
        return None

    best: Optional[Tuple[float, float, "carla.Transform", float]] = None
    traveled = 0.0
    previous = route[0]
    for waypoint in route[1:]:
        traveled += previous.transform.location.distance(waypoint.transform.location)
        previous = waypoint
        if traveled < min_distance_m or traveled > max_distance_m:
            continue
        nearest_crosswalk_gap = min(
            waypoint.transform.location.distance(point)
            for point in crosswalk_points
        )
        if nearest_crosswalk_gap > max_crosswalk_gap_m:
            continue
        score = nearest_crosswalk_gap + abs(traveled - desired_distance_m) * 0.08
        if best is None or score < best[0]:
            best = (score, traveled, waypoint.transform, nearest_crosswalk_gap)
    if best is None:
        return None
    return best[1], best[2], best[3]


def first_route_branch_distance(
    start_wp: "carla.Waypoint",
    route_choice: str,
    search_distance_m: float = 65.0,
    step_m: float = 3.0,
) -> Optional[float]:
    if route_choice not in {"left", "right"}:
        return None
    traveled = 0.0
    current = start_wp
    while traveled < search_distance_m:
        next_wps = current.next(step_m)
        if not next_wps:
            return None
        if len(next_wps) > 1:
            current_yaw = float(current.transform.rotation.yaw)
            deltas = [
                signed_angular_difference_deg(float(wp.transform.rotation.yaw), current_yaw)
                for wp in next_wps
            ]
            if route_choice == "left" and max(deltas) > 20.0:
                return traveled
            if route_choice == "right" and min(deltas) < -20.0:
                return traveled
        nxt = choose_route_next_waypoint(current, next_wps, route_choice)
        traveled += current.transform.location.distance(nxt.transform.location)
        current = nxt
    return None


def choose_vehicle_blueprints(world: "carla.World", cars_only: bool = True) -> List["carla.ActorBlueprint"]:
    blueprints = []
    for blueprint in world.get_blueprint_library().filter("vehicle.*"):
        if blueprint.has_attribute("number_of_wheels"):
            try:
                if int(blueprint.get_attribute("number_of_wheels").as_int()) != 4:
                    continue
            except RuntimeError:
                pass
        if cars_only and blueprint.has_attribute("base_type"):
            if str(blueprint.get_attribute("base_type")) != "car":
                continue
        blueprints.append(blueprint)
    if not blueprints and cars_only:
        return choose_vehicle_blueprints(world, cars_only=False)
    return sorted(blueprints, key=lambda bp: bp.id)


def choose_walker_blueprints(world: "carla.World") -> List["carla.ActorBlueprint"]:
    return sorted(world.get_blueprint_library().filter("walker.pedestrian.*"), key=lambda bp: bp.id)


def configure_vehicle_blueprint(
    blueprint: "carla.ActorBlueprint",
    role_name: str,
    rng: random.Random,
) -> "carla.ActorBlueprint":
    configured = blueprint
    if configured.has_attribute("role_name"):
        configured.set_attribute("role_name", role_name)
    if configured.has_attribute("color"):
        values = configured.get_attribute("color").recommended_values
        if values:
            configured.set_attribute("color", rng.choice(values))
    if configured.has_attribute("driver_id"):
        values = configured.get_attribute("driver_id").recommended_values
        if values:
            configured.set_attribute("driver_id", rng.choice(values))
    return configured


def configure_walker_blueprint(
    blueprint: "carla.ActorBlueprint",
    speed_mode: str = "default",
) -> "carla.ActorBlueprint":
    configured = blueprint
    if configured.has_attribute("is_invincible"):
        configured.set_attribute("is_invincible", "false")
    if str(speed_mode) == "running" and configured.has_attribute("speed"):
        values = list(configured.get_attribute("speed").recommended_values)
        if len(values) >= 3:
            configured.set_attribute("speed", str(values[2]))
        elif values:
            configured.set_attribute("speed", str(values[-1]))
    return configured


def spawn_points_near(
    world: "carla.World",
    anchor: "carla.Location",
    radius_m: float,
    min_distance_m: float = 0.0,
) -> List["carla.Transform"]:
    spawn_points = world.get_map().get_spawn_points()
    nearby = [
        sp
        for sp in spawn_points
        if min_distance_m <= sp.location.distance(anchor) <= radius_m
    ]
    if not nearby:
        nearby = spawn_points
    return sorted(nearby, key=lambda sp: sp.location.distance(anchor))


def pick_spawn_point(
    candidates: Sequence["carla.Transform"],
    anchor: "carla.Location",
    target_distance_m: float,
    used_locations: Sequence["carla.Location"],
    min_gap_m: float = 7.5,
) -> Optional["carla.Transform"]:
    for sp in sorted(candidates, key=lambda item: abs(item.location.distance(anchor) - target_distance_m)):
        if all(sp.location.distance(other) >= min_gap_m for other in used_locations):
            return sp
    return None


def spawn_vehicle(
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    blueprint_id: str,
    transform: "carla.Transform",
    role_name: str,
    rng: random.Random,
    autopilot: bool,
) -> Optional["carla.Actor"]:
    blueprint = configure_vehicle_blueprint(world.get_blueprint_library().find(blueprint_id), role_name, rng)
    command = carla.command.SpawnActor(blueprint, transform)
    if autopilot:
        command = command.then(
            carla.command.SetAutopilot(
                carla.command.FutureActor,
                True,
                traffic_manager.get_port(),
            )
        )
    response = client.apply_batch_sync([command], True)[0]
    if response.error:
        return None
    return world.get_actor(response.actor_id)


def find_first_blueprint(
    world: "carla.World",
    preferred_ids: Sequence[str],
    fallback_filter: str,
) -> "carla.ActorBlueprint":
    library = world.get_blueprint_library()
    available = {bp.id: bp for bp in library.filter(fallback_filter)}
    for bp_id in preferred_ids:
        if bp_id in available:
            return available[bp_id]
    if available:
        return sorted(available.values(), key=lambda bp: bp.id)[0]
    return library.find(preferred_ids[0])


def try_spawn_configured_actor(
    world: "carla.World",
    blueprint: "carla.ActorBlueprint",
    transform: "carla.Transform",
    role_name: str,
) -> Optional["carla.Actor"]:
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")
    try:
        return world.try_spawn_actor(blueprint, transform)
    except RuntimeError:
        return None


def unique_blueprints_by_id(
    blueprints: Iterable["carla.ActorBlueprint"],
) -> List["carla.ActorBlueprint"]:
    seen = set()
    unique: List["carla.ActorBlueprint"] = []
    for blueprint in blueprints:
        bp_id = str(blueprint.id)
        if bp_id in seen:
            continue
        seen.add(bp_id)
        unique.append(blueprint)
    return unique


def try_spawn_configured_actor_variants(
    world: "carla.World",
    blueprint_options: Sequence["carla.ActorBlueprint"],
    transform: "carla.Transform",
    role_name: str,
    nudge_yaw_deg: float,
    forward_nudges_m: Sequence[float] = (0.0,),
    right_nudges_m: Sequence[float] = (0.0,),
    z_nudges_m: Sequence[float] = (0.0,),
    yaw_nudges_deg: Sequence[float] = (0.0,),
) -> Tuple[Optional["carla.Actor"], Optional["carla.ActorBlueprint"], Optional["carla.Transform"], Dict[str, object]]:
    attempts = 0
    blueprints = unique_blueprints_by_id(blueprint_options)
    for blueprint in blueprints:
        for forward_nudge in forward_nudges_m:
            for right_nudge in right_nudges_m:
                for z_nudge in z_nudges_m:
                    for yaw_nudge in yaw_nudges_deg:
                        attempts += 1
                        candidate = carla.Transform(
                            offset_location(
                                transform.location,
                                nudge_yaw_deg,
                                forward_m=float(forward_nudge),
                                right_m=float(right_nudge),
                                z_offset_m=float(z_nudge),
                            ),
                            carla.Rotation(
                                pitch=float(transform.rotation.pitch),
                                yaw=float(transform.rotation.yaw) + float(yaw_nudge),
                                roll=float(transform.rotation.roll),
                            ),
                        )
                        actor = try_spawn_configured_actor(world, blueprint, candidate, role_name)
                        if actor is not None:
                            return (
                                actor,
                                blueprint,
                                candidate,
                                {
                                    "attempts": attempts,
                                    "blueprint_id": str(blueprint.id),
                                    "forward_nudge_m": float(forward_nudge),
                                    "right_nudge_m": float(right_nudge),
                                    "z_nudge_m": float(z_nudge),
                                    "yaw_nudge_deg": float(yaw_nudge),
                                },
                            )
    return (
        None,
        None,
        None,
        {
            "attempts": attempts,
            "blueprint_ids": [str(blueprint.id) for blueprint in blueprints],
        },
    )


def resolve_vehicle_blueprint_ids(world: "carla.World", rng: random.Random) -> List[str]:
    by_id = {bp.id: bp for bp in choose_vehicle_blueprints(world, cars_only=True)}
    preferred = [bp_id for bp_id in SAFE_VEHICLE_BLUEPRINTS if bp_id in by_id]
    if preferred:
        return preferred
    return [bp.id for bp in choose_vehicle_blueprints(world, cars_only=True)]


def spawn_ego(
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    candidates: Sequence["carla.Transform"],
    anchor: "carla.Location",
    spec: ScenarioSpec,
    rng: random.Random,
    autopilot: bool,
) -> "carla.Actor":
    blueprint_ids = resolve_vehicle_blueprint_ids(world, rng)
    used: List["carla.Location"] = []
    for sp in sorted(candidates, key=lambda item: abs(item.location.distance(anchor) - spec.ego_distance_m)):
        actor = spawn_vehicle(
            client,
            world,
            traffic_manager,
            rng.choice(blueprint_ids),
            sp,
            f"{SCENESENSE_ROLE_PREFIX}{spec.name}_ego",
            rng,
            autopilot,
        )
        if actor is not None:
            return actor
        used.append(sp.location)
    raise RuntimeError("Unable to spawn ego vehicle for scenario.")


def choose_ego_spawn_toward_anchor(
    world: "carla.World",
    candidates: Sequence["carla.Transform"],
    anchor: "carla.Location",
    target_distance_m: float,
    route_choice: str,
) -> "carla.Transform":
    world_map = world.get_map()
    usable = []
    for sp in candidates:
        if not (20.0 <= sp.location.distance(anchor) <= 85.0):
            continue
        if angular_difference_deg(float(sp.rotation.yaw), vector_bearing_deg(sp.location, anchor)) > 100.0:
            continue
        try:
            wp = world_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except RuntimeError:
            wp = None
        if wp is None:
            continue
        branch_distance = first_route_branch_distance(wp, route_choice)
        branch_penalty = 0.0
        if route_choice in {"left", "right"}:
            branch_penalty = 500.0 if branch_distance is None else abs(branch_distance - 20.0)
        usable.append((ego_spawn_score(sp, anchor, target_distance_m) + branch_penalty, sp))
    pool = [sp for _, sp in sorted(usable, key=lambda item: item[0])] or list(candidates)
    if not pool:
        raise RuntimeError("No spawn points available for occlusion crossing scenario.")
    return pool[0]


def spawn_occlusion_crossing_layout(
    world: "carla.World",
    client: "carla.Client",
    traffic_manager: "carla.TrafficManager",
    candidates: Sequence["carla.Transform"],
    anchor: "carla.Location",
    spec: ScenarioSpec,
    rng: random.Random,
    ego_autopilot: bool,
    route_choice: str,
) -> Tuple["carla.Actor", List["carla.Actor"], Dict[str, object]]:
    ego_sp = choose_ego_spawn_toward_anchor(world, candidates, anchor, spec.ego_distance_m, route_choice)
    start_wp = world.get_map().get_waypoint(
        ego_sp.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    route = generate_route_waypoints(start_wp, route_choice, total_distance_m=55.0)
    occluder_route_tf = route_transform_at_distance(route, 18.0)
    target_route_tf = route_transform_at_distance(route, 26.0)

    ego = spawn_vehicle(
        client,
        world,
        traffic_manager,
        rng.choice(resolve_vehicle_blueprint_ids(world, rng)),
        ego_sp,
        f"{SCENESENSE_ROLE_PREFIX}{spec.name}_ego",
        rng,
        autopilot=ego_autopilot,
    )
    if ego is None:
        raise RuntimeError("Unable to spawn occlusion-crossing ego vehicle.")

    yaw = float(occluder_route_tf.rotation.yaw)
    occluder_transform = carla.Transform(
        offset_location(occluder_route_tf.location, yaw, forward_m=0.0, right_m=2.6, z_offset_m=0.4),
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
    )
    target_yaw = float(target_route_tf.rotation.yaw)
    target_start = offset_location(target_route_tf.location, target_yaw, forward_m=0.0, right_m=3.4, z_offset_m=1.0)
    target_end = offset_location(target_route_tf.location, target_yaw, forward_m=0.0, right_m=-3.0, z_offset_m=1.0)
    target_transform = carla.Transform(
        target_start,
        carla.Rotation(pitch=0.0, yaw=target_yaw - 90.0, roll=0.0),
    )

    occluder_bp = find_first_blueprint(
        world,
        (
            "vehicle.carlamotors.carlacola",
            "vehicle.mercedes.sprinter",
            "vehicle.volkswagen.t2",
            "vehicle.ford.ambulance",
            "vehicle.lincoln.mkz",
        ),
        "vehicle.*",
    )
    occluder = try_spawn_configured_actor(
        world,
        occluder_bp,
        occluder_transform,
        f"{SCENESENSE_ROLE_PREFIX}{spec.name}_occluder_vehicle",
    )
    if occluder is None:
        yaw = float(occluder_route_tf.rotation.yaw)
        occluder_transform = carla.Transform(
            offset_location(occluder_route_tf.location, yaw, forward_m=0.0, right_m=3.8, z_offset_m=0.4),
            carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        )
        occluder = try_spawn_configured_actor(
            world,
            occluder_bp,
            occluder_transform,
            f"{SCENESENSE_ROLE_PREFIX}{spec.name}_occluder_vehicle",
        )

    walker_bp = configure_walker_blueprint(
        find_first_blueprint(
            world,
            ("walker.pedestrian.0001", "walker.pedestrian.0010", "walker.pedestrian.0020"),
            "walker.pedestrian.*",
        ),
        speed_mode="running",
    )
    target = try_spawn_configured_actor(
        world,
        walker_bp,
        target_transform,
        f"{SCENESENSE_ROLE_PREFIX}{spec.name}_target_pedestrian",
    )
    if target is None:
        target_transform = carla.Transform(
            offset_location(target_route_tf.location, target_yaw, forward_m=0.0, right_m=4.2, z_offset_m=1.0),
            carla.Rotation(pitch=0.0, yaw=target_yaw - 90.0, roll=0.0),
        )
        target = try_spawn_configured_actor(
            world,
            walker_bp,
            target_transform,
            f"{SCENESENSE_ROLE_PREFIX}{spec.name}_target_pedestrian",
        )

    if occluder is None or target is None:
        for actor in (target, occluder, ego):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        raise RuntimeError("Unable to spawn required occlusion-crossing occluder/target actors.")

    extra_actors = [actor for actor in (occluder, target) if actor is not None]
    layout = {
        "type": "occlusion_crossing_ego",
        "ego_actor_id": int(ego.id),
        "occluder_actor_id": None if occluder is None else int(occluder.id),
        "target_actor_id": None if target is None else int(target.id),
        "target_role": f"{SCENESENSE_ROLE_PREFIX}{spec.name}_target_pedestrian",
        "ego_spawn_transform": transform_to_dict(ego_sp),
        "occluder_transform": transform_to_dict(occluder_transform),
        "target_start_transform": transform_to_dict(target_transform),
        "target_crossing_end_location": location_to_dict(target_end),
        "route_choice": route_choice,
        "controller_route_transforms": [transform_to_dict(waypoint.transform) for waypoint in route],
        "route_waypoints": [
            transform_to_dict(waypoint.transform)
            for waypoint in route[:: max(1, len(route) // 12)]
        ],
        "design_note": (
            "The target starts behind/near the occluder from the ego route. "
            "Use --target-crossing and --scripted-ego-drive for a route-aware failure-case motion pass."
        ),
    }
    return ego, extra_actors, layout


def spawn_curbside_parked_pedestrian_layout(
    world: "carla.World",
    client: "carla.Client",
    traffic_manager: "carla.TrafficManager",
    candidates: Sequence["carla.Transform"],
    anchor: "carla.Location",
    spec: ScenarioSpec,
    rng: random.Random,
    ego_autopilot: bool,
    route_choice: str,
    ego_spawn_index: int = -1,
    curbside_conflict_distance_m: float = 31.0,
    curbside_occluder_lateral_offset_m: float = 3.2,
    curbside_target_start_lateral_offset_m: float = 3.8,
    curbside_target_end_lateral_offset_m: float = -0.6,
    curbside_target_forward_offset_m: float = 1.2,
    curbside_ego_start_forward_m: float = 0.0,
    curbside_target_prewalk_distance_m: float = 0.0,
    curbside_target_prewalk_lateral_offset_m: float = -1.0,
    curbside_heavy_occluder_first: bool = True,
    helper_vehicle: bool = False,
    helper_drive: bool = False,
    helper_spawn_forward_m: float = 35.0,
    helper_target_forward_m: float = -30.0,
    extra_occluder_forward_m: float = 0.0,
    slot_0_forward_m: float = -5.5,
    slot_1_forward_m: float = -1.0,
    occluder_count: int = 3,
    forced_occluder_blueprint_id: str = "",
    occluder_yaw_offset_deg: float = 0.0,
    occluder_z_offset_m: float = 0.1,
) -> Tuple["carla.Actor", List["carla.Actor"], Dict[str, object]]:
    requested_route_choice = str(route_choice)
    route_choice = "straight"
    spawn_points = world.get_map().get_spawn_points()
    if int(ego_spawn_index) >= 0 and spawn_points:
        ego_spawn_index = int(ego_spawn_index) % len(spawn_points)
        ego_sp = spawn_points[ego_spawn_index]
    else:
        ego_spawn_index = -1
        ego_sp = choose_ego_spawn_toward_anchor(world, candidates, anchor, spec.ego_distance_m, route_choice)
    start_wp = world.get_map().get_waypoint(
        ego_sp.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    route = generate_route_waypoints(start_wp, route_choice, total_distance_m=80.0)
    if len(route) < 8:
        raise RuntimeError("Unable to generate a long enough curbside occlusion ego route.")

    route_base_ego_sp = ego_sp
    conflict_distance = max(15.0, float(curbside_conflict_distance_m))
    ego_start_forward_m = max(0.0, float(curbside_ego_start_forward_m))
    # Keep at least a small runway before the conflict zone. Larger offsets
    # make the ego spawn too close to the occluder/pedestrian cluster and can
    # disturb the scene geometry we are trying to preserve.
    ego_start_forward_m = min(ego_start_forward_m, max(0.0, conflict_distance - 8.0))
    if ego_start_forward_m > 0.01:
        sampled_ego_tf = route_transform_at_distance(route, ego_start_forward_m)
        ego_sp = carla.Transform(
            carla.Location(
                x=float(sampled_ego_tf.location.x),
                y=float(sampled_ego_tf.location.y),
                z=float(sampled_ego_tf.location.z + 0.2),
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=float(sampled_ego_tf.rotation.yaw),
                roll=0.0,
            ),
        )

    primary_occluder_distance = max(12.0, conflict_distance - 2.0)
    conflict_tf = route_transform_at_distance(route, conflict_distance)
    occluder_tf = route_transform_at_distance(route, primary_occluder_distance)
    route_yaw = float(conflict_tf.rotation.yaw)
    curb_side = 1.0
    occluder_lateral_offset_m = curb_side * abs(float(curbside_occluder_lateral_offset_m))
    target_start_lateral_offset_m = curb_side * abs(float(curbside_target_start_lateral_offset_m))
    target_end_lateral_offset_m = -curb_side * abs(float(curbside_target_end_lateral_offset_m))
    target_forward_offset_m = float(curbside_target_forward_offset_m)
    target_prewalk_distance_m = max(0.0, float(curbside_target_prewalk_distance_m))
    if float(curbside_target_prewalk_lateral_offset_m) >= 0.0:
        target_prewalk_lateral_offset_m = curb_side * abs(float(curbside_target_prewalk_lateral_offset_m))
    else:
        target_prewalk_lateral_offset_m = target_start_lateral_offset_m

    ego = spawn_vehicle(
        client,
        world,
        traffic_manager,
        rng.choice(resolve_vehicle_blueprint_ids(world, rng)),
        ego_sp,
        f"{SCENESENSE_ROLE_PREFIX}{spec.name}_ego",
        rng,
        autopilot=ego_autopilot,
    )
    if ego is None:
        raise RuntimeError("Unable to spawn curbside-occlusion ego vehicle.")

    vehicle_blueprints = sorted(world.get_blueprint_library().filter("vehicle.*"), key=lambda bp: bp.id)
    vehicle_blueprints_by_id = {bp.id: bp for bp in vehicle_blueprints}
    # CARLA 0.10 truck/van pool. Confirmed available blueprint IDs (from
    # `--list-vehicles`): sprinter.mercedes, carlacola.actors, firetruck.actors,
    # ambulance.ford. Older IDs (carlamotors.carlacola, carlamotors.firetruck,
    # mercedes.sprinter, volkswagen.t2, tesla.cybertruck, mitsubishi.fusorosa)
    # don't exist in 0.10. Order: medium vans first (visually consistent with
    # typical curbside), larger trucks last as fallback.
    heavy_preferred_ids = (
        "vehicle.sprinter.mercedes",
        "vehicle.carlacola.actors",
        "vehicle.ambulance.ford",
        # Largest, last resort:
        "vehicle.firetruck.actors",
    )
    # CARLA 0.10 sedan/car pool. Older IDs (toyota.prius, audi.a2,
    # mkz_2017/2020, mercedes.coupe_2020) don't exist in 0.10.
    car_fallback_ids = (
        "vehicle.lincoln.mkz",
        "vehicle.dodge.charger",
        "vehicle.mini.cooper",
        "vehicle.taxi.ford",
        "vehicle.nissan.patrol",
    )
    heavy_tokens = ("carlacola", "firetruck", "ambulance", "bus", "truck", "sprinter", "van", "cybertruck")
    heavy_occluder_candidate_ids = [
        str(bp.id)
        for bp in vehicle_blueprints
        if any(token in str(bp.id).lower() for token in heavy_tokens)
    ]
    # When the user asks for heavy occluders first, treat that as STRICT:
    # only vans / trucks / box-vehicles are eligible for any of the three
    # parked slots. Sedan fallbacks were leaking in when the heavy spawn
    # collided on first try, letting the pedestrian's head clear short
    # rooflines and ruining the occlusion from the ego camera.
    if curbside_heavy_occluder_first:
        occluder_blueprint_options = [
            vehicle_blueprints_by_id[bp_id]
            for bp_id in heavy_preferred_ids
            if bp_id in vehicle_blueprints_by_id
        ]
        # Also add any other blueprint matching a heavy token (covers map
        # variants we didn't enumerate explicitly), but no sedan fallback.
        for token in heavy_tokens:
            occluder_blueprint_options.extend(
                [bp for bp in vehicle_blueprints if token in str(bp.id).lower()]
            )
    else:
        preferred_ids = (*car_fallback_ids, *heavy_preferred_ids)
        occluder_blueprint_options = [
            vehicle_blueprints_by_id[bp_id]
            for bp_id in preferred_ids
            if bp_id in vehicle_blueprints_by_id
        ]
        for token in heavy_tokens:
            occluder_blueprint_options.extend(
                [bp for bp in vehicle_blueprints if token in str(bp.id).lower()]
            )
        for bp_id in SAFE_VEHICLE_BLUEPRINTS:
            if bp_id in vehicle_blueprints_by_id:
                occluder_blueprint_options.append(vehicle_blueprints_by_id[bp_id])
    occluder_blueprint_options = unique_blueprints_by_id(occluder_blueprint_options or vehicle_blueprints[:5])

    # Per-slot blueprint pools so the pedestrian's emergence is between a van
    # (slot 1, immediately on the ego-approach side) and a car (slot 2, past
    # the pedestrian). Pattern: [van, van, sedan]. Slot 1 must be tall to
    # actually occlude the standing pedestrian from the ego camera; slot 2 is
    # a sedan so the visual is "between a van and a car", matching the typical
    # mixed curbside in the demo brief.
    # If user forced a specific blueprint via --curbside-occluder-blueprint,
    # use only that one (with optional retries via spawn nudges). Otherwise
    # fall back to the heavy/sedan pool selection.
    forced_bp = None
    if forced_occluder_blueprint_id:
        forced_bp = vehicle_blueprints_by_id.get(forced_occluder_blueprint_id)
        if forced_bp is None:
            print(
                f"[curbside] WARNING: --curbside-occluder-blueprint "
                f"'{forced_occluder_blueprint_id}' not found in this CARLA world; "
                f"falling back to the heavy-pool default."
            )

    if forced_bp is not None:
        forced_only_pool = [forced_bp]
        slot_blueprint_options = {0: forced_only_pool, 1: forced_only_pool, 2: forced_only_pool}
    elif curbside_heavy_occluder_first:
        van_pool = unique_blueprints_by_id([
            vehicle_blueprints_by_id[bp_id]
            for bp_id in heavy_preferred_ids
            if bp_id in vehicle_blueprints_by_id
        ] + [
            bp for bp in vehicle_blueprints
            if any(token in str(bp.id).lower() for token in heavy_tokens)
        ])
        sedan_pool = unique_blueprints_by_id([
            vehicle_blueprints_by_id[bp_id]
            for bp_id in SAFE_VEHICLE_BLUEPRINTS
            if bp_id in vehicle_blueprints_by_id
        ] + [
            vehicle_blueprints_by_id[bp_id]
            for bp_id in car_fallback_ids
            if bp_id in vehicle_blueprints_by_id
        ])
        slot_blueprint_options = {0: van_pool, 1: van_pool, 2: sedan_pool}
    else:
        slot_blueprint_options = {0: occluder_blueprint_options, 1: occluder_blueprint_options, 2: occluder_blueprint_options}

    occluders: List["carla.Actor"] = []
    occluder_spawn_infos: List[Dict[str, object]] = []
    occluder_blueprint_ids: List[str] = []
    planned_occluder_transforms: List["carla.Transform"] = []
    spawned_occluder_transforms: List["carla.Transform"] = []
    # Tighter spacing closes the gaps between parked vehicles so the pedestrian
    # cannot be seen "between" cars from the ego camera. Original spacing
    # (-8.0, -1.6, 5.0) left ~2.5 m air gaps; this brings them down to ~0.5 m
    # while still giving the variant-spawner enough room to nudge for spawn
    # collisions. The middle slot is the heavy occluder (van/truck), the rear
    # slot is a sedan so the pedestrian emerges between a van and a car.
    # Choose which slots to spawn based on --curbside-occluder-count.
    # count=3 → all three [slot 0, slot 1, slot 2] (default)
    # count=2 → drop slot 0 (keep the truck + the rear sedan)
    # count=1 → only slot 1 (the truck) — use this when baked map vehicles
    #          already fill the rest of the curbside row, so the truck just
    #          adds the immediate occluder for the pedestrian.
    all_slots = (
        (0, float(slot_0_forward_m)),
        (1, float(slot_1_forward_m)),
        (2, 4.5),
    )
    if int(occluder_count) == 1:
        selected_slots = (all_slots[1],)
    elif int(occluder_count) == 2:
        selected_slots = (all_slots[1], all_slots[2])
    else:
        selected_slots = all_slots
    for index, forward_m in selected_slots:
        if index == 1:
            role_suffix = "parked_van_occluder"
        elif index == 2:
            role_suffix = "parked_car_far"
        else:
            role_suffix = f"parked_curb_vehicle_{index}"
        role = f"{SCENESENSE_ROLE_PREFIX}{spec.name}_{role_suffix}"
        transform = carla.Transform(
            offset_location(
                occluder_tf.location,
                route_yaw,
                forward_m=forward_m,
                right_m=occluder_lateral_offset_m,
                z_offset_m=float(occluder_z_offset_m),
            ),
            carla.Rotation(pitch=0.0, yaw=route_yaw + float(occluder_yaw_offset_deg), roll=0.0),
        )
        planned_occluder_transforms.append(transform)
        slot_options = slot_blueprint_options.get(index) or occluder_blueprint_options
        actor, used_bp, used_transform, spawn_info = try_spawn_configured_actor_variants(
            world,
            slot_options,
            transform,
            role,
            nudge_yaw_deg=route_yaw,
            forward_nudges_m=(0.0, -0.8, 0.8, -1.6, 1.6),
            right_nudges_m=(0.0, curb_side * 0.6, -curb_side * 0.4, curb_side * 1.0),
            z_nudges_m=(0.0, 0.2),
        )
        spawn_info.update({"role": role, "planned_forward_m": float(forward_m)})
        occluder_spawn_infos.append(spawn_info)
        if actor is not None:
            try:
                actor.set_simulate_physics(False)
            except RuntimeError:
                pass
            occluders.append(actor)
            if used_bp is not None:
                occluder_blueprint_ids.append(str(used_bp.id))
            if used_transform is not None:
                spawned_occluder_transforms.append(used_transform)

    # Optional 4th occluder (van) at a user-specified forward offset. Useful
    # when the spawn point has baked-in map vehicles that consume one of the
    # default three slots, leaving an obvious empty parking spot. The user
    # supplies the forward offset via --curbside-extra-occluder-forward-m.
    if abs(float(extra_occluder_forward_m)) > 0.01:
        extra_role = f"{SCENESENSE_ROLE_PREFIX}{spec.name}_parked_extra_van"
        extra_transform = carla.Transform(
            offset_location(
                occluder_tf.location,
                route_yaw,
                forward_m=float(extra_occluder_forward_m),
                right_m=occluder_lateral_offset_m,
                z_offset_m=float(occluder_z_offset_m),
            ),
            carla.Rotation(pitch=0.0, yaw=route_yaw + float(occluder_yaw_offset_deg), roll=0.0),
        )
        planned_occluder_transforms.append(extra_transform)
        # Use the van pool (same one slot 1 draws from when heavy-first is on).
        extra_pool = slot_blueprint_options.get(1) or occluder_blueprint_options
        extra_actor, extra_bp, extra_used_tf, extra_info = try_spawn_configured_actor_variants(
            world,
            extra_pool,
            extra_transform,
            extra_role,
            nudge_yaw_deg=route_yaw,
            forward_nudges_m=(0.0, -0.8, 0.8, -1.6, 1.6),
            right_nudges_m=(0.0, curb_side * 0.6, -curb_side * 0.4, curb_side * 1.0),
            z_nudges_m=(0.0, 0.2),
        )
        extra_info.update({"role": extra_role, "planned_forward_m": float(extra_occluder_forward_m)})
        occluder_spawn_infos.append(extra_info)
        if extra_actor is not None:
            try:
                extra_actor.set_simulate_physics(False)
            except RuntimeError:
                pass
            occluders.append(extra_actor)
            if extra_bp is not None:
                occluder_blueprint_ids.append(str(extra_bp.id))
            if extra_used_tf is not None:
                spawned_occluder_transforms.append(extra_used_tf)

    # Walker anchor: the route position where the walker actually crosses.
    # Previously target_start used forward_m=target_forward_offset_m but
    # target_end used forward_m=0, which made the walker walk diagonally
    # toward conflict_tf if target_forward_offset_m != 0. Now we resolve a
    # walker_anchor along the route at conflict_distance + offset, and the
    # walker walks STRAIGHT across (perpendicular) at that anchor. The
    # crossing trigger is repositioned to this anchor so ego times its
    # arrival to where the walker actually crosses, not to conflict_tf.
    walker_anchor_route_distance = float(conflict_distance) + float(target_forward_offset_m)
    walker_anchor_tf = route_transform_at_distance(route, walker_anchor_route_distance)
    walker_anchor_yaw = float(walker_anchor_tf.rotation.yaw)
    target_start = offset_location(
        walker_anchor_tf.location,
        walker_anchor_yaw,
        forward_m=0.0,
        right_m=target_start_lateral_offset_m,
        z_offset_m=1.0,
    )
    target_end = offset_location(
        walker_anchor_tf.location,
        walker_anchor_yaw,
        forward_m=0.0,
        right_m=target_end_lateral_offset_m,
        z_offset_m=1.0,
    )
    target_prewalk_start = (
        offset_location(
            walker_anchor_tf.location,
            walker_anchor_yaw,
            forward_m=-target_prewalk_distance_m,
            right_m=target_prewalk_lateral_offset_m,
            z_offset_m=1.0,
        )
        if target_prewalk_distance_m > 0.0
        else target_start
    )
    target_transform = carla.Transform(
        target_prewalk_start,
        carla.Rotation(pitch=0.0, yaw=route_yaw - 90.0, roll=0.0),
    )
    walker_bp = configure_walker_blueprint(
        find_first_blueprint(
            world,
            ("walker.pedestrian.0001", "walker.pedestrian.0010", "walker.pedestrian.0020"),
            "walker.pedestrian.*",
        ),
        speed_mode="running",
    )
    target, _target_bp, spawned_target_transform, target_spawn_info = try_spawn_configured_actor_variants(
        world,
        [walker_bp],
        target_transform,
        f"{SCENESENSE_ROLE_PREFIX}{spec.name}_hidden_pedestrian",
        nudge_yaw_deg=route_yaw,
        forward_nudges_m=(0.0, -0.8, 0.8, -1.6, 1.6),
        right_nudges_m=(0.0, curb_side * 0.6, -curb_side * 0.6, curb_side * 1.2),
        z_nudges_m=(0.0, 0.3),
    )
    if spawned_target_transform is not None:
        target_transform = spawned_target_transform

    if not occluders or target is None:
        for actor in [target, *occluders, ego]:
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        raise RuntimeError(
            "Unable to spawn required curbside parked-vehicle/pedestrian actors "
            f"(occluders_spawned={len(occluders)}, target_spawned={target is not None}, "
            f"occluder_spawn_infos={occluder_spawn_infos}, target_spawn_info={target_spawn_info})."
        )

    helper_actor: Optional["carla.Actor"] = None
    helper_blueprint_id = ""
    helper_spawn_info: Dict[str, object] = {}
    helper_transform: Optional["carla.Transform"] = None
    helper_lateral_offset_m = -3.6
    # Helper geometry: spawn far enough past the conflict that the helper
    # arrives at the conflict point during the pedestrian's crossing window.
    # Calibration: time-to-arrival at conflict = helper_spawn_forward_m /
    # helper_target_speed. The defaults (35 m, ~4.5 m/s) put helper at
    # the conflict at ~7-8 s; pair with --target-crossing-delay-s to align
    # the walker with that window. Both values are now CLI-tunable:
    #   --curbside-helper-spawn-forward-m
    #   --curbside-helper-target-forward-m
    # The target stays well past the conflict so the helper does not brake
    # within the danger window.
    helper_target_location = offset_location(
        conflict_tf.location,
        route_yaw,
        forward_m=helper_target_forward_m,
        right_m=helper_lateral_offset_m,
        z_offset_m=0.4,
    )
    if helper_vehicle:
        helper_role = f"{SCENESENSE_ROLE_PREFIX}{spec.name}_opposite_lane_helper"
        helper_base_transform = carla.Transform(
            offset_location(
                conflict_tf.location,
                route_yaw,
                forward_m=helper_spawn_forward_m,
                right_m=helper_lateral_offset_m,
                z_offset_m=0.4,
            ),
            carla.Rotation(pitch=0.0, yaw=route_yaw + 180.0, roll=0.0),
        )
        helper_options = [
            vehicle_blueprints_by_id[bp_id]
            for bp_id in (
                "vehicle.lincoln.mkz",
                "vehicle.tesla.model3",
                "vehicle.audi.a2",
                "vehicle.toyota.prius",
            )
            if bp_id in vehicle_blueprints_by_id
        ]
        helper_actor, helper_bp, helper_transform, helper_spawn_info = try_spawn_configured_actor_variants(
            world,
            helper_options or vehicle_blueprints[:5],
            helper_base_transform,
            helper_role,
            nudge_yaw_deg=route_yaw,
            forward_nudges_m=(0.0, -2.0, 2.0, -4.0, 4.0),
            right_nudges_m=(0.0, -0.6, 0.6),
            z_nudges_m=(0.0, 0.2),
        )
        if helper_actor is not None:
            if not helper_drive:
                try:
                    helper_actor.set_simulate_physics(False)
                except RuntimeError:
                    pass
            if helper_bp is not None:
                helper_blueprint_id = str(helper_bp.id)

    observer_target = carla.Location(
        x=(float(conflict_tf.location.x) + float(occluder_tf.location.x) + float(target_start.x)) / 3.0,
        y=(float(conflict_tf.location.y) + float(occluder_tf.location.y) + float(target_start.y)) / 3.0,
        z=float(conflict_tf.location.z + 1.0),
    )
    observer_camera_location = offset_location(
        observer_target,
        route_yaw,
        forward_m=-18.0,
        right_m=-curb_side * 16.0,
        z_offset_m=22.0,
    )
    observer_spectator_transform = carla.Transform(
        observer_camera_location,
        look_at_rotation(observer_camera_location, observer_target),
    )

    layout = {
        "type": spec.name,
        "ego_actor_id": int(ego.id),
        "ego_spawn_transform": transform_to_dict(ego_sp),
        "ego_route_base_spawn_transform": transform_to_dict(route_base_ego_sp),
        "ego_spawn_index": int(ego_spawn_index),
        "ego_start_forward_m": float(ego_start_forward_m),
        "occlusion_mode": "curbside_parked_vehicle_hidden_pedestrian",
        "requested_route_choice": requested_route_choice,
        "effective_route_choice": route_choice,
        "target_motion_mode": "walker_control",
        "target_walker_speed_mode": "running",
        "target_crossing_control_speed_override": 4.5,
        "target_motion_mode_note": (
            "Curbside baseline uses direct WalkerControl so the hidden pedestrian follows the "
            "planned road-crossing vector. CARLA's ai_controller follows the navigation mesh and "
            "can route along the sidewalk instead of crossing."
        ),
        "target_actor_id": int(target.id),
        "target_role": f"{SCENESENSE_ROLE_PREFIX}{spec.name}_hidden_pedestrian",
        "target_spawn_info": target_spawn_info,
        "target_start_transform": transform_to_dict(target_transform),
        "target_crossing_start_location": location_to_dict(target_start),
        "target_crossing_end_location": location_to_dict(target_end),
        "target_prewalk_start_location": location_to_dict(target_prewalk_start),
        "target_prewalk_end_location": location_to_dict(target_start),
        "target_prewalk_distance_m": float(target_prewalk_distance_m),
        "target_prewalk_lateral_offset_m": float(target_prewalk_lateral_offset_m),
        # Trigger fires when ego is near where the walker will actually cross,
        # not at conflict_tf — important when --curbside-target-forward-offset-m
        # shifts the walker away from conflict_tf.
        "target_crossing_trigger_location": location_to_dict(walker_anchor_tf.location),
        "target_forward_offset_m": float(target_forward_offset_m),
        "target_start_lateral_offset_m": float(target_start_lateral_offset_m),
        "target_end_lateral_offset_m": float(target_end_lateral_offset_m),
        "helper_vehicle_enabled": bool(helper_vehicle),
        "helper_vehicle_actor_id": None if helper_actor is None else int(helper_actor.id),
        "helper_vehicle_role": f"{SCENESENSE_ROLE_PREFIX}{spec.name}_opposite_lane_helper",
        "helper_vehicle_blueprint_id": helper_blueprint_id,
        "helper_vehicle_drive": bool(helper_drive),
        "helper_vehicle_lateral_offset_m": float(helper_lateral_offset_m),
        "helper_vehicle_spawn_forward_m": float(helper_spawn_forward_m),
        "helper_vehicle_target_forward_m": float(helper_target_forward_m),
        "helper_vehicle_target_location": location_to_dict(helper_target_location),
        "helper_vehicle_spawn_info": helper_spawn_info,
        "helper_vehicle_transform": None if helper_transform is None else transform_to_dict(helper_transform),
        "helper_vehicle_purpose": (
            "Opposite-lane observer viewpoint: should see the hidden pedestrian earlier than the ego camera "
            "while continuing through its own lane, not participating in the ego-target collision."
        ),
        "conflict_distance_m": float(conflict_distance),
        "conflict_location": location_to_dict(conflict_tf.location),
        "primary_occluder_distance_m": float(primary_occluder_distance),
        "occluder_lateral_offset_m": float(occluder_lateral_offset_m),
        "occluder_actor_ids": [int(actor.id) for actor in occluders],
        "occluder_primary_actor_id": int(occluders[min(1, len(occluders) - 1)].id),
        "occluder_blueprint_ids": occluder_blueprint_ids,
        "occluder_heavy_blueprint_available": bool(heavy_occluder_candidate_ids),
        "occluder_heavy_candidate_ids": heavy_occluder_candidate_ids,
        "occluder_heavy_occluder_first": bool(curbside_heavy_occluder_first),
        "occluder_blueprint_id": occluder_blueprint_ids[min(1, len(occluder_blueprint_ids) - 1)]
        if occluder_blueprint_ids
        else "",
        "occluder_simulate_physics": False,
        "occluder_spawn_infos": occluder_spawn_infos,
        "occluder_planned_transforms": [transform_to_dict(transform) for transform in planned_occluder_transforms],
        "occluder_spawned_transforms": [transform_to_dict(transform) for transform in spawned_occluder_transforms],
        "occluder_actor_transforms_at_spawn": [transform_to_dict(actor.get_transform()) for actor in occluders],
        "observer_location": location_to_dict(observer_target),
        "observer_spectator_transform": transform_to_dict(observer_spectator_transform),
        "route_choice": route_choice,
        "controller_route_transforms": [transform_to_dict(waypoint.transform) for waypoint in route],
        "route_waypoints": [
            transform_to_dict(waypoint.transform)
            for waypoint in route[:: max(1, len(route) // 14)]
        ],
        "design_note": (
            "Mid-block curbside occlusion: ego drives along a straight urban road while a hidden pedestrian "
            "emerges from behind a parked van/vehicle row. This is intended as a defensible urban blind-spot "
            "case for teleoperation or assisted autonomy, without relying on awkward intersection parking. "
            "The optional helper vehicle is an opposite-lane observer camera for ego-blind/helper-visible "
            "evidence, not yet an autonomous cooperative-control agent."
        ),
    }
    extra_actors = [*occluders, target]
    if helper_actor is not None:
        extra_actors.append(helper_actor)

    # Diagnostic prints so the user can see where everything actually landed
    # and tune --curbside-slot-1-forward-m / --curbside-target-forward-offset-m
    # against the visible curbside layout in CARLA. We use the spawned
    # transforms tracked above rather than actor.get_location() — the latter
    # returns (0,0,0) before CARLA has ticked once.
    try:
        print("[curbside-layout] anchor route_distance (conflict_tf): "
              f"{conflict_distance:.2f} m at x={conflict_tf.location.x:.1f}, y={conflict_tf.location.y:.1f}")
        print(f"[curbside-layout] walker_anchor route_distance: "
              f"{walker_anchor_route_distance:.2f} m at x={walker_anchor_tf.location.x:.1f}, y={walker_anchor_tf.location.y:.1f}")
        for idx, occ_tf in enumerate(spawned_occluder_transforms):
            bp_id = occluder_blueprint_ids[idx] if idx < len(occluder_blueprint_ids) else "?"
            print(f"[curbside-layout] occluder[{idx}]: blueprint={bp_id} "
                  f"at x={occ_tf.location.x:.1f}, y={occ_tf.location.y:.1f}, z={occ_tf.location.z:.1f}")
        # target_start is the (x,y,z) Location where the pedestrian was spawned,
        # computed before the actor existed (so its coords are valid immediately).
        print(f"[curbside-layout] pedestrian start: "
              f"x={target_start.x:.1f}, y={target_start.y:.1f}, z={target_start.z:.1f}")
        print(f"[curbside-layout] pedestrian end:   "
              f"x={target_end.x:.1f}, y={target_end.y:.1f}, z={target_end.z:.1f}")
        if helper_actor is not None and helper_transform is not None:
            print(f"[curbside-layout] helper spawn:    "
                  f"x={helper_transform.location.x:.1f}, y={helper_transform.location.y:.1f}, "
                  f"yaw={helper_transform.rotation.yaw:.0f}")
        # Find baked / non-spawned vehicles within 30 m of conflict for context.
        # CARLA's static decoration "vehicles" don't always appear in world.get_actors()
        # (they may be baked into the level mesh as props rather than spawned actors).
        # Try anyway — anything that DOES register is useful.
        own_ids = {ego.id} | {a.id for a in extra_actors if a is not None}
        try:
            nearby = []
            for actor in world.get_actors().filter("vehicle.*"):
                if actor.id in own_ids:
                    continue
                try:
                    d = float(actor.get_location().distance(conflict_tf.location))
                except Exception:
                    continue
                if 0.01 < d <= 30.0:  # skip uninitialized (0,0,0) actors
                    nearby.append((d, actor))
            nearby.sort(key=lambda kv: kv[0])
            if nearby:
                print(f"[curbside-layout] {len(nearby)} other actor vehicle(s) within 30 m of conflict:")
                for d, actor in nearby[:8]:
                    al = actor.get_location()
                    print(f"  [curbside-layout]   d={d:5.1f} m  type={actor.type_id}  "
                          f"x={al.x:.1f}, y={al.y:.1f}")
            else:
                print("[curbside-layout] no nearby actor vehicles within 30 m "
                      "(baked map decoration vehicles are typically static props, not actors)")
        except (RuntimeError, AttributeError):
            pass
    except Exception as diag_exc:
        print(f"[curbside-layout] diagnostic print failed: {diag_exc}")

    return ego, extra_actors, layout


def spawn_intersection_truck_pedestrian_layout(
    world: "carla.World",
    client: "carla.Client",
    traffic_manager: "carla.TrafficManager",
    candidates: Sequence["carla.Transform"],
    anchor: "carla.Location",
    spec: ScenarioSpec,
    rng: random.Random,
    ego_autopilot: bool,
    route_choice: str,
) -> Tuple["carla.Actor", List["carla.Actor"], Dict[str, object]]:
    occlusion_mode = str(spec.intersection_occlusion_mode)
    requested_route_choice = str(route_choice)
    if occlusion_mode == "right_turn_occluded_failure":
        route_choice = "right"
    ego_sp = choose_ego_spawn_toward_anchor(world, candidates, anchor, spec.ego_distance_m, route_choice)
    start_wp = world.get_map().get_waypoint(
        ego_sp.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    branch_distance = first_route_branch_distance(start_wp, route_choice)
    route = generate_route_waypoints(start_wp, route_choice, total_distance_m=85.0)
    conflict_crosswalk_gap_m: Optional[float] = None
    if occlusion_mode == "right_turn_occluded_failure":
        turn_entry_distance = branch_distance or 20.0
        desired_conflict_distance = turn_entry_distance + 4.0
        crosswalk_candidate = route_crosswalk_candidate(
            world.get_map(),
            route,
            min_distance_m=max(10.0, turn_entry_distance - 3.0),
            max_distance_m=min(42.0, turn_entry_distance + 14.0),
            desired_distance_m=desired_conflict_distance,
        )
        if crosswalk_candidate is not None:
            conflict_distance, conflict_tf, conflict_crosswalk_gap_m = crosswalk_candidate
        else:
            conflict_distance = clamp(desired_conflict_distance, 20.0, 34.0)
            conflict_tf = route_transform_at_distance(route, conflict_distance)
        # Put the stopped truck/van close enough to the crossing sightline to
        # actually hide the pedestrian from the ego camera, while keeping it
        # laterally offset from the ego route so the ego does not simply rear-end it.
        occluder_distance = max(10.0, min(conflict_distance - 5.5, turn_entry_distance + 2.0))
    else:
        fallback_conflict_base_m = 28.0 if occlusion_mode == "occluded_failure" else 18.0
        conflict_distance = clamp((branch_distance or fallback_conflict_base_m) + 11.0, 24.0, 46.0)
        occluder_gap_m = 2.8 if occlusion_mode == "occluded_failure" else 6.0
        occluder_distance = max(12.0, conflict_distance - occluder_gap_m)
        conflict_tf = route_transform_at_distance(route, conflict_distance)
    occluder_tf = route_transform_at_distance(route, occluder_distance)

    ego = spawn_vehicle(
        client,
        world,
        traffic_manager,
        rng.choice(resolve_vehicle_blueprint_ids(world, rng)),
        ego_sp,
        f"{SCENESENSE_ROLE_PREFIX}{spec.name}_ego",
        rng,
        autopilot=ego_autopilot,
    )
    if ego is None:
        raise RuntimeError("Unable to spawn intersection-occlusion ego vehicle.")

    if occlusion_mode == "right_turn_occluded_failure":
        heavy_occluder_exact_ids = (
            "vehicle.sprinter.mercedes",
            "vehicle.mercedes.sprinter",
            "vehicle.volkswagen.t2",
            "vehicle.tesla.cybertruck",
            "vehicle.ambulance.ford",
            "vehicle.ford.ambulance",
            "vehicle.carlacola.actors",
            "vehicle.carlamotors.carlacola",
            "vehicle.firetruck.actors",
            "vehicle.carlamotors.firetruck",
            "vehicle.mitsubishi.fusorosa",
        )
        heavy_occluder_tokens = (
            "ambulance",
            "sprinter",
            "volkswagen.t2",
            "cybertruck",
            "van",
            "bus",
            "truck",
            "carlacola",
            "firetruck",
            "fusorosa",
            "hgv",
        )
    else:
        heavy_occluder_exact_ids = (
            "vehicle.mitsubishi.fusorosa",
            "vehicle.carlamotors.carlacola",
            "vehicle.carlacola.actors",
            "vehicle.carlamotors.firetruck",
            "vehicle.firetruck.actors",
            "vehicle.tesla.cybertruck",
            "vehicle.ford.ambulance",
            "vehicle.ambulance.ford",
            "vehicle.mercedes.sprinter",
            "vehicle.sprinter.mercedes",
            "vehicle.volkswagen.t2",
        )
        heavy_occluder_tokens = (
            "fusorosa",
            "carlacola",
            "firetruck",
            "cybertruck",
            "ambulance",
            "sprinter",
            "volkswagen.t2",
            "truck",
            "bus",
            "van",
            "hgv",
        )
    vehicle_blueprints = sorted(world.get_blueprint_library().filter("vehicle.*"), key=lambda bp: bp.id)
    vehicle_blueprints_by_id = {bp.id: bp for bp in vehicle_blueprints}
    heavy_occluder_candidate_ids = [
        str(bp.id)
        for bp in vehicle_blueprints
        if any(token in str(bp.id).lower() for token in heavy_occluder_tokens)
    ]
    occluder_blueprint_options: List["carla.ActorBlueprint"] = []
    occluder_bp = None
    for bp_id in heavy_occluder_exact_ids:
        if bp_id in vehicle_blueprints_by_id:
            occluder_bp = vehicle_blueprints_by_id[bp_id]
            occluder_blueprint_options.append(occluder_bp)
            break
    for token in heavy_occluder_tokens:
        occluder_blueprint_options.extend(
            [bp for bp in vehicle_blueprints if token in str(bp.id).lower()]
        )
    for bp_id in SAFE_VEHICLE_BLUEPRINTS:
        if bp_id in vehicle_blueprints_by_id:
            occluder_blueprint_options.append(vehicle_blueprints_by_id[bp_id])
    occluder_blueprint_options.extend(vehicle_blueprints[:5])
    occluder_blueprint_options = unique_blueprints_by_id(occluder_blueprint_options)
    if occluder_bp is None and occluder_blueprint_options:
        occluder_bp = occluder_blueprint_options[0]
    if occluder_bp is None:
        occluder_bp = find_first_blueprint(world, ("vehicle.lincoln.mkz",), "vehicle.*")
        occluder_blueprint_options = [occluder_bp]
    occluder_is_heavy = any(token in str(occluder_bp.id).lower() for token in heavy_occluder_tokens)
    occluders: List["carla.Actor"] = []
    occluder_spawn_infos: List[Dict[str, object]] = []
    occluder_blueprint_ids: List[str] = []
    planned_occluder_transforms: List["carla.Transform"] = []
    spawned_occluder_transforms: List["carla.Transform"] = []
    route_yaw = float(occluder_tf.rotation.yaw)
    curb_side = -1.0 if route_choice == "left" else 1.0
    if occlusion_mode == "right_turn_occluded_failure":
        occluder_lateral_offset_m = curb_side * 4.6
        target_start_lateral_offset_m = curb_side * 6.4
        # Keep the occluder on the straight approach before the right turn so
        # the truck reads as a stopped vehicle queue, not a vehicle parked at
        # an odd angle in the intersection.
        occluder_yaw = route_yaw
        if occluder_is_heavy:
            occluder_offsets = (
                (0.0, occluder_lateral_offset_m),
            )
        else:
            occluder_offsets = (
                (-10.8, occluder_lateral_offset_m),
                (-5.3, occluder_lateral_offset_m),
                (0.2, occluder_lateral_offset_m),
            )
    elif occlusion_mode == "occluded_failure":
        occluder_lateral_offset_m = curb_side * 6.2
        target_start_lateral_offset_m = curb_side * 8.0
        # Vehicles are stopped in a queue before the crossing line. They are
        # parallel to traffic flow and block the ego camera without forcing the
        # pedestrian to walk through a vehicle body.
        occluder_yaw = route_yaw
        if occluder_is_heavy:
            occluder_offsets = (
                (-6.0, occluder_lateral_offset_m),
                (0.2, occluder_lateral_offset_m),
            )
        else:
            occluder_offsets = (
                (-10.8, occluder_lateral_offset_m),
                (-5.3, occluder_lateral_offset_m),
                (0.2, occluder_lateral_offset_m),
            )
    else:
        occluder_lateral_offset_m = curb_side * 7.2
        target_start_lateral_offset_m = curb_side * 5.8
        occluder_yaw = route_yaw
        occluder_offsets = (
            (0.0, occluder_lateral_offset_m),
            (6.2, occluder_lateral_offset_m),
        )
    for index, (forward_m, right_m) in enumerate(occluder_offsets):
        if occlusion_mode in {"occluded_failure", "right_turn_occluded_failure"}:
            queue_position = "front" if index == len(occluder_offsets) - 1 else f"tail_{index}"
            role = f"{SCENESENSE_ROLE_PREFIX}{spec.name}_stopped_queue_{queue_position}"
        else:
            role = (
                f"{SCENESENSE_ROLE_PREFIX}{spec.name}_parked_truck"
                if index == 0
                else f"{SCENESENSE_ROLE_PREFIX}{spec.name}_parked_truck_tail"
            )
        transform = carla.Transform(
            offset_location(occluder_tf.location, route_yaw, forward_m=forward_m, right_m=right_m, z_offset_m=0.4),
            carla.Rotation(pitch=0.0, yaw=occluder_yaw, roll=0.0),
        )
        planned_occluder_transforms.append(transform)
        if occlusion_mode == "right_turn_occluded_failure":
            forward_nudges = (0.0, -1.0, 1.0, -2.0, 2.0)
            right_nudges = (
                0.0,
                curb_side * 0.8,
                curb_side * 1.6,
                curb_side * 2.4,
                -curb_side * 0.8,
            )
            z_nudges = (0.0,)
        else:
            forward_nudges = (0.0, -0.8, 0.8)
            right_nudges = (0.0, -0.6, 0.6)
            z_nudges = (0.0, 0.3)
        actor, used_bp, used_transform, spawn_info = try_spawn_configured_actor_variants(
            world,
            occluder_blueprint_options,
            transform,
            role,
            nudge_yaw_deg=route_yaw,
            forward_nudges_m=forward_nudges,
            right_nudges_m=right_nudges,
            z_nudges_m=z_nudges,
        )
        spawn_info.update(
            {
                "role": role,
                "planned_forward_m": float(forward_m),
                "planned_right_m": float(right_m),
            }
        )
        occluder_spawn_infos.append(spawn_info)
        if actor is not None:
            try:
                actor.set_simulate_physics(False)
            except RuntimeError:
                pass
            occluders.append(actor)
            if used_bp is not None:
                occluder_blueprint_ids.append(str(used_bp.id))
            if used_transform is not None:
                spawned_occluder_transforms.append(used_transform)

    target_yaw = float(conflict_tf.rotation.yaw)
    target_start = offset_location(
        conflict_tf.location,
        target_yaw,
        forward_m=0.0,
        right_m=target_start_lateral_offset_m,
        z_offset_m=1.0,
    )
    target_end = offset_location(
        conflict_tf.location,
        target_yaw,
        forward_m=0.0,
        right_m=0.0,
        z_offset_m=1.0,
    )
    target_transform = carla.Transform(
        target_start,
        carla.Rotation(pitch=0.0, yaw=target_yaw - 90.0, roll=0.0),
    )
    walker_bp = configure_walker_blueprint(
        find_first_blueprint(
            world,
            ("walker.pedestrian.0001", "walker.pedestrian.0010", "walker.pedestrian.0020"),
            "walker.pedestrian.*",
        )
    )
    target, _target_bp, spawned_target_transform, target_spawn_info = try_spawn_configured_actor_variants(
        world,
        [walker_bp],
        target_transform,
        f"{SCENESENSE_ROLE_PREFIX}{spec.name}_hidden_pedestrian",
        nudge_yaw_deg=target_yaw,
        forward_nudges_m=(0.0, -0.8, 0.8, -1.6, 1.6),
        right_nudges_m=(0.0, -0.8, 0.8, -1.6, 1.6),
        z_nudges_m=(0.0, 0.3, 0.6),
    )
    if spawned_target_transform is not None:
        target_transform = spawned_target_transform

    if not occluders or target is None:
        for actor in [target, *occluders, ego]:
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        raise RuntimeError(
            "Unable to spawn required intersection truck/pedestrian occlusion actors "
            f"(occluders_spawned={len(occluders)}/{len(occluder_offsets)}, "
            f"target_spawned={target is not None}, "
            f"occluder_candidates={[str(bp.id) for bp in occluder_blueprint_options]}, "
            f"occluder_spawn_infos={occluder_spawn_infos}, "
            f"target_spawn_info={target_spawn_info})."
        )

    observer_target = carla.Location(
        x=(float(conflict_tf.location.x) + float(occluder_tf.location.x) + float(target_start.x)) / 3.0,
        y=(float(conflict_tf.location.y) + float(occluder_tf.location.y) + float(target_start.y)) / 3.0,
        z=float(conflict_tf.location.z + 1.0),
    )
    if occlusion_mode == "right_turn_occluded_failure":
        observer_camera_location = offset_location(
            observer_target,
            route_yaw,
            forward_m=-24.0,
            right_m=-curb_side * 18.0,
            z_offset_m=23.0,
        )
        observer_spectator_transform = carla.Transform(
            observer_camera_location,
            look_at_rotation(observer_camera_location, observer_target),
        )
    else:
        observer_spectator_transform = None

    extra_actors = [*occluders, target]
    layout = {
        "type": spec.name,
        "ego_actor_id": int(ego.id),
        "occluder_actor_ids": [int(actor.id) for actor in occluders],
        "occluder_primary_actor_id": int(occluders[-1].id),
        "occluder_blueprint_id": occluder_blueprint_ids[-1] if occluder_blueprint_ids else str(occluder_bp.id),
        "occluder_blueprint_ids": occluder_blueprint_ids,
        "occluder_heavy_blueprint_available": bool(occluder_is_heavy),
        "occluder_heavy_candidate_ids": heavy_occluder_candidate_ids,
        "occluder_spawn_infos": occluder_spawn_infos,
        "occluder_simulate_physics": False,
        "occluder_requested_count": int(len(occluder_offsets)),
        "occluder_forward_offsets_m": [float(item[0]) for item in occluder_offsets],
        "occluder_lateral_offset_m": float(occluder_lateral_offset_m),
        "target_start_lateral_offset_m": float(target_start_lateral_offset_m),
        "target_actor_id": int(target.id),
        "target_role": f"{SCENESENSE_ROLE_PREFIX}{spec.name}_hidden_pedestrian",
        "ego_spawn_transform": transform_to_dict(ego_sp),
        "occlusion_mode": occlusion_mode,
        "requested_route_choice": requested_route_choice,
        "effective_route_choice": route_choice,
        "branch_distance_m": branch_distance,
        "conflict_distance_m": conflict_distance,
        "conflict_crosswalk_gap_m": conflict_crosswalk_gap_m,
        "conflict_location": location_to_dict(conflict_tf.location),
        "observer_location": location_to_dict(
            carla.Location(
                x=float(conflict_tf.location.x),
                y=float(conflict_tf.location.y),
                z=float(conflict_tf.location.z + 1.0),
            )
        ),
        "observer_spectator_transform": None
        if observer_spectator_transform is None
        else transform_to_dict(observer_spectator_transform),
        "occluder_distance_m": occluder_distance,
        "occlusion_curb_side": "negative_right_vector" if curb_side < 0 else "positive_right_vector",
        "occluder_planned_transforms": [transform_to_dict(transform) for transform in planned_occluder_transforms],
        "occluder_spawned_transforms": [transform_to_dict(transform) for transform in spawned_occluder_transforms],
        "occluder_actor_transforms_at_spawn": [transform_to_dict(actor.get_transform()) for actor in occluders],
        "target_spawn_info": target_spawn_info,
        "target_start_transform": transform_to_dict(target_transform),
        "target_crossing_end_location": location_to_dict(target_end),
        "target_crossing_trigger_location": location_to_dict(conflict_tf.location),
        "route_choice": route_choice,
        "controller_route_transforms": [transform_to_dict(waypoint.transform) for waypoint in route],
        "route_waypoints": [
            transform_to_dict(waypoint.transform)
            for waypoint in route[:: max(1, len(route) // 14)]
        ],
        "pole_observer": {
            "mount": "traffic_light_pole_or_intersection_edge",
            "anchor_traffic_light_id": str(spec.traffic_light_id),
            "anchor_location": location_to_dict(anchor),
            "purpose": "Elevated view should see behind the parked truck before the ego camera can.",
        },
        "design_note": (
            "Intersection crossing failure: ego follows a scripted turn while a pedestrian crosses into its path. "
            "The visible_control mode is a positive-control failure case; occluded_failure places a stopped "
            "vehicle queue between the ego camera and pedestrian start to create the hidden-hazard variant. "
            "right_turn_occluded_failure forces a right turn and keeps the occluder on the straight approach "
            "before the corner, matching a turn-yield-to-pedestrian blind spot. The right-turn occluder should "
            "be interpreted as a stopped queue or service vehicle in an adjacent lane, not as curb parking. "
            "If this CARLA build does not expose a bus/truck/van blueprint, the layout requests a denser "
            "multi-car stopped queue and records that fallback in occluder_heavy_blueprint_available."
        ),
    }
    return ego, extra_actors, layout


def choose_occlusion_pair(
    candidates: Sequence["carla.Transform"],
    anchor: "carla.Location",
    used_locations: Sequence["carla.Location"],
) -> Tuple[Optional["carla.Transform"], Optional["carla.Transform"]]:
    best: Optional[Tuple[float, "carla.Transform", "carla.Transform"]] = None
    filtered = [
        sp
        for sp in candidates
        if 10.0 <= sp.location.distance(anchor) <= 60.0
        and all(sp.location.distance(loc) > 7.5 for loc in used_locations)
    ]
    for occluder in filtered:
        occ_dist = occluder.location.distance(anchor)
        occ_bearing = vector_bearing_deg(anchor, occluder.location)
        for target in filtered:
            if target is occluder:
                continue
            target_dist = target.location.distance(anchor)
            if target_dist <= occ_dist + 8.0:
                continue
            if target.location.distance(occluder.location) < 8.0:
                continue
            bearing_delta = angular_difference_deg(occ_bearing, vector_bearing_deg(anchor, target.location))
            if bearing_delta > 35.0:
                continue
            score = bearing_delta + abs(occ_dist - 18.0) * 0.4 + abs(target_dist - 34.0) * 0.2
            if best is None or score < best[0]:
                best = (score, occluder, target)
    if best is None:
        return None, None
    return best[1], best[2]


def spawn_background_vehicles(
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    candidates: Sequence["carla.Transform"],
    anchor: "carla.Location",
    spec: ScenarioSpec,
    count: int,
    rng: random.Random,
    used_locations: List["carla.Location"],
    autopilot: bool,
) -> List["carla.Actor"]:
    blueprint_ids = resolve_vehicle_blueprint_ids(world, rng)
    spawned: List["carla.Actor"] = []

    if spec.occlusion_pair and count >= 2:
        occluder_sp, target_sp = choose_occlusion_pair(candidates, anchor, used_locations)
        for role_suffix, sp in (("occluder_vehicle", occluder_sp), ("target_vehicle", target_sp)):
            if sp is None:
                continue
            actor = spawn_vehicle(
                client,
                world,
                traffic_manager,
                rng.choice(blueprint_ids),
                sp,
                f"{SCENESENSE_ROLE_PREFIX}{spec.name}_{role_suffix}",
                rng,
                autopilot=False,
            )
            if actor is not None:
                spawned.append(actor)
                used_locations.append(actor.get_location())

    for sp in candidates:
        if len(spawned) >= count:
            break
        if all(sp.location.distance(loc) >= 8.0 for loc in used_locations):
            actor = spawn_vehicle(
                client,
                world,
                traffic_manager,
                rng.choice(blueprint_ids),
                sp,
                f"{SCENESENSE_ROLE_PREFIX}{spec.name}_background_vehicle",
                rng,
                autopilot=autopilot,
            )
            if actor is not None:
                spawned.append(actor)
                used_locations.append(actor.get_location())
    return spawned


def resolve_walker_speed(blueprint: "carla.ActorBlueprint") -> float:
    if blueprint.has_attribute("speed"):
        values = list(blueprint.get_attribute("speed").recommended_values)
        if len(values) >= 2:
            return float(values[1])
        if values:
            return float(values[-1])
    return 1.2


def pedestrian_spawn_points_near(
    world: "carla.World",
    anchor: "carla.Location",
    count: int,
    radius_m: float,
    rng: random.Random,
) -> List["carla.Transform"]:
    points: List["carla.Transform"] = []
    attempts = max(count * 20, 60)
    for _ in range(attempts):
        if len(points) >= count:
            break
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if loc.distance(anchor) > radius_m:
            continue
        if any(loc.distance(existing.location) < 1.5 for existing in points):
            continue
        yaw = rng.uniform(-180.0, 180.0)
        points.append(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0), carla.Rotation(yaw=yaw)))
    return points


def spawn_pedestrians(
    client: "carla.Client",
    world: "carla.World",
    anchor: "carla.Location",
    spec: ScenarioSpec,
    count: int,
    rng: random.Random,
    move: bool,
) -> Tuple[List["carla.Actor"], List["carla.Actor"]]:
    if count <= 0:
        return [], []
    blueprints = choose_walker_blueprints(world)
    if not blueprints:
        return [], []

    spawn_points = pedestrian_spawn_points_near(world, anchor, count, spec.anchor_radius_m, rng)
    walker_batch = []
    speeds: List[float] = []
    for sp in spawn_points:
        blueprint = configure_walker_blueprint(world.get_blueprint_library().find(rng.choice(blueprints).id))
        walker_batch.append(carla.command.SpawnActor(blueprint, sp))
        speeds.append(resolve_walker_speed(blueprint))

    walker_ids: List[int] = []
    walker_speeds: List[float] = []
    if walker_batch:
        responses = client.apply_batch_sync(walker_batch, True)
        for response, speed in zip(responses, speeds):
            if not response.error:
                walker_ids.append(response.actor_id)
                walker_speeds.append(speed)

    walkers = [world.get_actor(actor_id) for actor_id in walker_ids]
    walkers = [actor for actor in walkers if actor is not None]
    if not move or not walkers:
        return walkers, []

    controller_blueprint = world.get_blueprint_library().find("controller.ai.walker")
    controller_batch = [
        carla.command.SpawnActor(controller_blueprint, carla.Transform(), walker.id)
        for walker in walkers
    ]
    controller_ids: List[int] = []
    responses = client.apply_batch_sync(controller_batch, True)
    for response in responses:
        if not response.error:
            controller_ids.append(response.actor_id)

    controllers = [world.get_actor(actor_id) for actor_id in controller_ids]
    controllers = [actor for actor in controllers if actor is not None]
    for controller, speed in zip(controllers, walker_speeds):
        controller.start()
        destination = world.get_random_location_from_navigation()
        if destination is not None:
            controller.go_to_location(destination)
        controller.set_max_speed(speed)
    return walkers, controllers


def extent_to_dict(extent: "carla.Vector3D") -> Dict[str, float]:
    return {"x": float(extent.x), "y": float(extent.y), "z": float(extent.z)}


def actor_record(actor: "carla.Actor", role: str) -> Dict[str, object]:
    bbox = getattr(actor, "bounding_box", None)
    return {
        "id": int(actor.id),
        "type_id": str(actor.type_id),
        "role": role,
        "transform": transform_to_dict(actor.get_transform()),
        "location": location_to_dict(actor.get_location()),
        "bounding_box": None
        if bbox is None
        else {
            "location": location_to_dict(bbox.location),
            "extent": extent_to_dict(bbox.extent),
        },
        "attributes": {str(key): str(value) for key, value in actor.attributes.items()},
    }


def weather_to_dict(weather: "carla.WeatherParameters") -> Dict[str, float]:
    fields = (
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wind_intensity",
        "sun_azimuth_angle",
        "sun_altitude_angle",
        "fog_density",
        "fog_distance",
        "wetness",
    )
    result: Dict[str, float] = {}
    for field in fields:
        if hasattr(weather, field):
            result[field] = float(getattr(weather, field))
    return result


def suggested_sensor_placements(anchor: "carla.Location", traffic_light_id: str) -> List[Dict[str, object]]:
    return [
        {
            "name": "fusion_tl_14",
            "mount": "traffic_light_pole_candidate",
            "traffic_light_id": str(traffic_light_id),
            "camera_args": {
                "camera_x": 9.0,
                "camera_y": 2.0,
                "camera_pitch": -30.0,
                "camera_yaw_offset": 50.0,
                "camera_roll": 0.0,
                "camera_fov": 100.0,
            },
            "anchor_location": location_to_dict(anchor),
        },
        {
            "name": "fusion_tl_14_view_2",
            "mount": "traffic_light_pole_candidate",
            "traffic_light_id": str(traffic_light_id),
            "camera_args": {
                "camera_x": 11.0,
                "camera_y": 2.0,
                "camera_pitch": -30.0,
                "camera_yaw_offset": 120.0,
                "camera_roll": 0.0,
                "camera_fov": 100.0,
            },
            "anchor_location": location_to_dict(anchor),
        },
        {
            "name": "parked_ego_front",
            "mount": "ego_vehicle_candidate",
            "camera_transform_relative": {
                "location": {"x": 1.8, "y": 0.0, "z": 1.6},
                "rotation": {"pitch": -5.0, "yaw": 0.0, "roll": 0.0},
            },
            "radar_transform_relative": {
                "location": {"x": 2.0, "y": 0.0, "z": 1.0},
                "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            },
        },
    ]


class EgoSensorMonitor:
    """Small ego RGB/radar smoke-test helper for scenario inspection."""

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Actor",
        camera_width: int,
        camera_height: int,
        camera_fov: float,
        radar_range: float,
        radar_hfov: float,
        radar_vfov: float,
        radar_pps: int,
        preview: bool,
        evidence_capture: bool = False,
        evidence_buffer_size: int = 80,
        evidence_stride: int = 2,
    ) -> None:
        self.world = world
        self.ego = ego
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_fov = float(camera_fov)
        self.radar_range = float(radar_range)
        self.radar_hfov = float(radar_hfov)
        self.radar_vfov = float(radar_vfov)
        self.radar_pps = int(radar_pps)
        self.preview_requested = bool(preview)
        self.preview = bool(preview)
        self.evidence_capture = bool(evidence_capture)
        self.evidence_buffer_size = max(1, int(evidence_buffer_size))
        self.evidence_stride = max(1, int(evidence_stride))
        self.sensors: List["carla.Actor"] = []
        self.camera_queue: "queue.Queue[object]" = queue.Queue(maxsize=2)
        self.evidence_frames = deque(maxlen=self.evidence_buffer_size)
        self.camera_frames = 0
        self.radar_frames = 0
        self.radar_points_total = 0
        self.last_camera_frame: Optional[int] = None
        self.last_radar_frame: Optional[int] = None
        self.started_at = time.time()
        self.cv2 = None
        self.np = None
        self.preview_error: Optional[str] = None

        if self.preview:
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore

                self.cv2 = cv2
                self.np = np
            except Exception as exc:
                self.preview = False
                self.preview_error = f"OpenCV preview disabled: {exc}"

    def spawn(self) -> None:
        bp_lib = self.world.get_blueprint_library()

        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.camera_height))
        camera_bp.set_attribute("fov", str(self.camera_fov))
        camera_transform = carla.Transform(
            carla.Location(x=1.8, y=0.0, z=1.6),
            carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
        )
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego)
        camera.listen(self._on_camera)
        self.sensors.append(camera)

        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", str(self.radar_hfov))
        radar_bp.set_attribute("vertical_fov", str(self.radar_vfov))
        radar_bp.set_attribute("range", str(self.radar_range))
        radar_bp.set_attribute("points_per_second", str(self.radar_pps))
        radar_transform = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.0),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego)
        radar.listen(self._on_radar)
        self.sensors.append(radar)

    def _on_camera(self, image: object) -> None:
        self.camera_frames += 1
        self.last_camera_frame = int(getattr(image, "frame", -1))
        if self.evidence_capture and self.camera_frames % self.evidence_stride == 0:
            self.evidence_frames.append(
                {
                    "frame": self.last_camera_frame,
                    "timestamp": float(getattr(image, "timestamp", -1.0)),
                    "width": int(getattr(image, "width")),
                    "height": int(getattr(image, "height")),
                    "raw_data": bytes(getattr(image, "raw_data")),
                }
            )
        try:
            if self.camera_queue.full():
                self.camera_queue.get_nowait()
            self.camera_queue.put_nowait(image)
        except queue.Full:
            pass

    def _on_radar(self, measurement: object) -> None:
        self.radar_frames += 1
        self.last_radar_frame = int(getattr(measurement, "frame", -1))
        try:
            points = len(measurement)  # type: ignore[arg-type]
        except TypeError:
            points = sum(1 for _ in measurement)  # type: ignore[operator]
        self.radar_points_total += int(points)

    def poll_preview(self) -> bool:
        if not self.preview or self.cv2 is None or self.np is None:
            return True
        try:
            image = self.camera_queue.get_nowait()
        except queue.Empty:
            key = self.cv2.waitKey(1) & 0xFF
            return key not in (27, ord("q"))

        array = self.np.frombuffer(image.raw_data, dtype=self.np.uint8)  # type: ignore[attr-defined]
        frame = array.reshape((image.height, image.width, 4))[:, :, :3].copy()  # type: ignore[attr-defined]
        self.cv2.putText(
            frame,
            f"ego RGB frame={self.last_camera_frame} radar_frames={self.radar_frames}",
            (12, 24),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )
        self.cv2.imshow("SceneSense Ego RGB Preview", frame)
        key = self.cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def metadata(self) -> Dict[str, object]:
        return {
            "enabled": True,
            "ego_actor_id": int(self.ego.id),
            "camera": {
                "type_id": "sensor.camera.rgb",
                "width": self.camera_width,
                "height": self.camera_height,
                "fov": self.camera_fov,
                "transform_relative": {
                    "location": {"x": 1.8, "y": 0.0, "z": 1.6},
                    "rotation": {"pitch": -5.0, "yaw": 0.0, "roll": 0.0},
                },
            },
            "radar": {
                "type_id": "sensor.other.radar",
                "range": self.radar_range,
                "horizontal_fov": self.radar_hfov,
                "vertical_fov": self.radar_vfov,
                "points_per_second": self.radar_pps,
                "transform_relative": {
                    "location": {"x": 2.0, "y": 0.0, "z": 1.0},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
            },
            "preview_requested": bool(self.preview_requested),
            "preview_active": bool(self.preview),
            "preview_error": self.preview_error,
            "evidence_capture": bool(self.evidence_capture),
            "evidence_buffer_size": int(self.evidence_buffer_size),
            "evidence_stride": int(self.evidence_stride),
        }

    def summary(self) -> Dict[str, object]:
        elapsed = max(1e-9, time.time() - self.started_at)
        return {
            "ego_actor_id": int(self.ego.id),
            "elapsed_s": elapsed,
            "camera_frames": int(self.camera_frames),
            "camera_fps": float(self.camera_frames / elapsed),
            "last_camera_frame": self.last_camera_frame,
            "radar_frames": int(self.radar_frames),
            "radar_fps": float(self.radar_frames / elapsed),
            "last_radar_frame": self.last_radar_frame,
            "radar_points_total": int(self.radar_points_total),
            "radar_points_per_frame_avg": float(
                self.radar_points_total / max(1, self.radar_frames)
            ),
            "evidence_frames_buffered": int(len(self.evidence_frames)),
        }

    def write_summary(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ego_sensor_summary.json").write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_evidence_frames(self, evidence_dir: Path, label: str = "ego") -> Dict[str, object]:
        frames_dir = evidence_dir / f"{sanitize_token(label)}_rgb_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        index_rows: List[Dict[str, object]] = []
        for idx, record in enumerate(self.evidence_frames):
            frame_id = int(record["frame"])
            filename = write_bgra_image(record, frames_dir / f"{idx:04d}_frame_{frame_id}")
            index_rows.append(
                {
                    "camera_label": label,
                    "sequence_index": idx,
                    "frame": frame_id,
                    "timestamp": record.get("timestamp", ""),
                    "width": record.get("width", ""),
                    "height": record.get("height", ""),
                    "file": f"{frames_dir.name}/{filename}",
                }
            )
        if index_rows:
            with (evidence_dir / f"{sanitize_token(label)}_rgb_frame_index.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()))
                writer.writeheader()
                writer.writerows(index_rows)
        return {
            "camera_label": label,
            "frames_dir": frames_dir.name,
            "frames_written": int(len(index_rows)),
            "frame_index_file": f"{sanitize_token(label)}_rgb_frame_index.csv" if index_rows else None,
        }

    def destroy(self) -> None:
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception:
                pass
        for sensor in reversed(self.sensors):
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass
        if self.cv2 is not None:
            try:
                self.cv2.destroyWindow("SceneSense Ego RGB Preview")
            except Exception:
                pass


class ActorCameraMonitor:
    """Small RGB preview helper for non-ego observer actors."""

    def __init__(
        self,
        world: "carla.World",
        actor: "carla.Actor",
        label: str,
        window_name: str,
        camera_width: int,
        camera_height: int,
        camera_fov: float,
        preview: bool,
        evidence_capture: bool = False,
        evidence_buffer_size: int = 80,
        evidence_stride: int = 2,
    ) -> None:
        self.world = world
        self.actor = actor
        self.label = str(label)
        self.window_name = str(window_name)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_fov = float(camera_fov)
        self.preview_requested = bool(preview)
        self.preview = bool(preview)
        self.evidence_capture = bool(evidence_capture)
        self.evidence_buffer_size = max(1, int(evidence_buffer_size))
        self.evidence_stride = max(1, int(evidence_stride))
        self.sensors: List["carla.Actor"] = []
        self.camera_queue: "queue.Queue[object]" = queue.Queue(maxsize=2)
        self.evidence_frames = deque(maxlen=self.evidence_buffer_size)
        self.camera_frames = 0
        self.last_camera_frame: Optional[int] = None
        self.started_at = time.time()
        self.cv2 = None
        self.np = None
        self.preview_error: Optional[str] = None

        if self.preview:
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore

                self.cv2 = cv2
                self.np = np
            except Exception as exc:
                self.preview = False
                self.preview_error = f"{self.label} preview disabled: {exc}"

    def spawn(self) -> None:
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.camera_height))
        camera_bp.set_attribute("fov", str(self.camera_fov))
        camera_transform = carla.Transform(
            carla.Location(x=1.8, y=0.0, z=1.55),
            carla.Rotation(pitch=-4.0, yaw=0.0, roll=0.0),
        )
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.actor)
        camera.listen(self._on_camera)
        self.sensors.append(camera)

    def _on_camera(self, image: object) -> None:
        self.camera_frames += 1
        self.last_camera_frame = int(getattr(image, "frame", -1))
        if self.evidence_capture and self.camera_frames % self.evidence_stride == 0:
            self.evidence_frames.append(
                {
                    "frame": self.last_camera_frame,
                    "timestamp": float(getattr(image, "timestamp", -1.0)),
                    "width": int(getattr(image, "width")),
                    "height": int(getattr(image, "height")),
                    "raw_data": bytes(getattr(image, "raw_data")),
                }
            )
        try:
            if self.camera_queue.full():
                self.camera_queue.get_nowait()
            self.camera_queue.put_nowait(image)
        except queue.Full:
            pass

    def poll_preview(self) -> bool:
        if not self.preview or self.cv2 is None or self.np is None:
            return True
        try:
            image = self.camera_queue.get_nowait()
        except queue.Empty:
            key = self.cv2.waitKey(1) & 0xFF
            return key not in (27, ord("q"))

        array = self.np.frombuffer(image.raw_data, dtype=self.np.uint8)  # type: ignore[attr-defined]
        frame = array.reshape((image.height, image.width, 4))[:, :, :3].copy()  # type: ignore[attr-defined]
        self.cv2.putText(
            frame,
            f"{self.label} frame={self.last_camera_frame}",
            (12, 24),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )
        self.cv2.imshow(self.window_name, frame)
        key = self.cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def metadata(self) -> Dict[str, object]:
        return {
            "enabled": True,
            "actor_id": int(self.actor.id),
            "label": self.label,
            "camera": {
                "type_id": "sensor.camera.rgb",
                "width": self.camera_width,
                "height": self.camera_height,
                "fov": self.camera_fov,
                "transform_relative": {
                    "location": {"x": 1.8, "y": 0.0, "z": 1.55},
                    "rotation": {"pitch": -4.0, "yaw": 0.0, "roll": 0.0},
                },
            },
            "preview_requested": bool(self.preview_requested),
            "preview_active": bool(self.preview),
            "preview_error": self.preview_error,
            "evidence_capture": bool(self.evidence_capture),
            "evidence_buffer_size": int(self.evidence_buffer_size),
            "evidence_stride": int(self.evidence_stride),
        }

    def summary(self) -> Dict[str, object]:
        elapsed = max(1e-9, time.time() - self.started_at)
        return {
            "actor_id": int(self.actor.id),
            "label": self.label,
            "elapsed_s": elapsed,
            "camera_frames": int(self.camera_frames),
            "camera_fps": float(self.camera_frames / elapsed),
            "last_camera_frame": self.last_camera_frame,
            "evidence_frames_buffered": int(len(self.evidence_frames)),
        }

    def write_summary(self, out_dir: Path, filename: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / filename).write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_evidence_frames(self, evidence_dir: Path, label: Optional[str] = None) -> Dict[str, object]:
        camera_label = sanitize_token(label or self.label)
        frames_dir = evidence_dir / f"{camera_label}_rgb_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        index_rows: List[Dict[str, object]] = []
        for idx, record in enumerate(self.evidence_frames):
            frame_id = int(record["frame"])
            filename = write_bgra_image(record, frames_dir / f"{idx:04d}_frame_{frame_id}")
            index_rows.append(
                {
                    "camera_label": camera_label,
                    "sequence_index": idx,
                    "frame": frame_id,
                    "timestamp": record.get("timestamp", ""),
                    "width": record.get("width", ""),
                    "height": record.get("height", ""),
                    "file": f"{frames_dir.name}/{filename}",
                }
            )
        index_file = f"{camera_label}_rgb_frame_index.csv"
        if index_rows:
            with (evidence_dir / index_file).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()))
                writer.writeheader()
                writer.writerows(index_rows)
        return {
            "camera_label": camera_label,
            "frames_dir": frames_dir.name,
            "frames_written": int(len(index_rows)),
            "frame_index_file": index_file if index_rows else None,
        }

    def destroy(self) -> None:
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception:
                pass
        for sensor in reversed(self.sensors):
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass
        if self.cv2 is not None:
            try:
                self.cv2.destroyWindow(self.window_name)
            except Exception:
                pass


class HelperVehicleController:
    """Simple scripted controller for an opposite-lane helper vehicle."""

    def __init__(
        self,
        actor: "carla.Actor",
        target_location: "carla.Location",
        target_speed: float,
        stop_distance_m: float,
        gate_predicate: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.actor = actor
        self.target_location = copy_location(target_location)
        self.target_speed = max(0.1, float(target_speed))
        self.stop_distance_m = max(0.0, float(stop_distance_m))
        self.start_location = copy_location(actor.get_location())
        self.last_location = copy_location(self.start_location)
        self.ticks = 0
        self.last_distance_m = float(self.start_location.distance(self.target_location))
        self.min_distance_m = self.last_distance_m
        self.stopped = False
        # Optional gate: when set, helper holds the brake (does not drive)
        # until the predicate returns True. Used by --helper-pause-until-crossing
        # so the helper sits stationary and only starts driving once the
        # pedestrian's crossing actually fires.
        self.gate_predicate = gate_predicate
        self.released = gate_predicate is None

    def tick(self) -> None:
        if self.actor is None:
            return
        self.ticks += 1
        # Hold the brake until the gate predicate releases us.
        if not self.released:
            if self.gate_predicate is not None and self.gate_predicate():
                self.released = True
            else:
                try:
                    self.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                except RuntimeError:
                    pass
                return
        loc = self.actor.get_location()
        self.last_location = copy_location(loc)
        distance = float(loc.distance(self.target_location))
        self.last_distance_m = distance
        self.min_distance_m = min(self.min_distance_m, distance)
        if distance <= self.stop_distance_m:
            self.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            self.stopped = True
            return

        # Off-road safety brake: if the helper has drifted more than 3 m off
        # the nearest drivable lane (happens on curved roads where straight-
        # line steering cuts the corner), brake hard and stop. This is a
        # safety net; we keep the original straight-line steering as the
        # primary mode because waypoint following at intersections was
        # picking turn-into-side-street branches and making things worse.
        try:
            world = self.actor.get_world()
            wp = world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None or wp.transform.location.distance(loc) > 3.0:
                self.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                self.stopped = True
                return
        except RuntimeError:
            pass

        transform = self.actor.get_transform()
        target_yaw = math.degrees(math.atan2(self.target_location.y - loc.y, self.target_location.x - loc.x))
        yaw_error = signed_angular_difference_deg(target_yaw, float(transform.rotation.yaw))
        steer = clamp(yaw_error / 45.0, -0.45, 0.45)
        velocity = self.actor.get_velocity()
        speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
        throttle = 0.35 if speed < self.target_speed else 0.0
        brake = 0.0 if speed < self.target_speed * 1.2 else 0.2
        self.actor.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    def stop(self) -> None:
        try:
            self.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass

    def summary(self) -> Dict[str, object]:
        return {
            "actor_id": int(self.actor.id),
            "target_speed_mps": float(self.target_speed),
            "stop_distance_m": float(self.stop_distance_m),
            "ticks": int(self.ticks),
            "stopped": bool(self.stopped),
            "start_location": location_to_dict(self.start_location),
            "last_location": location_to_dict(self.last_location),
            "target_location": location_to_dict(self.target_location),
            "last_distance_to_target_m": float(self.last_distance_m),
            "min_distance_to_target_m": float(self.min_distance_m),
            "distance_traveled_m": float(self.start_location.distance(self.last_location)),
        }

    def write_summary(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "helper_vehicle_summary.json").write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class ScenarioEvidenceRecorder:
    """Record actor ground truth and export a compact evidence pack."""

    def __init__(
        self,
        world: "carla.World",
        actors: Sequence["carla.Actor"],
        event_monitor: Optional["OcclusionEventMonitor"],
        window_s: float,
    ) -> None:
        self.world = world
        self.actors = list(actors)
        self.event_monitor = event_monitor
        self.window_s = max(0.0, float(window_s))
        self.started_at = time.time()
        self.rows: List[Dict[str, object]] = []

    def tick(self) -> None:
        elapsed_s = self._elapsed_s()
        try:
            frame = int(self.world.get_snapshot().frame)
        except Exception:
            frame = -1
        for actor in self.actors:
            if actor is None:
                continue
            try:
                if not actor.is_alive:
                    continue
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                angular_velocity = actor.get_angular_velocity()
                acceleration = actor.get_acceleration()
            except Exception:
                continue
            self.rows.append(
                {
                    "elapsed_s": float(elapsed_s),
                    "frame": frame,
                    "actor_id": int(actor.id),
                    "type_id": str(actor.type_id),
                    "role_name": str(actor.attributes.get("role_name", "")),
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                    "pitch": float(transform.rotation.pitch),
                    "yaw": float(transform.rotation.yaw),
                    "roll": float(transform.rotation.roll),
                    "velocity_x": float(velocity.x),
                    "velocity_y": float(velocity.y),
                    "velocity_z": float(velocity.z),
                    "speed_mps": math.sqrt(
                        float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
                    ),
                    "angular_velocity_x": float(angular_velocity.x),
                    "angular_velocity_y": float(angular_velocity.y),
                    "angular_velocity_z": float(angular_velocity.z),
                    "acceleration_x": float(acceleration.x),
                    "acceleration_y": float(acceleration.y),
                    "acceleration_z": float(acceleration.z),
                }
            )

    def _elapsed_s(self) -> float:
        if self.event_monitor is not None:
            return float(self.event_monitor._elapsed_s())
        try:
            return float(self.world.get_snapshot().timestamp.elapsed_seconds)
        except Exception:
            return float(time.time() - self.started_at)

    def _event_center(self) -> Tuple[Optional[float], str]:
        if self.event_monitor is None:
            return None, "no_event_monitor"
        target_id = None if self.event_monitor.target is None else int(self.event_monitor.target.id)
        for event in self.event_monitor.collision_events:
            if target_id is None or event.get("other_actor_id") == target_id:
                return float(event.get("elapsed_s", 0.0)), "target_collision"
        if self.event_monitor.target_started_at_s is not None:
            return float(self.event_monitor.target_started_at_s), "target_start"
        return None, "no_target_event"

    def _write_csv(self, path: Path, rows: Sequence[Dict[str, object]]) -> Optional[str]:
        if not rows:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path.name

    def write(
        self,
        out_dir: Path,
        ego_sensor_monitor: Optional[EgoSensorMonitor] = None,
        helper_camera_monitor: Optional[ActorCameraMonitor] = None,
    ) -> None:
        evidence_dir = out_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        files: Dict[str, object] = {}
        files["ground_truth_actor_trace"] = self._write_csv(
            evidence_dir / "ground_truth_actor_trace.csv",
            self.rows,
        )

        event_center_s, event_center_reason = self._event_center()
        window_rows: List[Dict[str, object]] = []
        event_window_rows: List[Dict[str, object]] = []
        if event_center_s is not None:
            start_s = float(event_center_s) - self.window_s
            end_s = float(event_center_s) + self.window_s
            window_rows = [
                row
                for row in self.rows
                if start_s <= float(row.get("elapsed_s", -1.0)) <= end_s
            ]
            if self.event_monitor is not None:
                event_window_rows = [
                    row
                    for row in self.event_monitor.trace_rows
                    if start_s <= float(row.get("elapsed_s", -1.0)) <= end_s
                ]
        files["ground_truth_event_window"] = self._write_csv(
            evidence_dir / "ground_truth_event_window.csv",
            window_rows,
        )
        files["scenario_event_window"] = self._write_csv(
            evidence_dir / "scenario_event_window.csv",
            event_window_rows,
        )

        camera_exports: List[Dict[str, object]] = []
        if ego_sensor_monitor is not None:
            camera_exports.append(ego_sensor_monitor.write_evidence_frames(evidence_dir, "ego"))
        if helper_camera_monitor is not None:
            camera_exports.append(helper_camera_monitor.write_evidence_frames(evidence_dir, "helper"))

        summary = {
            "enabled": True,
            "evidence_dir": "evidence",
            "window_s": float(self.window_s),
            "event_center_s": event_center_s,
            "event_center_reason": event_center_reason,
            "actor_trace_rows": int(len(self.rows)),
            "actor_trace_window_rows": int(len(window_rows)),
            "event_trace_window_rows": int(len(event_window_rows)),
            "files": files,
            "camera_exports": camera_exports,
        }
        (evidence_dir / "evidence_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary_path = out_dir / "summary.txt"
        if summary_path.exists():
            summary_lines = summary_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
        else:
            summary_lines = []
        summary_lines.extend(
            [
                "evidence_pack_enabled=True",
                f"evidence_dir={evidence_dir}",
                f"evidence_event_center_s={event_center_s}",
                f"evidence_event_center_reason={event_center_reason}",
                f"evidence_actor_trace_rows={len(self.rows)}",
            ]
        )
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


class OcclusionEventMonitor:
    """Script simple ego/target motion and record collision/closest-distance events."""

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Actor",
        target: Optional["carla.Actor"],
        target_end_location: Optional["carla.Location"],
        scripted_ego_drive: bool,
        ego_drive_mode: str,
        ego_route_choice: str,
        ego_route_transforms: Optional[Sequence["carla.Transform"]],
        ego_drive_throttle: float,
        ego_target_speed: float,
        ego_route_lookahead: float,
        target_crossing: bool,
        target_crossing_delay_s: float,
        target_crossing_speed: float,
        target_crossing_trigger_location: Optional["carla.Location"],
        target_crossing_trigger_distance_m: float,
        target_crossing_trigger_ttc_s: float = 0.0,
        target_crossing_trigger_route_lead_m: float = 0.0,
        target_crossing_trigger_min_ego_speed_mps: float = 0.0,
        target_motion_mode: str = "walker_control",
        target_crossing_control_speed_override: Optional[float] = None,
        target_prewalk: bool = False,
        target_prewalk_end_location: Optional["carla.Location"] = None,
        target_prewalk_speed: float = 1.2,
        target_prewalk_mode: str = "animated",
    ) -> None:
        self.world = world
        self.ego = ego
        self.target = target
        self.target_end_location = None if target_end_location is None else copy_location(target_end_location)
        self.scripted_ego_drive = bool(scripted_ego_drive)
        self.ego_drive_mode = str(ego_drive_mode)
        self.ego_route_choice = str(ego_route_choice)
        self.ego_route_transforms = list(ego_route_transforms or [])
        self.ego_route_index = 0
        self.ego_drive_throttle = max(0.0, min(1.0, float(ego_drive_throttle)))
        self.ego_target_speed = max(0.1, float(ego_target_speed))
        self.ego_route_lookahead = max(2.0, float(ego_route_lookahead))
        self.target_crossing = bool(target_crossing and target is not None and target_end_location is not None)
        self.target_motion_mode = str(target_motion_mode or "walker_control")
        if self.target_motion_mode not in {
            "walker_control",
            "deterministic",
            "exact_transform",
            "scripted_transform",
            "ai_controller",
        }:
            self.target_motion_mode = "walker_control"
        self.target_crossing_delay_s = max(0.0, float(target_crossing_delay_s))
        self.target_crossing_speed = max(0.1, float(target_crossing_speed))
        if target_crossing_control_speed_override is None:
            self.target_crossing_control_speed = self.target_crossing_speed
        else:
            self.target_crossing_control_speed = max(0.1, float(target_crossing_control_speed_override))
        self.target_crossing_trigger_location = (
            None if target_crossing_trigger_location is None else copy_location(target_crossing_trigger_location)
        )
        self.target_crossing_trigger_distance_m = max(0.0, float(target_crossing_trigger_distance_m))
        self.target_crossing_trigger_ttc_s = max(0.0, float(target_crossing_trigger_ttc_s))
        self.target_crossing_trigger_route_lead_m = max(0.0, float(target_crossing_trigger_route_lead_m))
        self.target_crossing_trigger_min_ego_speed_mps = max(
            0.0,
            float(target_crossing_trigger_min_ego_speed_mps),
        )
        self.ego_route_distances_m = self._route_cumulative_distances(self.ego_route_transforms)
        self.target_crossing_trigger_route_distance_m: Optional[float] = None
        self.target_crossing_trigger_route_gap_m: Optional[float] = None
        if self.target_crossing_trigger_location is not None and self.ego_route_transforms:
            route_projection = self._project_location_onto_ego_route(self.target_crossing_trigger_location)
            if route_projection is not None:
                (
                    self.target_crossing_trigger_route_distance_m,
                    self.target_crossing_trigger_route_gap_m,
                ) = route_projection
        self.ego_route_progress_m: Optional[float] = None
        self.ego_route_projection_gap_m: Optional[float] = None
        self.target_prewalk = bool(target_prewalk and target is not None and target_prewalk_end_location is not None)
        self.target_prewalk_end_location = (
            None if target_prewalk_end_location is None else copy_location(target_prewalk_end_location)
        )
        self.target_prewalk_speed = max(0.1, float(target_prewalk_speed))
        self.target_prewalk_mode = str(target_prewalk_mode or "animated")
        self.target_prewalk_start_location: Optional["carla.Location"] = None
        self.target_prewalk_started_at_s: Optional[float] = None
        self.started_at = time.time()
        self.started_at_sim_s: Optional[float] = None
        self.target_start_location: Optional["carla.Location"] = None
        self.min_target_distance_m: Optional[float] = None
        self.collision_events: List[Dict[str, object]] = []
        self.trace_rows: List[Dict[str, object]] = []
        self.collision_sensor: Optional["carla.Actor"] = None
        self.target_controller: Optional["carla.Actor"] = None
        self.target_started = False
        self.target_start_reason: Optional[str] = None
        self.target_started_at_s: Optional[float] = None
        self.target_crossing_completed = False
        self.target_crossing_initial_distance_m: Optional[float] = None
        self._map = world.get_map()

    def spawn(self) -> None:
        self.started_at_sim_s = self._current_world_elapsed_s()
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(self._on_collision)
        if self.target_motion_mode == "ai_controller" and self.target is not None:
            controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
            self.target_controller = self.world.spawn_actor(
                controller_bp,
                carla.Transform(),
                attach_to=self.target,
            )

    def _on_collision(self, event: object) -> None:
        other_actor = getattr(event, "other_actor", None)
        impulse = getattr(event, "normal_impulse", None)
        impulse_mag = None
        if impulse is not None:
            impulse_mag = math.sqrt(float(impulse.x) ** 2 + float(impulse.y) ** 2 + float(impulse.z) ** 2)
        self.collision_events.append(
            {
                "elapsed_s": float(self._elapsed_s()),
                "frame": int(getattr(event, "frame", -1)),
                "other_actor_id": None if other_actor is None else int(other_actor.id),
                "other_type_id": None if other_actor is None else str(other_actor.type_id),
                "normal_impulse": None
                if impulse is None
                else {"x": float(impulse.x), "y": float(impulse.y), "z": float(impulse.z)},
                "normal_impulse_magnitude": impulse_mag,
            }
        )

    def target_collision_count(self) -> int:
        if self.target is None:
            return 0
        target_id = int(self.target.id)
        return int(
            sum(
                1
                for event in self.collision_events
                if event.get("other_actor_id") == target_id
            )
        )

    def has_target_collision(self) -> bool:
        return self.target_collision_count() > 0

    def tick(self) -> None:
        elapsed = self._elapsed_s()
        self._update_ego_route_projection()
        if self.target is not None:
            distance = float(self.ego.get_location().distance(self.target.get_location()))
            if self.min_target_distance_m is None or distance < self.min_target_distance_m:
                self.min_target_distance_m = distance

        if self.target_crossing and not self.target_started:
            if self.target_prewalk:
                self._apply_target_prewalk(elapsed)
            self._maybe_start_target_crossing(elapsed)

        if self.target_crossing and self.target_started:
            if self.has_target_collision():
                self._stop_target_walker()
            else:
                self._apply_manual_target_crossing(elapsed)

        if self.scripted_ego_drive:
            if self.collision_events:
                self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            elif self.ego_drive_mode == "straight":
                self.ego.apply_control(
                    carla.VehicleControl(throttle=self.ego_drive_throttle, steer=0.0, brake=0.0)
                )
            elif self.ego_route_transforms:
                self._apply_preplanned_route_control()
            else:
                self._apply_waypoint_control()

        self._record_trace(elapsed)

    def _maybe_start_target_crossing(self, elapsed_s: float) -> None:
        if elapsed_s < self.target_crossing_delay_s:
            return
        if self.target is None or self.target_end_location is None:
            return
        if self.target_crossing_trigger_min_ego_speed_mps > 0.0:
            ego_speed = self._ego_speed_mps()
            if ego_speed < self.target_crossing_trigger_min_ego_speed_mps:
                return
        # Route-lead mode: start when the ego reaches a calibrated location
        # along the planned route. This is closer to "start N ticks/metres
        # before the ego reaches the collision point" than Euclidean TTC.
        if (
            self.target_crossing_trigger_route_lead_m > 0.0
            and self.target_crossing_trigger_route_distance_m is not None
            and self.ego_route_progress_m is not None
        ):
            trigger_at = float(self.target_crossing_trigger_route_distance_m) - float(
                self.target_crossing_trigger_route_lead_m
            )
            if float(self.ego_route_progress_m) >= trigger_at:
                self._start_target_crossing(elapsed_s, "ego_route_lead_match")
            return
        # TTC mode (preferred when set): fires when ego is N seconds away from
        # the conflict point at its current speed. Self-adjusts to ego speed
        # and pedestrian walk distance so we don't have to re-tune the trigger
        # every time we change geometry.
        if (
            self.target_crossing_trigger_ttc_s > 0.0
            and self.target_crossing_trigger_location is not None
        ):
            ego_speed = self._ego_speed_mps()
            # Ego still accelerating from spawn — wait until it has meaningful
            # forward velocity, otherwise TTC is undefined / huge.
            if ego_speed < 0.5:
                return
            ego_distance = float(self.ego.get_location().distance(self.target_crossing_trigger_location))
            ego_ttc_s = ego_distance / ego_speed
            if ego_ttc_s <= self.target_crossing_trigger_ttc_s:
                self._start_target_crossing(elapsed_s, "ego_ttc_match")
            return
        # Legacy distance mode.
        if self.target_crossing_trigger_location is None or self.target_crossing_trigger_distance_m <= 0.0:
            self._start_target_crossing(elapsed_s, "delay")
            return
        ego_distance = float(self.ego.get_location().distance(self.target_crossing_trigger_location))
        if ego_distance <= self.target_crossing_trigger_distance_m:
            self._start_target_crossing(elapsed_s, "ego_near_conflict")

    def _apply_target_prewalk(self, elapsed_s: float) -> None:
        if self.target is None or self.target_prewalk_end_location is None:
            return
        if self.target_motion_mode == "ai_controller":
            return
        if self.target_prewalk_start_location is None:
            self.target_prewalk_start_location = copy_location(self.target.get_location())
            self.target_prewalk_started_at_s = float(elapsed_s)
        start = self.target_prewalk_start_location
        end = self.target_prewalk_end_location
        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        dz = float(end.z - start.z)
        total_distance = math.sqrt(dx * dx + dy * dy)
        if total_distance <= 0.05:
            self.target.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(0.0, 0.0, 0.0), speed=0.0, jump=False)
            )
            return
        direction = carla.Vector3D(
            x=dx / total_distance,
            y=dy / total_distance,
            z=0.0,
        )
        if self.target_prewalk_mode == "animated":
            current = self.target.get_location()
            remaining = math.sqrt(float(end.x - current.x) ** 2 + float(end.y - current.y) ** 2)
            if remaining <= 0.35:
                self.target.apply_control(
                    carla.WalkerControl(direction=carla.Vector3D(0.0, 0.0, 0.0), speed=0.0, jump=False)
                )
                return
            animated_direction = carla.Vector3D(
                x=float(end.x - current.x) / remaining,
                y=float(end.y - current.y) / remaining,
                z=0.0,
            )
            self.target.apply_control(
                carla.WalkerControl(direction=animated_direction, speed=self.target_prewalk_speed, jump=False)
            )
            return
        prewalk_elapsed = max(0.0, float(elapsed_s) - float(self.target_prewalk_started_at_s or elapsed_s))
        progress = min(1.0, prewalk_elapsed * self.target_prewalk_speed / total_distance)
        new_location = carla.Location(
            x=float(start.x + dx * progress),
            y=float(start.y + dy * progress),
            z=float(start.z + dz * progress),
        )
        yaw = math.degrees(math.atan2(dy, dx))
        try:
            self.target.set_transform(
                carla.Transform(new_location, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))
            )
        except RuntimeError:
            pass
        if progress >= 1.0:
            self.target.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(0.0, 0.0, 0.0), speed=0.0, jump=False)
            )
            return
        self.target.apply_control(
            carla.WalkerControl(direction=direction, speed=min(self.target_prewalk_speed, 1.6), jump=False)
        )

    def _start_target_crossing(self, elapsed_s: float, reason: str) -> None:
        if self.target is None:
            return
        self.target_start_location = copy_location(self.target.get_location())
        if self.target_end_location is not None:
            dx = float(self.target_end_location.x - self.target_start_location.x)
            dy = float(self.target_end_location.y - self.target_start_location.y)
            self.target_crossing_initial_distance_m = math.sqrt(dx * dx + dy * dy)
        self.target_started = True
        self.target_start_reason = reason
        self.target_started_at_s = float(elapsed_s)
        if self.target_motion_mode != "ai_controller" and self.target_end_location is not None:
            dx = float(self.target_end_location.x - self.target_start_location.x)
            dy = float(self.target_end_location.y - self.target_start_location.y)
            if math.sqrt(dx * dx + dy * dy) > 0.05:
                transform = self.target.get_transform()
                try:
                    self.target.set_transform(
                        carla.Transform(
                            transform.location,
                            carla.Rotation(
                                pitch=0.0,
                                yaw=math.degrees(math.atan2(dy, dx)),
                                roll=0.0,
                            ),
                        )
                    )
                except RuntimeError:
                    pass
        if self.target_motion_mode == "ai_controller" and self.target_controller is not None:
            try:
                self.target_controller.start()
                self.target_controller.set_max_speed(
                    max(self.target_crossing_speed, self.target_crossing_control_speed)
                )
                if self.target_end_location is not None:
                    self.target_controller.go_to_location(self.target_end_location)
            except RuntimeError:
                pass

    def _record_trace(self, elapsed_s: float) -> None:
        ego_location = self.ego.get_location()
        row: Dict[str, object] = {
            "elapsed_s": float(elapsed_s),
            "ego_actor_id": int(self.ego.id),
            "ego_x": float(ego_location.x),
            "ego_y": float(ego_location.y),
            "ego_z": float(ego_location.z),
            "ego_speed_mps": float(self._ego_speed_mps()),
            "ego_route_index": int(self.ego_route_index),
            "ego_route_progress_m": "" if self.ego_route_progress_m is None else float(self.ego_route_progress_m),
            "ego_route_projection_gap_m": ""
            if self.ego_route_projection_gap_m is None
            else float(self.ego_route_projection_gap_m),
            "target_crossing_trigger_route_lead_m": float(self.target_crossing_trigger_route_lead_m),
            "target_crossing_trigger_route_distance_m": ""
            if self.target_crossing_trigger_route_distance_m is None
            else float(self.target_crossing_trigger_route_distance_m),
            "target_motion_mode": self.target_motion_mode,
            "target_crossing_control_speed": float(self.target_crossing_control_speed),
            "target_crossing_trigger_ttc_s": float(self.target_crossing_trigger_ttc_s),
            "target_crossing_trigger_min_ego_speed_mps": float(
                self.target_crossing_trigger_min_ego_speed_mps
            ),
            "target_started": int(bool(self.target_started)),
            "target_start_reason": "" if self.target_start_reason is None else self.target_start_reason,
            "target_started_at_s": "" if self.target_started_at_s is None else float(self.target_started_at_s),
            "target_crossing_completed": int(bool(self.target_crossing_completed)),
            "target_crossing_initial_distance_m": ""
            if self.target_crossing_initial_distance_m is None
            else float(self.target_crossing_initial_distance_m),
        }
        try:
            row["frame"] = int(self.world.get_snapshot().frame)
        except Exception:
            row["frame"] = -1
        if self.target_crossing_trigger_location is not None:
            ego_to_conflict_distance = float(ego_location.distance(self.target_crossing_trigger_location))
            ego_speed = float(self._ego_speed_mps())
            row["ego_to_conflict_distance_m"] = ego_to_conflict_distance
            row["ego_ttc_to_conflict_s"] = (
                ego_to_conflict_distance / ego_speed if ego_speed > 0.05 else ""
            )
        else:
            row["ego_to_conflict_distance_m"] = ""
            row["ego_ttc_to_conflict_s"] = ""
        if self.ego_route_progress_m is not None and self.target_crossing_trigger_route_distance_m is not None:
            row["ego_route_distance_to_trigger_m"] = float(
                self.target_crossing_trigger_route_distance_m - self.ego_route_progress_m
            )
        else:
            row["ego_route_distance_to_trigger_m"] = ""
        if self.target is not None:
            target_location = self.target.get_location()
            target_velocity = self.target.get_velocity()
            target_speed_mps = math.sqrt(
                float(target_velocity.x) ** 2
                + float(target_velocity.y) ** 2
                + float(target_velocity.z) ** 2
            )
            if self.target_prewalk_end_location is not None:
                row["target_prewalk_distance_to_start_m"] = float(
                    target_location.distance(self.target_prewalk_end_location)
                )
            else:
                row["target_prewalk_distance_to_start_m"] = ""
            if self.target_end_location is not None:
                dx_to_end = float(self.target_end_location.x - target_location.x)
                dy_to_end = float(self.target_end_location.y - target_location.y)
                distance_to_end = math.sqrt(dx_to_end * dx_to_end + dy_to_end * dy_to_end)
                row["target_crossing_distance_to_end_m"] = float(distance_to_end)
                if self.target_crossing_initial_distance_m is not None and self.target_crossing_initial_distance_m > 0.05:
                    row["target_crossing_progress_ratio"] = float(
                        clamp(
                            (self.target_crossing_initial_distance_m - distance_to_end)
                            / self.target_crossing_initial_distance_m,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    row["target_crossing_progress_ratio"] = ""
            else:
                row["target_crossing_distance_to_end_m"] = ""
                row["target_crossing_progress_ratio"] = ""
            row.update(
                {
                    "target_actor_id": int(self.target.id),
                    "target_x": float(target_location.x),
                    "target_y": float(target_location.y),
                    "target_z": float(target_location.z),
                    "target_speed_mps": float(target_speed_mps),
                    "ego_target_distance_m": float(ego_location.distance(target_location)),
                }
            )
        else:
            row.update(
                {
                    "target_actor_id": "",
                    "target_x": "",
                    "target_y": "",
                    "target_z": "",
                    "target_speed_mps": "",
                    "ego_target_distance_m": "",
                    "target_prewalk_distance_to_start_m": "",
                    "target_crossing_distance_to_end_m": "",
                    "target_crossing_progress_ratio": "",
                }
            )
        self.trace_rows.append(row)

    def _current_world_elapsed_s(self) -> Optional[float]:
        try:
            return float(self.world.get_snapshot().timestamp.elapsed_seconds)
        except Exception:
            return None

    def _elapsed_s(self) -> float:
        world_elapsed = self._current_world_elapsed_s()
        if world_elapsed is not None:
            if self.started_at_sim_s is None:
                self.started_at_sim_s = world_elapsed
            return max(0.0, world_elapsed - self.started_at_sim_s)
        return float(time.time() - self.started_at)

    def _ego_speed_mps(self) -> float:
        velocity = self.ego.get_velocity()
        return math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)

    def _route_cumulative_distances(self, route: Sequence["carla.Transform"]) -> List[float]:
        distances = [0.0]
        for previous, current in zip(route, route[1:]):
            distances.append(
                distances[-1]
                + float(previous.location.distance(current.location))
            )
        return distances

    def _project_location_onto_ego_route(
        self,
        location: "carla.Location",
    ) -> Optional[Tuple[float, float]]:
        if len(self.ego_route_transforms) < 2 or len(self.ego_route_distances_m) != len(self.ego_route_transforms):
            return None
        best: Optional[Tuple[float, float]] = None
        px = float(location.x)
        py = float(location.y)
        for index, (start_tf, end_tf) in enumerate(zip(self.ego_route_transforms, self.ego_route_transforms[1:])):
            sx = float(start_tf.location.x)
            sy = float(start_tf.location.y)
            ex = float(end_tf.location.x)
            ey = float(end_tf.location.y)
            vx = ex - sx
            vy = ey - sy
            segment_len_sq = vx * vx + vy * vy
            if segment_len_sq <= 1e-6:
                continue
            t = clamp(((px - sx) * vx + (py - sy) * vy) / segment_len_sq, 0.0, 1.0)
            proj_x = sx + vx * t
            proj_y = sy + vy * t
            gap = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
            segment_len = math.sqrt(segment_len_sq)
            route_distance = float(self.ego_route_distances_m[index]) + segment_len * t
            if best is None or gap < best[1]:
                best = (route_distance, gap)
        return best

    def _update_ego_route_projection(self) -> None:
        projection = self._project_location_onto_ego_route(self.ego.get_location())
        if projection is None:
            return
        route_distance, route_gap = projection
        if self.ego_route_progress_m is None:
            self.ego_route_progress_m = route_distance
        else:
            # The route projection can jitter by a small amount around curved
            # segments; for trigger purposes ego progress should be monotonic.
            self.ego_route_progress_m = max(float(self.ego_route_progress_m), route_distance)
        self.ego_route_projection_gap_m = route_gap

    def _apply_manual_target_crossing(self, elapsed_s: float) -> None:
        if self.target is None or self.target_start_location is None or self.target_end_location is None:
            return
        if self.target_crossing_completed:
            return
        if self.target_motion_mode in {"exact_transform", "scripted_transform"}:
            self._apply_exact_transform_target_crossing(elapsed_s)
            return
        if self.target_motion_mode == "deterministic":
            self._apply_deterministic_target_crossing(elapsed_s)
            return
        current = self.target.get_location()
        end = self.target_end_location
        dx = float(end.x - current.x)
        dy = float(end.y - current.y)
        horizontal_distance = math.sqrt(dx * dx + dy * dy)
        if self.target_motion_mode == "ai_controller":
            if horizontal_distance <= 0.55:
                if self.target_controller is not None:
                    try:
                        self.target_controller.stop()
                    except RuntimeError:
                        pass
                self.target_crossing_completed = True
            return
        # Tighter stop tolerance (0.15 m, down from 0.35 m): at high commanded
        # walker speeds the previous threshold let the walker overshoot the
        # target by up to ~0.3 m and stop in a slightly random position,
        # making collision repeatability worse.
        if horizontal_distance <= 0.15:
            self.target.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(0.0, 0.0, 0.0), speed=0.0, jump=False)
            )
            self.target_crossing_completed = True
            return
        direction = carla.Vector3D(
            x=dx / horizontal_distance,
            y=dy / horizontal_distance,
            z=0.0,
        )
        self.target.apply_control(
            carla.WalkerControl(direction=direction, speed=self.target_crossing_control_speed, jump=False)
        )

    def _apply_deterministic_target_crossing(self, elapsed_s: float) -> None:
        if self.target is None or self.target_start_location is None or self.target_end_location is None:
            return
        current = self.target.get_location()
        end = self.target_end_location
        dx = float(end.x - current.x)
        dy = float(end.y - current.y)
        horizontal_distance = math.sqrt(dx * dx + dy * dy)
        if horizontal_distance <= 0.15:
            self.target_crossing_completed = True
            try:
                self.target.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                self.target.apply_control(
                    carla.WalkerControl(direction=carla.Vector3D(0.0, 0.0, 0.0), speed=0.0, jump=False)
                )
            except RuntimeError:
                pass
            return
        direction = carla.Vector3D(
            x=dx / horizontal_distance,
            y=dy / horizontal_distance,
            z=0.0,
        )
        root_speed = max(0.1, float(self.target_crossing_control_speed))
        animation_speed = max(0.8, min(5.5, root_speed))
        try:
            self.target.set_target_velocity(
                carla.Vector3D(
                    x=direction.x * root_speed,
                    y=direction.y * root_speed,
                    z=0.0,
                )
            )
            self.target.apply_control(
                carla.WalkerControl(direction=direction, speed=animation_speed, jump=False)
            )
        except RuntimeError:
            pass

    def _apply_exact_transform_target_crossing(self, elapsed_s: float) -> None:
        if self.target is None or self.target_start_location is None or self.target_end_location is None:
            return
        start = self.target_start_location
        end = self.target_end_location
        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        dz = float(end.z - start.z)
        horizontal_distance = math.sqrt(dx * dx + dy * dy)
        if horizontal_distance <= 0.05:
            self.target_crossing_completed = True
            return
        direction = carla.Vector3D(
            x=dx / horizontal_distance,
            y=dy / horizontal_distance,
            z=0.0,
        )
        crossing_elapsed = max(0.0, float(elapsed_s) - float(self.target_started_at_s or elapsed_s))
        progress = min(1.0, crossing_elapsed * self.target_crossing_control_speed / horizontal_distance)
        new_location = carla.Location(
            x=float(start.x + dx * progress),
            y=float(start.y + dy * progress),
            z=float(start.z + dz * progress),
        )
        yaw = math.degrees(math.atan2(dy, dx))
        try:
            self.target.set_transform(
                carla.Transform(new_location, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))
            )
            animation_speed = 0.0 if progress >= 1.0 else max(0.8, min(5.5, self.target_crossing_control_speed))
            velocity_speed = 0.0 if progress >= 1.0 else max(0.1, float(self.target_crossing_control_speed))
            self.target.set_target_velocity(
                carla.Vector3D(
                    x=direction.x * velocity_speed,
                    y=direction.y * velocity_speed,
                    z=0.0,
                )
            )
            self.target.apply_control(
                carla.WalkerControl(direction=direction, speed=animation_speed, jump=False)
            )
        except RuntimeError:
            pass
        if progress >= 1.0:
            self.target_crossing_completed = True

    def _stop_target_walker(self) -> None:
        if self.target is None:
            return
        if self.target_motion_mode == "ai_controller" and self.target_controller is not None:
            try:
                self.target_controller.stop()
            except RuntimeError:
                pass
            return
        try:
            self.target.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.target.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(0.0, 0.0, 0.0), speed=0.0, jump=False)
            )
        except RuntimeError:
            pass

    def _apply_waypoint_control(self) -> None:
        transform = self.ego.get_transform()
        location = transform.location
        try:
            current_wp = self._map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            current_wp = None
        if current_wp is None:
            # Lost the road (ego drifted off the drivable surface, or the route
            # ended). DO NOT keep applying throttle straight ahead — that's how
            # the ego ends up driving into buildings at spawn points where the
            # route runs out. Brake instead.
            self.ego.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            )
            return

        next_wps = current_wp.next(self.ego_route_lookahead)
        if not next_wps:
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.6))
            return
        target_wp = choose_route_next_waypoint(current_wp, next_wps, self.ego_route_choice)
        target_loc = target_wp.transform.location
        target_yaw = math.degrees(math.atan2(target_loc.y - location.y, target_loc.x - location.x))
        yaw_error = signed_angular_difference_deg(target_yaw, float(transform.rotation.yaw))
        steer = clamp(yaw_error / 45.0, -0.65, 0.65)
        speed = self._ego_speed_mps()
        throttle = self.ego_drive_throttle if speed < self.ego_target_speed else 0.0
        brake = 0.0 if speed < self.ego_target_speed * 1.15 else 0.15
        self.ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    def _apply_preplanned_route_control(self) -> None:
        transform = self.ego.get_transform()
        location = transform.location
        last_index = len(self.ego_route_transforms) - 1
        if self.ego_route_index >= last_index:
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))
            return

        if (
            self.ego_route_progress_m is not None
            and len(self.ego_route_distances_m) == len(self.ego_route_transforms)
        ):
            while (
                self.ego_route_index < last_index
                and float(self.ego_route_distances_m[self.ego_route_index])
                < float(self.ego_route_progress_m) - 1.0
            ):
                self.ego_route_index += 1

        while self.ego_route_index < last_index:
            current_target = self.ego_route_transforms[self.ego_route_index].location
            if location.distance(current_target) >= 3.5:
                break
            self.ego_route_index += 1

        target_index = self.ego_route_index
        while target_index < last_index:
            target_location = self.ego_route_transforms[target_index].location
            if location.distance(target_location) >= self.ego_route_lookahead:
                break
            target_index += 1

        target_location = self.ego_route_transforms[target_index].location
        target_yaw = math.degrees(math.atan2(target_location.y - location.y, target_location.x - location.x))
        yaw_error = signed_angular_difference_deg(target_yaw, float(transform.rotation.yaw))
        steer = clamp(yaw_error / 45.0, -0.75, 0.75)
        speed = self._ego_speed_mps()
        throttle = self.ego_drive_throttle if speed < self.ego_target_speed else 0.0
        brake = 0.0 if speed < self.ego_target_speed * 1.15 else 0.2
        self.ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    def metadata(self) -> Dict[str, object]:
        return {
            "enabled": True,
            "scripted_ego_drive": self.scripted_ego_drive,
            "ego_drive_mode": self.ego_drive_mode,
            "ego_route_choice": self.ego_route_choice,
            "ego_preplanned_route_points": int(len(self.ego_route_transforms)),
            "ego_drive_throttle": self.ego_drive_throttle,
            "ego_target_speed": self.ego_target_speed,
            "ego_route_lookahead": self.ego_route_lookahead,
            "target_crossing": self.target_crossing,
            "target_motion_mode": self.target_motion_mode,
            "target_crossing_delay_s": self.target_crossing_delay_s,
            "target_crossing_speed": self.target_crossing_speed,
            "target_crossing_control_speed": self.target_crossing_control_speed,
            "target_crossing_trigger_distance_m": self.target_crossing_trigger_distance_m,
            "target_crossing_trigger_ttc_s": self.target_crossing_trigger_ttc_s,
            "target_crossing_trigger_route_lead_m": self.target_crossing_trigger_route_lead_m,
            "target_crossing_trigger_min_ego_speed_mps": self.target_crossing_trigger_min_ego_speed_mps,
            "target_crossing_trigger_route_distance_m": self.target_crossing_trigger_route_distance_m,
            "target_crossing_trigger_route_gap_m": self.target_crossing_trigger_route_gap_m,
            "target_crossing_trigger_location": None
            if self.target_crossing_trigger_location is None
            else location_to_dict(self.target_crossing_trigger_location),
            "target_crossing_initial_distance_m": self.target_crossing_initial_distance_m,
            "target_prewalk": self.target_prewalk,
            "target_prewalk_speed": self.target_prewalk_speed,
            "target_prewalk_mode": self.target_prewalk_mode,
            "target_prewalk_start_location": None
            if self.target_prewalk_start_location is None
            else location_to_dict(self.target_prewalk_start_location),
            "target_prewalk_end_location": None
            if self.target_prewalk_end_location is None
            else location_to_dict(self.target_prewalk_end_location),
            "target_end_location": None
            if self.target_end_location is None
            else location_to_dict(self.target_end_location),
        }

    def summary(self) -> Dict[str, object]:
        return {
            "elapsed_s": float(self._elapsed_s()),
            "ego_actor_id": int(self.ego.id),
            "target_actor_id": None if self.target is None else int(self.target.id),
            "scripted_ego_drive": self.scripted_ego_drive,
            "ego_drive_mode": self.ego_drive_mode,
            "ego_route_choice": self.ego_route_choice,
            "ego_preplanned_route_points": int(len(self.ego_route_transforms)),
            "ego_route_index": int(self.ego_route_index),
            "target_crossing": self.target_crossing,
            "target_motion_mode": self.target_motion_mode,
            "target_crossing_completed": self.target_crossing_completed,
            "target_crossing_initial_distance_m": self.target_crossing_initial_distance_m,
            "target_started": self.target_started,
            "target_start_reason": self.target_start_reason,
            "target_started_at_s": self.target_started_at_s,
            "target_prewalk": self.target_prewalk,
            "target_prewalk_speed": self.target_prewalk_speed,
            "target_prewalk_mode": self.target_prewalk_mode,
            "target_prewalk_start_location": None
            if self.target_prewalk_start_location is None
            else location_to_dict(self.target_prewalk_start_location),
            "target_prewalk_end_location": None
            if self.target_prewalk_end_location is None
            else location_to_dict(self.target_prewalk_end_location),
            "target_crossing_trigger_distance_m": self.target_crossing_trigger_distance_m,
            "target_crossing_trigger_ttc_s": self.target_crossing_trigger_ttc_s,
            "target_crossing_trigger_route_lead_m": self.target_crossing_trigger_route_lead_m,
            "target_crossing_trigger_min_ego_speed_mps": self.target_crossing_trigger_min_ego_speed_mps,
            "target_crossing_trigger_route_distance_m": self.target_crossing_trigger_route_distance_m,
            "target_crossing_trigger_route_gap_m": self.target_crossing_trigger_route_gap_m,
            "ego_route_progress_m": self.ego_route_progress_m,
            "ego_route_projection_gap_m": self.ego_route_projection_gap_m,
            "target_crossing_trigger_location": None
            if self.target_crossing_trigger_location is None
            else location_to_dict(self.target_crossing_trigger_location),
            "min_target_distance_m": self.min_target_distance_m,
            "target_near_miss_threshold_m": 3.0,
            "target_danger_event": bool(
                self.min_target_distance_m is not None and self.min_target_distance_m <= 3.0
            ),
            "event_trace_file": "scenario_event_trace.csv",
            "collision_count": int(len(self.collision_events)),
            "target_collision_count": int(
                sum(
                    1
                    for event in self.collision_events
                    if self.target is not None and event.get("other_actor_id") == int(self.target.id)
                )
            ),
            "collision_events": self.collision_events,
            "ego_final_transform": transform_to_dict(self.ego.get_transform()),
            "target_final_transform": None
            if self.target is None
            else transform_to_dict(self.target.get_transform()),
        }

    def write_summary(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        (out_dir / "scenario_event_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary_path = out_dir / "summary.txt"
        if summary_path.exists():
            summary_lines = summary_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
        else:
            summary_lines = []
        summary_lines.extend(
            [
                f"event_target_started={summary.get('target_started')}",
                f"event_target_crossing_completed={summary.get('target_crossing_completed')}",
                f"event_target_collision_count={summary.get('target_collision_count')}",
                f"event_collision_count={summary.get('collision_count')}",
                f"event_target_danger_event={summary.get('target_danger_event')}",
                f"event_min_target_distance_m={summary.get('min_target_distance_m')}",
                f"event_target_started_at_s={summary.get('target_started_at_s')}",
                f"event_target_start_reason={summary.get('target_start_reason')}",
                f"event_target_crossing_initial_distance_m={summary.get('target_crossing_initial_distance_m')}",
                f"event_target_crossing_trigger_route_lead_m={summary.get('target_crossing_trigger_route_lead_m')}",
                f"event_target_crossing_trigger_min_ego_speed_mps={summary.get('target_crossing_trigger_min_ego_speed_mps')}",
                f"event_target_crossing_trigger_route_distance_m={summary.get('target_crossing_trigger_route_distance_m')}",
                f"event_target_prewalk={summary.get('target_prewalk')}",
                f"event_target_prewalk_speed={summary.get('target_prewalk_speed')}",
                f"event_target_prewalk_mode={summary.get('target_prewalk_mode')}",
            ]
        )
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        if self.trace_rows:
            fieldnames = [
                "elapsed_s",
                "frame",
                "ego_actor_id",
                "ego_x",
                "ego_y",
                "ego_z",
                "ego_speed_mps",
                "ego_route_index",
                "ego_route_progress_m",
                "ego_route_projection_gap_m",
                "ego_to_conflict_distance_m",
                "ego_ttc_to_conflict_s",
                "ego_route_distance_to_trigger_m",
                "target_motion_mode",
                "target_actor_id",
                "target_x",
                "target_y",
                "target_z",
                "target_speed_mps",
                "ego_target_distance_m",
                "target_prewalk_distance_to_start_m",
                "target_crossing_distance_to_end_m",
                "target_crossing_progress_ratio",
                "target_crossing_completed",
                "target_crossing_initial_distance_m",
                "target_crossing_control_speed",
                "target_crossing_trigger_ttc_s",
                "target_crossing_trigger_route_lead_m",
                "target_crossing_trigger_min_ego_speed_mps",
                "target_crossing_trigger_route_distance_m",
                "target_started",
                "target_start_reason",
                "target_started_at_s",
            ]
            with (out_dir / "scenario_event_trace.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.trace_rows)

    def destroy(self) -> None:
        if self.scripted_ego_drive:
            try:
                self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            except Exception:
                pass
        if self.target_controller is not None:
            try:
                self.target_controller.stop()
            except Exception:
                pass
            try:
                if self.target_controller.is_alive:
                    self.target_controller.destroy()
            except Exception:
                pass
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
            except Exception:
                pass
            try:
                if self.collision_sensor.is_alive:
                    self.collision_sensor.destroy()
            except Exception:
                pass


def look_at_rotation(source: "carla.Location", target: "carla.Location") -> "carla.Rotation":
    dx = float(target.x - source.x)
    dy = float(target.y - source.y)
    dz = float(target.z - source.z)
    yaw = math.degrees(math.atan2(dy, dx))
    horizontal = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horizontal))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def set_spectator(world: "carla.World", anchor: "carla.Location", height_m: float) -> None:
    spectator = world.get_spectator()
    spectator_location = carla.Location(
        x=anchor.x - 35.0,
        y=anchor.y - 35.0,
        z=anchor.z + height_m,
    )
    target_location = carla.Location(x=anchor.x, y=anchor.y, z=anchor.z + 1.0)
    spectator.set_transform(carla.Transform(spectator_location, look_at_rotation(spectator_location, target_location)))


def build_output_dir(output_root: Path, scenario: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_{sanitize_token(scenario)}_seed{seed}"


def destroy_actors(actors: Sequence["carla.Actor"], controllers: Sequence["carla.Actor"]) -> None:
    for controller in controllers:
        try:
            controller.stop()
        except Exception:
            pass
    for actor in list(controllers) + list(reversed(actors)):
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except Exception:
            pass


def write_outputs(
    out_dir: Path,
    manifest: Dict[str, object],
    actor_rows: Sequence[Dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenario_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "actors.json").write_text(
        json.dumps(list(actor_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_lines = [
        f"scenario={manifest['scenario']['name']}",
        f"seed={manifest['seed']}",
        f"map={manifest['carla']['map_name']}",
        f"anchor={manifest['anchor']['source']} id={manifest['anchor']['traffic_light_id']}",
        f"actors_total={len(actor_rows)}",
        f"vehicles_spawned={manifest['spawn_counts']['vehicles_spawned']}",
        f"pedestrians_spawned={manifest['spawn_counts']['pedestrians_spawned']}",
        f"ego_sensors_enabled={bool(manifest.get('ego_sensors'))}",
        f"occlusion_event_enabled={bool(manifest.get('occlusion_event'))}",
        f"output_dir={out_dir}",
    ]
    occlusion_event = manifest.get("occlusion_event")
    if isinstance(occlusion_event, dict):
        summary_lines.extend(
            [
                f"target_crossing_trigger_distance_m={occlusion_event.get('target_crossing_trigger_distance_m')}",
                f"target_crossing_trigger_ttc_s={occlusion_event.get('target_crossing_trigger_ttc_s')}",
                f"target_crossing_speed_mps={occlusion_event.get('target_crossing_speed')}",
                f"target_crossing_control_speed={occlusion_event.get('target_crossing_control_speed')}",
                f"target_motion_mode={occlusion_event.get('target_motion_mode')}",
            ]
        )
    occlusion_layout = manifest.get("occlusion_layout")
    if isinstance(occlusion_layout, dict):
        summary_lines.extend(
            [
                f"ego_spawn_index={occlusion_layout.get('ego_spawn_index')}",
                f"ego_start_forward_m={occlusion_layout.get('ego_start_forward_m')}",
                f"occluder_blueprint_id={occlusion_layout.get('occluder_blueprint_id')}",
                f"occluder_simulate_physics={occlusion_layout.get('occluder_simulate_physics')}",
                f"conflict_crosswalk_gap_m={occlusion_layout.get('conflict_crosswalk_gap_m')}",
                "occluder_distance_m="
                f"{occlusion_layout.get('occluder_distance_m', occlusion_layout.get('primary_occluder_distance_m'))}",
                f"conflict_distance_m={occlusion_layout.get('conflict_distance_m')}",
                f"helper_vehicle_enabled={occlusion_layout.get('helper_vehicle_enabled')}",
                f"helper_vehicle_actor_id={occlusion_layout.get('helper_vehicle_actor_id')}",
                f"helper_vehicle_blueprint_id={occlusion_layout.get('helper_vehicle_blueprint_id')}",
                f"helper_vehicle_drive={occlusion_layout.get('helper_vehicle_drive')}",
                f"helper_vehicle_lateral_offset_m={occlusion_layout.get('helper_vehicle_lateral_offset_m')}",
            ]
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def hold_scene(
    world: "carla.World",
    duration_s: float,
    sync_world: bool,
    ego_sensor_monitor: Optional[EgoSensorMonitor] = None,
    helper_camera_monitor: Optional[ActorCameraMonitor] = None,
    helper_vehicle_controller: Optional[HelperVehicleController] = None,
    event_monitor: Optional[OcclusionEventMonitor] = None,
    evidence_recorder: Optional[ScenarioEvidenceRecorder] = None,
    stop_on_target_collision: bool = False,
    post_target_collision_hold_s: float = 0.0,
) -> None:
    def poll_previews() -> bool:
        if ego_sensor_monitor is not None and not ego_sensor_monitor.poll_preview():
            print("Ego preview closed; ending scenario.")
            return False
        if helper_camera_monitor is not None and not helper_camera_monitor.poll_preview():
            print("Helper preview closed; ending scenario.")
            return False
        return True

    def hold_after_target_collision() -> None:
        hold_s = max(0.0, float(post_target_collision_hold_s))
        if hold_s <= 0.0:
            return
        print(f"Target collision detected; holding scene for {hold_s:.1f}s.")
        hold_end = time.monotonic() + hold_s
        while time.monotonic() < hold_end:
            if sync_world:
                world.tick()
            else:
                time.sleep(0.05)
            if helper_vehicle_controller is not None:
                helper_vehicle_controller.tick()
            if event_monitor is not None:
                event_monitor.tick()
            if evidence_recorder is not None:
                evidence_recorder.tick()
            if not poll_previews():
                return

    if duration_s == 0:
        print("Scenario active. Press Ctrl+C to stop.")
        while True:
            if sync_world:
                world.tick()
            else:
                time.sleep(0.05)
            if helper_vehicle_controller is not None:
                helper_vehicle_controller.tick()
            if event_monitor is not None:
                event_monitor.tick()
            if evidence_recorder is not None:
                evidence_recorder.tick()
            if event_monitor is not None:
                if stop_on_target_collision and event_monitor.has_target_collision():
                    hold_after_target_collision()
                    print("Target collision detected; ending scenario.")
                    return
            if not poll_previews():
                return
    end_time = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < end_time:
        if sync_world:
            world.tick()
        else:
            time.sleep(0.05)
        if helper_vehicle_controller is not None:
            helper_vehicle_controller.tick()
        if event_monitor is not None:
            event_monitor.tick()
        if evidence_recorder is not None:
            evidence_recorder.tick()
        if event_monitor is not None:
            if stop_on_target_collision and event_monitor.has_target_collision():
                hold_after_target_collision()
                print("Target collision detected; ending scenario.")
                return
        if not poll_previews():
            return


def list_scenarios() -> None:
    print("Available SceneSense scenarios:")
    for name, spec in sorted(SCENARIOS.items()):
        print(f"  {name}: {spec.description}")


def main() -> int:
    global carla
    args = parse_args()
    if args.list:
        list_scenarios()
        return 0

    if carla is None:
        carla = _bootstrap_carla()

    if getattr(args, "list_vehicles", False):
        client = carla.Client(args.host, int(args.port))
        client.set_timeout(15.0)
        world = client.get_world()
        bps = sorted(world.get_blueprint_library().filter("vehicle.*"), key=lambda b: b.id)
        truck_tokens = ("truck", "carlacola", "firetruck", "fusorosa", "bus", "sprinter", "van", "cybertruck", "t2", "ambulance")
        trucks = [b for b in bps if any(t in str(b.id).lower() for t in truck_tokens)]
        sedans = [b for b in bps if b not in trucks]
        print(f"Trucks/vans/large vehicles ({len(trucks)}):")
        for b in trucks:
            print(f"  {b.id}")
        print(f"\nCars/sedans/other ({len(sedans)}):")
        for b in sedans:
            print(f"  {b.id}")
        return 0

    spec = SCENARIOS[args.scenario]
    rng = random.Random(int(args.seed))
    random.seed(int(args.seed))
    output_root = Path(args.output_root).expanduser().resolve()
    out_dir = build_output_dir(output_root, spec.name, int(args.seed))

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(15.0)
    if args.load_town:
        town = args.town.strip() or spec.default_town
        print(f"Loading CARLA town {town}...")
        world = client.load_world(town)
    else:
        world = client.get_world()

    traffic_manager = client.get_trafficmanager(int(args.tm_port))
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    try:
        traffic_manager.set_random_device_seed(int(args.seed))
    except RuntimeError:
        pass
    try:
        world.set_pedestrians_seed(int(args.seed))
        world.set_pedestrians_cross_factor(0.0)
    except Exception:
        pass

    original_settings = world.get_settings()
    sync_world = not bool(args.async_world)
    if sync_world:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(args.fixed_delta_s)
        world.apply_settings(settings)
        try:
            traffic_manager.set_synchronous_mode(True)
        except RuntimeError:
            pass

    traffic_light_id = str(args.traffic_light_id or spec.traffic_light_id)
    anchor, anchor_info = resolve_anchor(
        world,
        traffic_light_id,
        anchor_source=str(args.anchor_source),
        anchor_spawn_index=int(args.anchor_spawn_index),
    )
    if args.set_spectator:
        set_spectator(world, anchor, spec.spectator_height_m)

    candidates = spawn_points_near(world, anchor, spec.anchor_radius_m, min_distance_m=6.0)
    vehicle_count = spec.background_vehicles if args.vehicle_count < 0 else int(args.vehicle_count)
    pedestrian_count = spec.pedestrians if args.pedestrian_count < 0 else int(args.pedestrian_count)
    actors: List["carla.Actor"] = []
    controllers: List["carla.Actor"] = []
    ego_sensor_monitor: Optional[EgoSensorMonitor] = None
    helper_camera_monitor: Optional[ActorCameraMonitor] = None
    helper_vehicle_controller: Optional[HelperVehicleController] = None
    event_monitor: Optional[OcclusionEventMonitor] = None
    evidence_recorder: Optional[ScenarioEvidenceRecorder] = None
    special_layout: Optional[Dict[str, object]] = None

    try:
        if spec.intersection_truck_occlusion:
            ego, special_actors, special_layout = spawn_intersection_truck_pedestrian_layout(
                world,
                client,
                traffic_manager,
                candidates,
                anchor,
                spec,
                rng,
                ego_autopilot=bool(args.ego_autopilot),
                route_choice=str(args.ego_route_choice),
            )
            actors.append(ego)
            actors.extend(special_actors)
            used_locations = [actor.get_location() for actor in actors]
            if args.set_spectator and args.spectator_focus == "conflict":
                observer_transform_raw = special_layout.get("observer_spectator_transform")
                if isinstance(observer_transform_raw, dict):
                    world.get_spectator().set_transform(transform_from_dict(observer_transform_raw))
                else:
                    observer_raw = special_layout.get("observer_location") or special_layout.get("conflict_location")
                    if isinstance(observer_raw, dict):
                        set_spectator(world, location_from_dict(observer_raw), spec.spectator_height_m)
        elif spec.curbside_occlusion:
            ego, special_actors, special_layout = spawn_curbside_parked_pedestrian_layout(
                world,
                client,
                traffic_manager,
                candidates,
                anchor,
                spec,
                rng,
                ego_autopilot=bool(args.ego_autopilot),
                route_choice=str(args.ego_route_choice),
                ego_spawn_index=int(args.ego_spawn_index),
                curbside_conflict_distance_m=float(args.curbside_conflict_distance_m),
                curbside_occluder_lateral_offset_m=float(args.curbside_occluder_lateral_offset_m),
                curbside_target_start_lateral_offset_m=float(args.curbside_target_start_lateral_offset_m),
                curbside_target_end_lateral_offset_m=float(args.curbside_target_end_lateral_offset_m),
                curbside_target_forward_offset_m=float(args.curbside_target_forward_offset_m),
                curbside_ego_start_forward_m=float(args.curbside_ego_start_forward_m),
                curbside_target_prewalk_distance_m=float(args.curbside_target_prewalk_distance_m),
                curbside_target_prewalk_lateral_offset_m=float(args.curbside_target_prewalk_lateral_offset_m),
                curbside_heavy_occluder_first=bool(args.curbside_heavy_occluder_first),
                helper_vehicle=bool(args.helper_vehicle or args.helper_camera_preview or args.helper_drive),
                helper_drive=bool(args.helper_drive),
                helper_spawn_forward_m=float(args.curbside_helper_spawn_forward_m),
                helper_target_forward_m=float(args.curbside_helper_target_forward_m),
                extra_occluder_forward_m=float(args.curbside_extra_occluder_forward_m),
                slot_0_forward_m=float(args.curbside_slot_0_forward_m),
                slot_1_forward_m=float(args.curbside_slot_1_forward_m),
                occluder_count=int(args.curbside_occluder_count),
                forced_occluder_blueprint_id=str(args.curbside_occluder_blueprint or ""),
                occluder_yaw_offset_deg=float(args.curbside_occluder_yaw_offset_deg),
                occluder_z_offset_m=float(args.curbside_occluder_z_offset_m),
            )
            actors.append(ego)
            actors.extend(special_actors)
            used_locations = [actor.get_location() for actor in actors]
            if args.set_spectator and args.spectator_focus == "conflict":
                observer_transform_raw = special_layout.get("observer_spectator_transform")
                if isinstance(observer_transform_raw, dict):
                    world.get_spectator().set_transform(transform_from_dict(observer_transform_raw))
                else:
                    observer_raw = special_layout.get("observer_location") or special_layout.get("conflict_location")
                    if isinstance(observer_raw, dict):
                        set_spectator(world, location_from_dict(observer_raw), spec.spectator_height_m)
        elif spec.manual_occlusion_crossing:
            ego, special_actors, special_layout = spawn_occlusion_crossing_layout(
                world,
                client,
                traffic_manager,
                candidates,
                anchor,
                spec,
                rng,
                ego_autopilot=bool(args.ego_autopilot),
                route_choice=str(args.ego_route_choice),
            )
            actors.append(ego)
            actors.extend(special_actors)
            used_locations = [actor.get_location() for actor in actors]
            if args.set_spectator and args.spectator_focus == "conflict":
                observer_transform_raw = special_layout.get("observer_spectator_transform")
                if isinstance(observer_transform_raw, dict):
                    world.get_spectator().set_transform(transform_from_dict(observer_transform_raw))
                else:
                    observer_raw = special_layout.get("observer_location") or special_layout.get("conflict_location")
                    if isinstance(observer_raw, dict):
                        set_spectator(world, location_from_dict(observer_raw), spec.spectator_height_m)
        else:
            ego = spawn_ego(
                client,
                world,
                traffic_manager,
                candidates,
                anchor,
                spec,
                rng,
                autopilot=bool(args.ego_autopilot),
            )
            actors.append(ego)
            used_locations = [ego.get_location()]

        if args.ego_sensors:
            ego_sensor_monitor = EgoSensorMonitor(
                world,
                ego,
                camera_width=int(args.ego_camera_width),
                camera_height=int(args.ego_camera_height),
                camera_fov=float(args.ego_camera_fov),
                radar_range=float(args.ego_radar_range),
                radar_hfov=float(args.ego_radar_hfov),
                radar_vfov=float(args.ego_radar_vfov),
                radar_pps=int(args.ego_radar_pps),
                preview=bool(args.ego_camera_preview),
                evidence_capture=bool(args.evidence_pack),
                evidence_buffer_size=int(args.evidence_camera_buffer_size),
                evidence_stride=int(args.evidence_camera_stride),
            )
            ego_sensor_monitor.spawn()
            if ego_sensor_monitor.preview_error:
                print(ego_sensor_monitor.preview_error)

        if args.helper_camera_preview and special_layout is not None:
            helper_actor_id = special_layout.get("helper_vehicle_actor_id")
            helper_actor = world.get_actor(int(helper_actor_id)) if helper_actor_id is not None else None
            if helper_actor is None:
                print("Helper camera preview requested, but no helper vehicle was spawned.")
            else:
                helper_camera_monitor = ActorCameraMonitor(
                    world,
                    helper_actor,
                    label="helper RGB",
                    window_name="SceneSense Helper RGB Preview",
                    camera_width=int(args.helper_camera_width),
                    camera_height=int(args.helper_camera_height),
                    camera_fov=float(args.helper_camera_fov),
                    preview=True,
                    evidence_capture=bool(args.evidence_pack),
                    evidence_buffer_size=int(args.evidence_camera_buffer_size),
                    evidence_stride=int(args.evidence_camera_stride),
                )
                helper_camera_monitor.spawn()
                if helper_camera_monitor.preview_error:
                    print(helper_camera_monitor.preview_error)

        # Mutable forward-reference cell so the helper's gate predicate can see
        # the OcclusionEventMonitor that will be constructed below.
        event_monitor_ref: List[Optional["OcclusionEventMonitor"]] = [None]

        if args.helper_drive and special_layout is not None:
            helper_actor_id = special_layout.get("helper_vehicle_actor_id")
            helper_actor = world.get_actor(int(helper_actor_id)) if helper_actor_id is not None else None
            helper_target_raw = special_layout.get("helper_vehicle_target_location") or special_layout.get(
                "conflict_location"
            )
            if helper_actor is None:
                print("Helper drive requested, but no helper vehicle was spawned.")
            elif not isinstance(helper_target_raw, dict):
                print("Helper drive requested, but no helper target location was recorded.")
            else:
                helper_gate = None
                if args.helper_pause_until_crossing:
                    # Release the helper the moment the pedestrian crossing
                    # actually fires. event_monitor.target_started flips to True
                    # inside OcclusionEventMonitor._start_target_crossing().
                    def helper_gate() -> bool:
                        monitor = event_monitor_ref[0]
                        return monitor is not None and bool(getattr(monitor, "target_started", False))
                helper_vehicle_controller = HelperVehicleController(
                    helper_actor,
                    location_from_dict(helper_target_raw),
                    target_speed=float(args.helper_target_speed),
                    stop_distance_m=float(args.helper_stop_distance_to_conflict_m),
                    gate_predicate=helper_gate,
                )

        if (
            spec.manual_occlusion_crossing
            or spec.intersection_truck_occlusion
            or spec.curbside_occlusion
        ) and special_layout is not None:
            target_actor_id = special_layout.get("target_actor_id")
            target_actor = world.get_actor(int(target_actor_id)) if target_actor_id is not None else None
            target_end_raw = special_layout.get("target_crossing_end_location")
            target_end = None
            if isinstance(target_end_raw, dict):
                target_end = location_from_dict(target_end_raw)
            target_trigger_location = None
            target_trigger_raw = (
                special_layout.get("target_crossing_trigger_location")
                or special_layout.get("conflict_location")
            )
            if isinstance(target_trigger_raw, dict):
                target_trigger_location = location_from_dict(target_trigger_raw)
            target_prewalk_end = None
            target_prewalk_end_raw = special_layout.get("target_prewalk_end_location")
            if isinstance(target_prewalk_end_raw, dict):
                target_prewalk_end = location_from_dict(target_prewalk_end_raw)
            route_rows = special_layout.get("controller_route_transforms")
            ego_route_transforms = []
            if isinstance(route_rows, list):
                ego_route_transforms = [
                    transform_from_dict(row)
                    for row in route_rows
                    if isinstance(row, dict) and "location" in row and "rotation" in row
                ]
            layout_control_speed = special_layout.get("target_crossing_control_speed_override")
            if float(args.target_crossing_control_speed) > 0.0:
                target_control_speed_override = float(args.target_crossing_control_speed)
            elif layout_control_speed is None:
                target_control_speed_override = None
            else:
                target_control_speed_override = float(layout_control_speed)
            target_motion_mode = str(special_layout.get("target_motion_mode", "walker_control"))
            if str(args.target_crossing_motion_mode):
                target_motion_mode = str(args.target_crossing_motion_mode)
            event_monitor = OcclusionEventMonitor(
                world,
                ego,
                target_actor,
                target_end,
                scripted_ego_drive=bool(args.scripted_ego_drive),
                ego_drive_mode=str(args.ego_drive_mode),
                ego_route_choice=str(args.ego_route_choice),
                ego_route_transforms=ego_route_transforms,
                ego_drive_throttle=float(args.ego_drive_throttle),
                ego_target_speed=float(args.ego_target_speed),
                ego_route_lookahead=float(args.ego_route_lookahead),
                target_crossing=bool(args.target_crossing),
                target_crossing_delay_s=float(args.target_crossing_delay_s),
                target_crossing_speed=float(args.target_crossing_speed),
                target_crossing_trigger_location=target_trigger_location,
                target_crossing_trigger_distance_m=float(args.target_crossing_trigger_distance_m),
                target_crossing_trigger_ttc_s=float(args.target_crossing_trigger_ttc_s),
                target_crossing_trigger_route_lead_m=float(args.target_crossing_trigger_route_lead_m),
                target_crossing_trigger_min_ego_speed_mps=float(
                    args.target_crossing_trigger_min_ego_speed_mps
                ),
                target_motion_mode=target_motion_mode,
                target_crossing_control_speed_override=target_control_speed_override,
                target_prewalk=bool(args.target_prewalk),
                target_prewalk_end_location=target_prewalk_end,
                target_prewalk_speed=float(args.target_prewalk_speed),
                target_prewalk_mode=str(args.target_prewalk_mode),
            )
            event_monitor.spawn()
            # Publish event_monitor into the forward-reference cell so the
            # helper's gate predicate (if any) can read target_started.
            event_monitor_ref[0] = event_monitor

        background = spawn_background_vehicles(
            client,
            world,
            traffic_manager,
            candidates,
            anchor,
            spec,
            vehicle_count,
            rng,
            used_locations,
            autopilot=bool(args.background_autopilot),
        )
        actors.extend(background)

        walkers, walker_controllers = spawn_pedestrians(
            client,
            world,
            anchor,
            spec,
            pedestrian_count,
            rng,
            move=bool(args.move_pedestrians),
        )
        actors.extend(walkers)
        controllers.extend(walker_controllers)

        if args.evidence_pack:
            evidence_recorder = ScenarioEvidenceRecorder(
                world,
                actors,
                event_monitor,
                window_s=float(args.evidence_window_s),
            )

        if sync_world:
            world.tick()
        else:
            world.wait_for_tick()

        actor_rows = []
        for actor in actors:
            role = str(actor.attributes.get("role_name", "walker" if actor.type_id.startswith("walker.") else ""))
            actor_rows.append(actor_record(actor, role))

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_status": git_status_note(),
            "seed": int(args.seed),
            "scenario": {
                "name": spec.name,
                "description": spec.description,
                "occlusion_pair_requested": bool(spec.occlusion_pair),
                "manual_occlusion_crossing": bool(spec.manual_occlusion_crossing),
                "intersection_truck_occlusion": bool(spec.intersection_truck_occlusion),
                "curbside_occlusion": bool(spec.curbside_occlusion),
                "intersection_occlusion_mode": str(spec.intersection_occlusion_mode),
            },
            "carla": {
                "host": args.host,
                "port": int(args.port),
                "map_name": world.get_map().name,
                "sync_world": bool(sync_world),
                "fixed_delta_s": None if not sync_world else float(args.fixed_delta_s),
                "weather": weather_to_dict(world.get_weather()),
            },
            "anchor": anchor_info,
            "spawn_requests": {
                "background_vehicles": int(vehicle_count),
                "pedestrians": int(pedestrian_count),
                "ego_autopilot": bool(args.ego_autopilot),
                "background_autopilot": bool(args.background_autopilot),
                "move_pedestrians": bool(args.move_pedestrians),
                "scripted_ego_drive": bool(args.scripted_ego_drive),
                "ego_drive_mode": str(args.ego_drive_mode),
                "ego_route_choice": str(args.ego_route_choice),
                "ego_target_speed": float(args.ego_target_speed),
                "target_crossing": bool(args.target_crossing),
                "target_crossing_delay_s": float(args.target_crossing_delay_s),
                "target_crossing_speed": float(args.target_crossing_speed),
                "curbside_ego_start_forward_m": float(args.curbside_ego_start_forward_m),
                "target_crossing_motion_mode": str(args.target_crossing_motion_mode),
                "target_crossing_trigger_distance_m": float(args.target_crossing_trigger_distance_m),
                "target_crossing_trigger_ttc_s": float(args.target_crossing_trigger_ttc_s),
                "target_crossing_trigger_route_lead_m": float(args.target_crossing_trigger_route_lead_m),
                "target_crossing_trigger_min_ego_speed_mps": float(
                    args.target_crossing_trigger_min_ego_speed_mps
                ),
                "helper_vehicle": bool(args.helper_vehicle or args.helper_camera_preview or args.helper_drive),
                "helper_drive": bool(args.helper_drive),
                "helper_target_speed": float(args.helper_target_speed),
                "helper_stop_distance_to_conflict_m": float(args.helper_stop_distance_to_conflict_m),
                "helper_camera_preview": bool(args.helper_camera_preview),
                "evidence_pack": bool(args.evidence_pack),
                "evidence_window_s": float(args.evidence_window_s),
                "evidence_camera_buffer_size": int(args.evidence_camera_buffer_size),
                "evidence_camera_stride": int(args.evidence_camera_stride),
                "post_target_collision_hold_s": float(args.post_target_collision_hold_s),
                "stop_on_target_collision": bool(args.stop_on_target_collision),
                "spectator_focus": str(args.spectator_focus),
                "anchor_source": str(args.anchor_source),
                "anchor_spawn_index": int(args.anchor_spawn_index),
                "ego_spawn_index": int(args.ego_spawn_index),
            },
            "spawn_counts": {
                "vehicles_spawned": int(sum(1 for actor in actors if actor.type_id.startswith("vehicle."))),
                "pedestrians_spawned": int(sum(1 for actor in actors if actor.type_id.startswith("walker."))),
                "controllers_spawned": int(len(controllers)),
                "total_actors_spawned": int(len(actors)),
            },
            "spawned_actor_ids": [int(actor.id) for actor in actors],
            "controller_actor_ids": [int(actor.id) for actor in controllers],
            "actors_file": "actors.json",
            "ego_sensors": None if ego_sensor_monitor is None else ego_sensor_monitor.metadata(),
            "helper_camera": None if helper_camera_monitor is None else helper_camera_monitor.metadata(),
            "occlusion_layout": special_layout,
            "occlusion_event": None if event_monitor is None else event_monitor.metadata(),
            "suggested_sensor_placements": suggested_sensor_placements(anchor, traffic_light_id),
            "notes": [
                "Scenario harness run: no perception model, training, or RL.",
                "Ego sensors are smoke-test sensors only; they do not save training data yet.",
                "Use this metadata to reproduce scene layout before adding data collection.",
            ],
        }
        write_outputs(out_dir, manifest, actor_rows)

        print(f"Scenario: {spec.name}")
        print(f"Map: {world.get_map().name}")
        print(f"Anchor: {anchor_info['source']} traffic_light_id={traffic_light_id}")
        print(f"Spawned vehicles: {manifest['spawn_counts']['vehicles_spawned']}")
        print(f"Spawned pedestrians: {manifest['spawn_counts']['pedestrians_spawned']}")
        if ego_sensor_monitor is not None:
            print("Ego sensors: front RGB + radar attached.")
            if ego_sensor_monitor.preview:
                print("Ego RGB preview active. Press q or Esc in the preview to stop.")
        if helper_camera_monitor is not None:
            print("Helper vehicle RGB preview active. Press q or Esc in the preview to stop.")
        if event_monitor is not None:
            print("Occlusion event monitor: collision + target-distance logging active.")
        print(f"Output: {out_dir}")
        hold_scene(
            world,
            float(args.duration_s),
            sync_world,
            ego_sensor_monitor=ego_sensor_monitor,
            helper_camera_monitor=helper_camera_monitor,
            helper_vehicle_controller=helper_vehicle_controller,
            event_monitor=event_monitor,
            evidence_recorder=evidence_recorder,
            stop_on_target_collision=bool(args.stop_on_target_collision),
            post_target_collision_hold_s=float(args.post_target_collision_hold_s),
        )
    except KeyboardInterrupt:
        print("Interrupted; ending scenario.")
    finally:
        if ego_sensor_monitor is not None:
            try:
                ego_sensor_monitor.write_summary(out_dir)
            except Exception as exc:
                print(f"Unable to write ego sensor summary: {exc}")
            ego_sensor_monitor.destroy()
        if helper_camera_monitor is not None:
            try:
                helper_camera_monitor.write_summary(out_dir, "helper_camera_summary.json")
            except Exception as exc:
                print(f"Unable to write helper camera summary: {exc}")
            helper_camera_monitor.destroy()
        if helper_vehicle_controller is not None:
            try:
                helper_vehicle_controller.write_summary(out_dir)
            except Exception as exc:
                print(f"Unable to write helper vehicle summary: {exc}")
            helper_vehicle_controller.stop()
        if event_monitor is not None:
            try:
                event_monitor.write_summary(out_dir)
            except Exception as exc:
                print(f"Unable to write occlusion event summary: {exc}")
        if evidence_recorder is not None:
            try:
                evidence_recorder.write(out_dir, ego_sensor_monitor, helper_camera_monitor)
            except Exception as exc:
                print(f"Unable to write evidence pack: {exc}")
        if event_monitor is not None:
            event_monitor.destroy()
        if not args.keep_actors:
            destroy_actors(actors, controllers)
        else:
            print("Leaving spawned actors in world because --keep-actors was set.")
        if sync_world:
            try:
                traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
            world.apply_settings(original_settings)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
