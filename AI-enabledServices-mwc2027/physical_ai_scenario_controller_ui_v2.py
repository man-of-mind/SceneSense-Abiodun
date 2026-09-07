#!/usr/bin/env python3
"""
Coordinate-aware Physical AI CARLA scenario controller UI (v2).

V2 preserves the deterministic scenario, shared RGB sensor, camera controls,
and replay behavior from v1. It adds restart-safe actor ownership/cleanup, a
configurable synchronous master clock, and an embedded
top-down map selector for the ego route endpoints and ordered via-points:

* Vehicle start/end use the loaded map's predefined road spawn transforms.
* Pedestrian start/end use the deterministic navigation-mesh sample associated
  with the active scenario seed.
* Ordered vehicle and pedestrian via-points can be appended, undone, and
  cleared directly on the map before starting or replaying a scenario.
* The selected vehicle start, ordered intermediate points, and end can be
  planned over CARLA's driving graph and atomically saved to a shared ego-route
  JSON file. Load route restores the selections and its dense road-following
  preview; Ctrl+S and Ctrl+O provide the same actions from the keyboard.
* Vehicle via-points are one ordered Python-agent global plan; Traffic Manager
  controls NPC traffic but cannot supersede an ego intermediate point.
* Vehicle AUTO mode uses a conservative lane-centered local planner with
  curve-speed reduction and a non-latching edge-of-lane throttle limiter.
* Both ego actors spawn before the occluder and NPC population; occupied ego
  points use deterministic catalog fallbacks.
* Driving-lane centerlines and road-adjacent building footprints provide the
  same static-map context used by traffic_lights_map.png.
* The ego vehicle and ego pedestrian each own a dedicated RGB sensor and each
  feed is shown in its own resizable 1080p desktop window, so the scenario
  window itself always shows the top-down map. Pole viewpoints keep sharing one
  relocatable sensor and open a third window only while a pole is the focused
  camera, so the client allocates three render targets at most rather than one
  per traffic-light pole.

The sidebar shows each selected endpoint's x/y/z coordinates. In map mode,
select an endpoint target and click once, or select a via-point target and
left-click several points in traversal order. Use the mouse wheel to zoom,
right-drag to pan, and Reset to fit the loaded world. Save/Load use the path
selected with ``--route-config`` (default: ``ego_vehicle_route_v1.json`` next
to this script).

Everything stays adjustable while the demo runs, from the scenario window: the
Vehicle autonomous and Ped autonomous switches, the Vehicle speed (km/h) and
Ped speed (m/s) steppers, and the camera yaw/pitch sliders. F2 and F3 toggle the
vehicle and pedestrian modes. F1 (or the WASD button) hands WASD to the other
ego actor and, with ui.control_switch_takes_manual, also takes that actor off
AUTO; the actor being released keeps whatever mode it already had, so both ego
actors can be manual at once. Arrow keys and the sliders aim the focused camera;
Tab and the camera buttons change which camera is focused. The scenario window
must hold keyboard focus for WASD and the arrow keys; every control also has a
mouse equivalent in the sidebar and the bottom bar.

This client does not start CARLA or load a map. By default it owns the CARLA
20 Hz synchronous clock while running and restores the prior world settings on
exit. Run only one synchronous master-clock client at a time. When
world.master_clock is false the client stays passive but still drives the ego
route agent and the route gates from its own UI loop.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import carla
import cv2
import numpy as np
import pygame

import physical_ai_scenario_controller_ui_v1 as v1
from ego_route_config import (
    ROUTE_CONFIG_TYPE,
    ROUTE_COORDINATE_SYSTEM,
    ROUTE_SCHEMA_VERSION,
    load_route_config,
    maps_match,
    save_route_config,
)

try:
    from agents.navigation.basic_agent import BasicAgent
except ImportError:
    BasicAgent = None


LOG = logging.getLogger("physical_ai_scenario_controller_v2")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "physical_ai_scenario_config_v2.yaml"
DEFAULT_TRAFFIC_LIGHT_DATA = SCRIPT_DIR / "traffic_lights_data.json"
DEFAULT_ROUTE_CONFIG = SCRIPT_DIR / "ego_vehicle_route_v1.json"

MAP_MODE_LABELS = {
    "vehicle_start": "Vehicle start",
    "vehicle_end": "Vehicle end",
    "vehicle_waypoints": "Vehicle waypoints",
    "pedestrian_start": "Ped start",
    "pedestrian_end": "Ped end",
    "pedestrian_waypoints": "Ped waypoints",
}
MAP_MODE_BUTTON_LABELS = {
    "vehicle_start": "Vehicle start",
    "vehicle_end": "Vehicle end",
    "vehicle_waypoints": "Vehicle via",
    "pedestrian_start": "Ped start",
    "pedestrian_end": "Ped end",
    "pedestrian_waypoints": "Ped via",
}

COLOR_ROAD = (65, 75, 88)
COLOR_GRID = (43, 50, 61)
COLOR_BUILDING_FILL = (42, 42, 42)
COLOR_BUILDING_EDGE = (64, 64, 64)
COLOR_LANE_CENTERLINE = (85, 85, 85)
COLOR_PLOT_BG = (18, 23, 30)
COLOR_VEHICLE_POINT = (72, 150, 220)
COLOR_PEDESTRIAN_POINT = (82, 195, 178)
COLOR_START = (65, 210, 225)
COLOR_END = (255, 173, 68)
COLOR_ROUTE_VEHICLE = (87, 171, 255)
COLOR_ROUTE_PEDESTRIAN = (211, 132, 255)
COLOR_WAYPOINT_TEXT = (235, 240, 247)
OWNED_ROLE_PREFIX = "physical_ai_"
MIN_BUILDING_HEIGHT_M = 2.0
MIN_BUILDING_AREA_M2 = 20.0
MIN_BUILDING_VOLUME_M3 = 80.0
BUILDING_ROAD_PROXIMITY_M = 20.0
BUILDING_EDGE_SAMPLE_M = 5.0

EGO_VEHICLE_VIEW_KEY = "ego_vehicle"
EGO_PEDESTRIAN_VIEW_KEY = "ego_pedestrian"
EGO_VIEW_KEYS = (EGO_VEHICLE_VIEW_KEY, EGO_PEDESTRIAN_VIEW_KEY)

DEFAULT_VIDEO_WINDOW_SIZE = (1920, 1080)

# Bottom-bar design rows, shared by the map-selection column on the left and the
# camera-aim column on the right so both read as one strip.
BOTTOM_BUTTON_WIDTH = 152
BOTTOM_ROWS = {
    "title": 18,
    "subtitle": 44,
    "buttons_1": 70,
    "buttons_2": 112,
    "buttons_3": 154,
    "yaw_track": 140,
    "pitch_track": 196,
    "text_1": 214,
    "text_2": 238,
}

# Every panel coordinate below is expressed in this design space and multiplied
# by the UI scale, so one layout serves a 1440x900 window and a 2560x1600 one.
DESIGN_PANEL_WIDTH = 390
DESIGN_PANEL_HEIGHT = 1000
DESIGN_BOTTOM_HEIGHT = 260
DESIGN_FONT_SIZE = 17
DESIGN_SMALL_FONT_SIZE = 14
DESIGN_TITLE_FONT_SIZE = 23
DESIGN_MAP_LABEL_FONT_SIZE = 13
MINIMUM_UI_SCALE = 0.85
MAXIMUM_UI_SCALE = 2.40
WINDOW_MODES = ("borderless", "windowed", "fullscreen")


def desktop_size() -> Tuple[int, int]:
    """The primary display's pixel size, initialising SDL video if needed."""
    pygame.init()
    try:
        sizes = pygame.display.get_desktop_sizes()
        if sizes:
            return int(sizes[0][0]), int(sizes[0][1])
    except (AttributeError, pygame.error):
        pass
    info = pygame.display.Info()
    return int(info.current_w), int(info.current_h)


def desktop_work_area() -> Tuple[int, int, int, int]:
    """
    The desktop area left free by panels and docks, as (x, y, width, height).

    pygame exposes no work-area query, so the X11 ``_NET_WORKAREA`` property is
    read directly; on this desktop it reports the 70 px Ubuntu dock and the
    27 px top bar. Falls back to the full display when unavailable.
    """
    width, height = desktop_size()
    try:
        result = subprocess.run(
            ["xprop", "-root", "-notype", "_NET_WORKAREA"],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.info("Unable to query _NET_WORKAREA (%s); using the full display", exc)
        return 0, 0, width, height
    if result.returncode != 0 or "=" not in result.stdout:
        LOG.info("No _NET_WORKAREA reported; using the full display")
        return 0, 0, width, height
    try:
        numbers = [
            int(value.strip())
            for value in result.stdout.split("=", 1)[1].split(",")[:4]
        ]
        area_x, area_y, area_width, area_height = numbers
    except (IndexError, ValueError):
        LOG.info("Could not parse _NET_WORKAREA; using the full display")
        return 0, 0, width, height
    if area_width <= 0 or area_height <= 0:
        return 0, 0, width, height
    LOG.info(
        "Desktop work area: %dx%d at (%d, %d) of a %dx%d display",
        area_width,
        area_height,
        area_x,
        area_y,
        width,
        height,
    )
    return area_x, area_y, area_width, area_height


def resolve_window_geometry(
    ui_config: Dict[str, Any],
) -> Tuple[int, int, int, float, str]:
    """
    Resolve the window size, SDL flags and UI scale from the ui config block.

    ``window_size: auto`` fills the whole display, which is what the demo wants
    on a laptop panel. The UI scale is derived from the resulting height so the
    sidebar, fonts and widgets grow with the window instead of shrinking into a
    corner of a 2560x1600 screen.

    ``windowed`` is the default: the window is sized to the desktop work area
    minus a margin and a title-bar allowance, so it never covers the Ubuntu dock
    or the top bar. pygame 2 cannot position a window, but a window that fits
    inside the work area is placed inside it by the window manager.
    """
    mode = str(ui_config.get("window_mode", "windowed")).lower()
    if mode not in WINDOW_MODES:
        LOG.warning(
            "Unknown ui.window_mode %r; using windowed", ui_config.get("window_mode")
        )
        mode = "windowed"
    requested = ui_config.get("window_size", "auto")
    screen_width, screen_height = desktop_size()
    if isinstance(requested, (list, tuple)) and len(requested) == 2:
        # An explicit size is honoured as given, even on a display that reports
        # itself smaller (headless/dummy SDL drivers do).
        width, height = int(requested[0]), int(requested[1])
    else:
        if not (
            isinstance(requested, str) and requested.strip().lower() == "auto"
        ):
            LOG.warning(
                "Unsupported ui.window_size %r; using the display size", requested
            )
        if mode == "windowed":
            # Fit inside the dock/panel-free work area, leaving a margin the
            # window manager can use for placement plus room for the title bar.
            _, _, area_width, area_height = desktop_work_area()
            margin = max(0, int(ui_config.get("window_margin_px", 24)))
            title_allowance = max(
                0, int(ui_config.get("title_bar_allowance_px", 44))
            )
            width = area_width - 2 * margin
            height = area_height - margin - title_allowance
        else:
            width, height = screen_width, screen_height
        # An auto size must never exceed the display: the usability floors
        # below must not push a windowed instance off a small screen.
        width = min(width, screen_width)
        height = min(height, screen_height)
    width = max(960, width)
    height = max(640, height)
    flags = 0
    if mode == "borderless":
        flags = pygame.NOFRAME
    elif mode == "fullscreen":
        flags = pygame.FULLSCREEN
    requested_scale = ui_config.get("scale", "auto")
    if isinstance(requested_scale, (int, float)):
        scale = float(requested_scale)
    else:
        scale = height / float(DESIGN_PANEL_HEIGHT)
    scale = v1.clamp(scale, MINIMUM_UI_SCALE, MAXIMUM_UI_SCALE)
    LOG.info(
        "UI geometry: %dx%d (%s, display %dx%d), scale %.2f",
        width,
        height,
        mode,
        screen_width,
        screen_height,
        scale,
    )
    return width, height, flags, scale, mode


def carla_image_to_bgr(image: carla.Image) -> Any:
    """
    View a CARLA RGB frame as a BGR array without copying or converting.

    CARLA delivers BGRA, which is already OpenCV's channel order, so dropping
    the alpha channel is the whole conversion. The array is a read-only view of
    the sensor buffer; callers that draw on it must copy first.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    expected = image.width * image.height * 4
    if array.size != expected:
        raise ValueError(
            "camera frame has {} bytes; expected {}".format(array.size, expected)
        )
    return array.reshape((image.height, image.width, 4))[:, :, :3]


def draw_ground_truth_boxes_bgr(
    image: Any,
    actors: Sequence[carla.Actor],
    camera_transform: carla.Transform,
    fov_deg: float,
    max_distance_m: float,
    overlap_threshold: float,
    excluded_ids: Sequence[int],
    occluder_id: Optional[int],
    line_scale: float = 1.0,
) -> None:
    """
    Draw LOS/NLOS ground-truth boxes straight onto a BGR frame, in place.

    The projection and the nearer-box overlap test are v1's, so the estimate
    shown in the separate video windows matches what the embedded panes drew.
    Rendering at the sensor's own resolution keeps the labels sharp instead of
    scaling a pre-rendered overlay.
    """
    height, width = image.shape[:2]
    calibration = v1.camera_calibration(width, height, fov_deg)
    excluded = {int(item) for item in excluded_ids}
    projected = []
    for actor in actors:
        try:
            actor_id = int(actor.id)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
        if actor_id in excluded or not v1.actor_alive(actor):
            continue
        result = v1.project_actor_box(
            actor, camera_transform, calibration, width, height
        )
        if result is None:
            continue
        rect, depth = result
        if depth > max_distance_m:
            continue
        projected.append((depth, rect, actor))
    projected.sort(key=lambda item: item[0])

    thickness = max(1, int(round(2 * line_scale)))
    font_scale = 0.45 * max(0.6, line_scale)
    foreground: List[pygame.Rect] = []
    for depth, rect, actor in projected:
        nlos = any(
            v1.intersection_fraction(rect, nearer) >= overlap_threshold
            for nearer in foreground
        )
        # v1's palette is RGB; OpenCV wants BGR.
        color = tuple(reversed(v1.COLOR_ORANGE if nlos else v1.COLOR_GREEN))
        kind = "PED" if str(actor.type_id).startswith("walker.") else "VEH"
        if occluder_id is not None and int(actor.id) == int(occluder_id):
            kind = "OCC"
        cv2.rectangle(
            image,
            (rect.left, rect.top),
            (rect.right, rect.bottom),
            color,
            thickness,
        )
        label = "{} {} id={} {:.1f}m".format(
            "NLOS*" if nlos else "LOS*", kind, actor.id, depth
        )
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        text_top = max(0, rect.top - text_height - 4)
        cv2.rectangle(
            image,
            (rect.left, text_top),
            (rect.left + text_width + 4, text_top + text_height + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            label,
            (rect.left + 2, text_top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        foreground.append(rect)


class VideoWindowBank:
    """
    The live camera feeds, each in its own resizable OpenCV window.

    The scenario UI owns only the top-down map, so the streams are presented as
    independent desktop windows the operator can move, resize or push to another
    display. OpenCV is used rather than a second pygame surface because a CARLA
    frame is already in its pixel format and because SDL allows one window per
    display context.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        positions: Dict[str, Tuple[int, int]],
        hud_scale: float = 1.0,
    ) -> None:
        self.width, self.height = int(size[0]), int(size[1])
        self.positions = dict(positions)
        self.hud_scale = float(hud_scale)
        self.open_windows: Dict[str, str] = {}
        self.frame_ids: Dict[str, int] = {}

    @staticmethod
    def window_title(view: v1.CameraView) -> str:
        return "CARLA {}".format(view.label)

    def ensure(self, view: v1.CameraView) -> str:
        """Create the window for a view on first use and return its title."""
        title = self.open_windows.get(view.key)
        if title is not None:
            return title
        title = self.window_title(view)
        # WINDOW_GUI_NORMAL drops OpenCV's Qt toolbar and status bar so the
        # window is just the picture plus the window manager's title bar.
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(title, self.width, self.height)
        # Pole views are keyed per traffic light, so they share one position.
        position = self.positions.get(view.key)
        if position is None and view.kind == "pole":
            position = self.positions.get("pole")
        if position is not None:
            cv2.moveWindow(title, int(position[0]), int(position[1]))
        self.open_windows[view.key] = title
        LOG.info(
            "Opened %dx%d video window '%s'", self.width, self.height, title
        )
        return title

    def show(self, view: v1.CameraView, image: Any, frame_id: int) -> None:
        cv2.imshow(self.ensure(view), image)
        self.frame_ids[view.key] = int(frame_id)

    def close(self, view_key: str) -> None:
        title = self.open_windows.pop(view_key, None)
        self.frame_ids.pop(view_key, None)
        if title is None:
            return
        try:
            cv2.destroyWindow(title)
            cv2.waitKey(1)
        except cv2.error as exc:
            LOG.debug("Unable to close video window %s: %s", title, exc)

    def close_all(self) -> None:
        for view_key in list(self.open_windows):
            self.close(view_key)

    @staticmethod
    def pump() -> None:
        """Service the OpenCV window event loop; required for them to redraw."""
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass


class ScaledStepper(v1.Stepper):
    """v1 stepper whose label is vertically centred on its scaled rects."""

    label_x = 18

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        label = font.render(self.label, True, v1.COLOR_TEXT)
        screen.blit(
            label,
            (self.label_x, self.minus.centery - label.get_height() // 2),
        )
        for rect, glyph_text in ((self.minus, "-"), (self.plus, "+")):
            pygame.draw.rect(screen, v1.COLOR_FIELD, rect, border_radius=4)
            pygame.draw.rect(screen, v1.COLOR_BORDER, rect, 1, border_radius=4)
            glyph = font.render(glyph_text, True, v1.COLOR_TEXT)
            screen.blit(glyph, glyph.get_rect(center=rect.center))
        pygame.draw.rect(screen, v1.COLOR_BG, self.value_rect, border_radius=3)
        text_value = (
            "{}".format(self.get())
            if self.integer
            else "{:.2f}".format(self.value)
        )
        text = font.render(text_value, True, v1.COLOR_ACCENT)
        screen.blit(text, text.get_rect(center=self.value_rect.center))


class ScaledToggle(v1.Toggle):
    """v1 toggle with a vertically centred label."""

    label_x = 18

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        label = font.render(self.label, True, v1.COLOR_TEXT)
        screen.blit(
            label,
            (self.label_x, self.rect.centery - label.get_height() // 2),
        )
        pygame.draw.rect(
            screen,
            v1.COLOR_GREEN if self.value else v1.COLOR_FIELD,
            self.rect,
            border_radius=4,
        )
        text = font.render(
            "ON" if self.value else "OFF",
            True,
            v1.COLOR_BG if self.value else v1.COLOR_TEXT,
        )
        screen.blit(text, text.get_rect(center=self.rect.center))


class ScaledCycleField(v1.CycleField):
    """v1 cycle field with a vertically centred label."""

    label_x = 18

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        label = font.render(self.label, True, v1.COLOR_TEXT)
        screen.blit(
            label,
            (self.label_x, self.rect.centery - label.get_height() // 2),
        )
        pygame.draw.rect(screen, v1.COLOR_FIELD, self.rect, border_radius=4)
        text = font.render(self.value.upper(), True, v1.COLOR_ACCENT)
        screen.blit(text, text.get_rect(center=self.rect.center))


class ScaledSlider(v1.Slider):
    """v1 slider with scale-aware label spacing, track and knob."""

    label_gap = 28
    track_width = 6
    knob_radius = 9

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        label = font.render(
            "{}: {:+.1f} deg".format(self.label, self.value), True, v1.COLOR_TEXT
        )
        screen.blit(label, (self.rect.left, self.rect.top - self.label_gap))
        pygame.draw.line(
            screen,
            v1.COLOR_BORDER,
            self.rect.midleft,
            self.rect.midright,
            self.track_width,
        )
        fraction = (self.value - self.minimum) / (self.maximum - self.minimum)
        x = int(self.rect.left + fraction * self.rect.width)
        pygame.draw.line(
            screen,
            v1.COLOR_ACCENT,
            self.rect.midleft,
            (x, self.rect.centery),
            self.track_width,
        )
        pygame.draw.circle(
            screen, v1.COLOR_TEXT, (x, self.rect.centery), self.knob_radius
        )


def format_location(location: Optional[carla.Location]) -> str:
    if location is None:
        return "(x=--, y=--, z=--)"
    return "(x={:+.2f}, y={:+.2f}, z={:+.2f})".format(
        location.x, location.y, location.z
    )


def point_location(point: Any) -> carla.Location:
    if isinstance(point, carla.Transform):
        return point.location
    return point


def actor_speed_mps(actor: Optional[carla.Actor]) -> float:
    """Return an actor's ground speed without raising on a dead actor."""
    if actor is None:
        return 0.0
    try:
        velocity = actor.get_velocity()
    except (AttributeError, RuntimeError):
        return 0.0
    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


def sensor_is_listening(sensor: Optional[carla.Sensor]) -> Optional[bool]:
    """
    Report a sensor wrapper's subscription state across CARLA API generations.

    CARLA 0.10 exposes ``Sensor.is_listening`` as a bound method while older
    releases expose a plain attribute.  ``bool(sensor.is_listening)`` is
    therefore always True on 0.10, which silently defeats any caller that tries
    to skip stop() for a wrapper it never subscribed.  None means unknown.
    """
    if sensor is None:
        return None
    try:
        state = sensor.is_listening
    except (AttributeError, RuntimeError):
        return None
    if callable(state):
        try:
            return bool(state())
        except (RuntimeError, TypeError):
            return None
    return bool(state)


# BasicAgent is optional at import time, so the ego agent subclass needs a base
# that exists either way. _create_vehicle_route_agent still refuses to build an
# agent when the CARLA navigation package is missing.
_AGENT_BASE = BasicAgent if BasicAgent is not None else object


class EgoRouteAgent(_AGENT_BASE):
    """
    BasicAgent that can ignore chosen actors and reports why it stopped.

    Two demo-specific behaviors are needed on top of the stock agent:

    * The scenario occluder is a parked prop whose purpose is to break line of
      sight.  Stock BasicAgent treats every ``vehicle.*`` actor as an obstacle,
      so an occluder anywhere near the ego lane holds an emergency stop forever
      and the ego never finishes its route.  Ignoring that one actor id keeps
      collision avoidance active for real traffic.
    * ``last_hazard`` records whether the most recent step braked for a vehicle
      or a red light, which lets the caller distinguish a legitimate stop from a
      controller stall.
    """

    def __init__(
        self,
        vehicle: carla.Vehicle,
        target_speed: float,
        opt_dict: Dict[str, Any],
        map_inst: Optional[carla.Map] = None,
        grp_inst: Optional[Any] = None,
        ignored_actor_ids: Sequence[int] = (),
        ignore_lights: bool = False,
        ignore_signs: bool = False,
    ) -> None:
        if BasicAgent is None:
            raise RuntimeError("CARLA BasicAgent is unavailable")
        super().__init__(
            vehicle,
            target_speed=target_speed,
            opt_dict=opt_dict,
            map_inst=map_inst,
            grp_inst=grp_inst,
        )
        self.ignored_actor_ids = {
            int(actor_id) for actor_id in ignored_actor_ids
        }
        self.last_hazard: Optional[str] = None
        if ignore_lights:
            self.ignore_traffic_lights(True)
        if ignore_signs:
            self.ignore_stop_signs(True)

    def ignore_actor(self, actor: Optional[carla.Actor]) -> None:
        if actor is None:
            return
        try:
            self.ignored_actor_ids.add(int(actor.id))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass

    def run_step(self) -> carla.VehicleControl:
        """BasicAgent.run_step with an obstacle allowlist and hazard reporting."""
        speed_mps = actor_speed_mps(self._vehicle)
        vehicles = [
            actor
            for actor in self._world.get_actors().filter("*vehicle*")
            if int(actor.id) not in self.ignored_actor_ids
        ]
        hazard: Optional[str] = None
        max_vehicle_distance = (
            self._base_vehicle_threshold + self._speed_ratio * speed_mps
        )
        affected, _, _ = self._vehicle_obstacle_detected(
            vehicles, max_vehicle_distance
        )
        if affected:
            hazard = "vehicle"
        if hazard is None:
            max_light_distance = (
                self._base_tlight_threshold + self._speed_ratio * speed_mps
            )
            affected, _ = self._affected_by_traffic_light(
                self._lights_list, max_light_distance
            )
            if affected:
                hazard = "traffic_light"
        control = self._local_planner.run_step()
        if hazard is not None:
            control = self.add_emergency_stop(control)
        self.last_hazard = hazard
        return control


class EgoCameraStream:
    """One dedicated ego RGB sensor plus its own latest-frame mailbox."""

    def __init__(
        self,
        view: v1.CameraView,
        sensor: carla.Sensor,
        transform: carla.Transform,
    ) -> None:
        self.view = view
        self.sensor: Optional[carla.Sensor] = sensor
        self.mailbox = v1.LatestMailbox()
        self.listening = False
        self.last_transform = v1.copy_transform(transform)
        # The relative yaw/pitch last written to the server. While it matches
        # the view, the rigid attachment alone keeps the camera in place.
        self.applied_aim: Optional[Tuple[float, float]] = None


class MultiStreamCameraDirector(v1.CameraDirector):
    """
    Dedicated RGB sensors for both ego actors plus one relocatable pole sensor.

    v1 routed every viewpoint through a single render target, which allowed only
    one live stream at a time.  The Physical AI demo needs the driver feed and
    the pedestrian feed simultaneously, so each ego actor owns a sensor while
    the traffic-light-pole viewpoints keep sharing the single relocatable
    sensor.  That is three render targets in total, which preserves the reason
    v1 shared a sensor -- avoiding one Vulkan render target per pole.
    """

    def _spawn_sensor(self) -> None:
        self.ego_streams: Dict[str, EgoCameraStream] = {}
        self.pole_views: List[v1.CameraView] = [
            view for view in self.views if view.kind == "pole"
        ]
        self.pole_stream_enabled = False
        self._pole_transform_current = False
        # A queued set_transform lands on the next tick, so an aim write is
        # composed from the parent pose projected one control period ahead.
        self.lead_seconds = max(
            0.0,
            float(self.config.get("world", {}).get("fixed_delta_seconds", 0.05)),
        )
        self._last_pole_view: Optional[v1.CameraView] = (
            self.pole_views[0] if self.pole_views else None
        )
        for view in self.views:
            if view.key not in EGO_VIEW_KEYS:
                continue
            if view.key == EGO_PEDESTRIAN_VIEW_KEY:
                self._clamp_pedestrian_mount(view)
            # Rigid attachment: the server applies mount + aim relative to the
            # actor every tick, so the camera never lags behind it. Positioning
            # an unattached sensor from get_transform() instead reproduced the
            # previous tick's pose, which read as the view sliding backwards the
            # faster the actor moved.
            relative = carla.Transform(
                v1.copy_location(view.mount),
                carla.Rotation(pitch=float(view.pitch), yaw=float(view.yaw)),
            )
            sensor = self.world.spawn_actor(
                self._rgb_blueprint("physical_ai_{}_rgb".format(view.key)),
                relative,
                attach_to=view.actor,
                attachment_type=carla.AttachmentType.Rigid,
            )
            stream = EgoCameraStream(view, sensor, self.transform_for(view))
            stream.applied_aim = (round(float(view.yaw), 3), round(float(view.pitch), 3))
            sensor.listen(stream.mailbox.push)
            stream.listening = True
            self.ego_streams[view.key] = stream
            LOG.info(
                "Spawned dedicated %s RGB camera actor %d, rigidly attached to "
                "actor %d",
                view.key,
                sensor.id,
                view.actor.id,
            )
        if self.pole_views:
            self.sensor = self.world.spawn_actor(
                self._rgb_blueprint("physical_ai_shared_pole_rgb"),
                self.transform_for(self.pole_views[0]),
            )
            LOG.info(
                "Spawned shared pole RGB camera actor %d for %d pole viewpoints",
                self.sensor.id,
                len(self.pole_views),
            )
        else:
            self.sensor = None
            LOG.warning(
                "No traffic-light poles resolved; only the ego streams are live"
            )
        self.listening = False

    def _rgb_blueprint(self, role_name: str) -> carla.ActorBlueprint:
        camera_cfg = self.config["camera"]
        width, height = map(int, camera_cfg["resolution"])
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(width))
        blueprint.set_attribute("image_size_y", str(height))
        blueprint.set_attribute("fov", str(float(camera_cfg["fov_deg"])))
        blueprint.set_attribute(
            "sensor_tick", str(float(camera_cfg["sensor_tick_s"]))
        )
        if blueprint.has_attribute("gamma"):
            blueprint.set_attribute("gamma", str(float(camera_cfg["gamma"])))
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)
        return blueprint

    @staticmethod
    def _clamp_pedestrian_mount(view: v1.CameraView) -> None:
        """Keep the head camera outside the walker mesh, as the reference client does."""
        try:
            bounds = view.actor.bounding_box
        except (AttributeError, RuntimeError):
            return
        minimum_x = float(bounds.location.x + bounds.extent.x) + 0.05
        if float(view.mount.x) >= minimum_x:
            return
        LOG.warning(
            "Ego pedestrian camera_mount x=%.2f m is inside the walker mesh; "
            "using %.2f m so the head does not occlude the feed",
            float(view.mount.x),
            minimum_x,
        )
        view.mount = carla.Location(
            x=minimum_x,
            y=float(view.mount.y),
            z=float(view.mount.z),
        )

    def owned_sensors(self) -> List[carla.Sensor]:
        """Every sensor wrapper this director created, for ordered cleanup."""
        sensors = [
            stream.sensor
            for stream in getattr(self, "ego_streams", {}).values()
            if stream.sensor is not None
        ]
        if self.sensor is not None:
            sensors.append(self.sensor)
        return sensors

    def mailboxes(self) -> List[v1.LatestMailbox]:
        boxes = [
            stream.mailbox for stream in getattr(self, "ego_streams", {}).values()
        ]
        boxes.append(self.mailbox)
        return boxes

    def forget_sensors(self) -> None:
        """Drop sensor references after an external batch destroy confirmed them."""
        for stream in getattr(self, "ego_streams", {}).values():
            stream.sensor = None
            stream.listening = False
        self.sensor = None
        self.listening = False
        self.pole_stream_enabled = False

    def invalidate_pole_transform(self) -> None:
        """Force the next update to re-apply the shared sensor pose."""
        self._pole_transform_current = False

    def stream_for(self, view_key: str) -> Optional[EgoCameraStream]:
        return getattr(self, "ego_streams", {}).get(view_key)

    def view_by_key(self, view_key: str) -> Optional[v1.CameraView]:
        for view in self.views:
            if view.key == view_key:
                return view
        return None

    @property
    def active_pole_view(self) -> Optional[v1.CameraView]:
        """The pole viewpoint the shared sensor currently renders."""
        if self.selected.kind == "pole":
            return self.selected
        return self._last_pole_view

    def set_pole_stream_enabled(self, enabled: bool) -> None:
        """Subscribe the shared pole sensor only while a pole pane is visible."""
        enabled = bool(enabled) and self.sensor is not None
        if enabled == self.pole_stream_enabled:
            return
        try:
            if enabled:
                self.mailbox.clear()
                self.update_transform(force=True)
                self.sensor.listen(self.mailbox.push)
            else:
                self.sensor.stop()
                self.mailbox.clear()
        except RuntimeError as exc:
            LOG.warning("Unable to change the pole stream state: %s", exc)
            return
        self.pole_stream_enabled = enabled
        self.listening = enabled

    def select(self, index: int) -> None:
        if not self.views:
            return
        index %= len(self.views)
        if index == self.selected_index:
            return
        self.selected_index = index
        current = self.selected
        if current.kind == "pole":
            changed_pole = current is not self._last_pole_view
            self._last_pole_view = current
            self._pole_transform_current = False
            if changed_pole:
                # Drop any frame still rendered from the previous pole pose so
                # the new viewpoint never shows one stale image.
                self.mailbox.clear()
        self.update_transform(force=True)
        LOG.info("Selected camera view: %s", current.label)

    def _sync_ego_camera(self, stream: EgoCameraStream, force: bool) -> None:
        """
        Refresh one ego camera's world pose and push its aim only when it moved.

        The sensor is rigidly attached, so its position needs no per-frame
        write: the server already tracks the actor exactly. Only a yaw/pitch
        change needs a transform, and because CARLA's set_transform on an
        attached actor takes a world pose and re-defines the rigid offset, that
        pose is composed from the parent projected one control period ahead so
        the re-defined offset stays the configured mount.
        """
        view = stream.view
        if v1.actor_alive(stream.sensor):
            try:
                # The server-side pose the last frame was actually rendered at,
                # which is what the ground-truth overlay must project with.
                stream.last_transform = stream.sensor.get_transform()
            except RuntimeError:
                pass
        desired = (round(float(view.yaw), 3), round(float(view.pitch), 3))
        if not force and desired == stream.applied_aim:
            return
        if not v1.actor_alive(stream.sensor) or not v1.actor_alive(view.actor):
            return
        try:
            parent = view.actor.get_transform()
            velocity = view.actor.get_velocity()
        except RuntimeError:
            return
        lead = self.lead_seconds
        predicted = carla.Transform(
            carla.Location(
                x=float(parent.location.x + velocity.x * lead),
                y=float(parent.location.y + velocity.y * lead),
                z=float(parent.location.z + velocity.z * lead),
            ),
            parent.rotation,
        )
        try:
            stream.sensor.set_transform(
                carla.Transform(
                    v1.local_to_world(predicted, view.mount),
                    carla.Rotation(
                        pitch=parent.rotation.pitch + float(view.pitch),
                        yaw=parent.rotation.yaw + float(view.yaw),
                        roll=parent.rotation.roll,
                    ),
                )
            )
        except RuntimeError:
            return
        stream.applied_aim = desired

    def update_transform(self, force: bool = False) -> None:
        for stream in getattr(self, "ego_streams", {}).values():
            self._sync_ego_camera(stream, force)
        pole_view = self.active_pole_view
        if pole_view is not None:
            transform = self.transform_for(pole_view)
            if v1.actor_alive(self.sensor) and (
                force or not self._pole_transform_current
            ):
                try:
                    self.sensor.set_transform(transform)
                    self._pole_transform_current = True
                except RuntimeError:
                    pass
            if self.selected.kind == "pole":
                self.last_transform = transform
        selected_stream = self.stream_for(self.selected.key)
        if selected_stream is not None:
            self.last_transform = selected_stream.last_transform

    def transform_of(self, view_key: str) -> carla.Transform:
        stream = self.stream_for(view_key)
        if stream is not None:
            return stream.last_transform
        return self.last_transform

    def destroy(self) -> None:
        for stream in list(getattr(self, "ego_streams", {}).values()):
            sensor = stream.sensor
            stream.sensor = None
            if sensor is None:
                continue
            try:
                if stream.listening:
                    sensor.stop()
                    stream.listening = False
            except RuntimeError:
                pass
            stream.mailbox.clear()
            try:
                if v1.actor_alive(sensor):
                    sensor.destroy()
            except RuntimeError:
                pass
        self.ego_streams = {}
        super().destroy()


class ScenarioControllerV2(v1.ScenarioController):
    """V1 scenario controller with stable preview catalogs for the map UI."""

    def __init__(
        self,
        client: carla.Client,
        world: carla.World,
        traffic_light_data: Path,
        map_waypoint_spacing_m: float,
        master_clock: bool = True,
        fixed_delta_seconds: float = 0.05,
        traffic_manager_port: int = 8000,
        restore_world_settings: bool = True,
        force_async_on_exit: bool = False,
    ) -> None:
        super().__init__(client, world, traffic_light_data)
        self.master_clock = bool(master_clock)
        self.fixed_delta_seconds = float(fixed_delta_seconds)
        if self.fixed_delta_seconds <= 0.0:
            raise ValueError("world.fixed_delta_seconds must be greater than zero")
        self.traffic_manager_port = int(traffic_manager_port)
        self.restore_world_settings = bool(restore_world_settings)
        initial_settings = self.world.get_settings()
        self._original_synchronous_mode = bool(initial_settings.synchronous_mode)
        self._original_fixed_delta_seconds = initial_settings.fixed_delta_seconds
        if force_async_on_exit:
            self._original_synchronous_mode = False
            self._original_fixed_delta_seconds = None
        elif self._original_synchronous_mode and self.master_clock:
            # Another master-clock client already owns the world. The recorded
            # "original" state is that client's, so exiting would hand back a
            # synchronous world with nobody ticking it, leaving the server
            # frozen. Run only one master-clock client, or use
            # --force-async-on-exit when no other master remains.
            LOG.warning(
                "The world is already synchronous (fixed_delta_seconds=%s). "
                "Another master-clock client may be running; on exit this "
                "client will restore synchronous mode, which freezes the "
                "server if no other client ticks it. Pass "
                "--force-async-on-exit if no other master client exists.",
                initial_settings.fixed_delta_seconds,
            )
        self._clock_configured = False
        self._next_world_tick_at = time.monotonic()
        waypoint_spacing = max(1.0, float(map_waypoint_spacing_m))
        road_waypoints = list(self.map.generate_waypoints(waypoint_spacing))
        self.vehicle_spawn_preview = [
            v1.copy_transform(transform) for transform in self.map.get_spawn_points()
        ]
        if not self.vehicle_spawn_preview:
            raise RuntimeError("loaded CARLA map has no vehicle spawn points")
        self.road_preview = [
            v1.copy_location(waypoint.transform.location)
            for waypoint in road_waypoints
        ]
        if not self.road_preview:
            self.road_preview = [
                v1.copy_location(transform.location)
                for transform in self.vehicle_spawn_preview
            ]
        self.road_polylines_preview = self._build_road_polylines(
            road_waypoints,
            waypoint_spacing,
        )
        self.building_footprints_preview = self._build_building_footprints(
            self.road_preview
        )
        LOG.info(
            "Top-down selector geometry: %d lane polylines, %d building footprints",
            len(self.road_polylines_preview),
            len(self.building_footprints_preview),
        )
        self._navigation_preview_seed: Optional[int] = None
        self._navigation_preview: List[carla.Location] = []
        # Keep wrappers for any server-confirmed cleanup survivor. In
        # particular, a live carla.Sensor wrapper must not go out of scope
        # before a later retry, or libcarla reports a leaked simulation actor.
        self._cleanup_survivors: List[carla.Actor] = []
        self._pending_vehicle_route_locations: List[carla.Location] = []
        self._pending_vehicle_route_segments: List[List[carla.Location]] = []
        self._pending_vehicle_route_targets: List[carla.Location] = []
        self._pending_vehicle_agent_plan: List[Tuple[Any, Any]] = []
        self._pending_pedestrian_route_locations: List[carla.Location] = []
        self._configured_vehicle_waypoint_indices: List[int] = []
        self._configured_pedestrian_waypoint_indices: List[int] = []
        self._vehicle_route_segment_queue: List[
            Tuple[List[carla.Location], carla.Location]
        ] = []
        self._vehicle_route_target: Optional[carla.Location] = None
        self._vehicle_route_target_number = 0
        self._vehicle_route_total_targets = 0
        self._vehicle_waypoint_reach_threshold_m = 8.0
        self._vehicle_scripted_route_requested = False
        self._vehicle_route_agent: Optional[Any] = None
        self._vehicle_cruise_speed_kmh = 18.0
        self._vehicle_curve_speed_kmh = 10.0
        self._vehicle_lane_guard_fraction = 0.80
        self._vehicle_lane_guard_brake_speed_mps = 2.0
        self._vehicle_lane_guard_recovery_throttle = 0.30
        self._vehicle_waypoint_purge_base_m = 3.0
        self._vehicle_waypoint_purge_speed_ratio = 0.5
        self._vehicle_stuck_recovery_attempts = 3
        self._vehicle_stuck_recovery_used = 0
        self._vehicle_reverse_until = 0.0
        self._vehicle_lane_guard_active = False
        self._vehicle_lane_guard_status = ""
        self._vehicle_stall_warning_seconds = 6.0
        self._vehicle_stalled_since: Optional[float] = None
        self._next_vehicle_stall_log_at = 0.0
        self._next_vehicle_safety_log_at = 0.0
        self._next_control_at = time.monotonic()
        self._route_detour_warnings: List[str] = []
        self._pedestrian_route_queue: List[carla.Location] = []
        self._pedestrian_route_target: Optional[carla.Location] = None
        self._pedestrian_route_target_number = 0
        self._pedestrian_route_total_targets = 0
        self._pedestrian_waypoint_reach_threshold_m = 1.5
        self._pedestrian_guide_active = False
        self._next_pedestrian_guide_at = 0.0
        self.occluder_blueprint_resolution: Optional[Dict[str, Any]] = None
        self.occluder_lateral_offset_resolved: Optional[Dict[str, Any]] = None
        if self.master_clock:
            self._configure_master_clock()

    @staticmethod
    def _build_road_polylines(
        waypoints: Sequence[Any],
        sample_spacing: float,
    ) -> List[List[carla.Location]]:
        """Group driving waypoints into ordered lane centerlines."""
        lane_samples: Dict[Tuple[int, int, int], List[Tuple[float, carla.Location]]] = {}
        for waypoint in waypoints:
            try:
                if waypoint.lane_type != carla.LaneType.Driving:
                    continue
                key = (
                    int(waypoint.road_id),
                    int(waypoint.section_id),
                    int(waypoint.lane_id),
                )
                lane_samples.setdefault(key, []).append(
                    (
                        float(waypoint.s),
                        v1.copy_location(waypoint.transform.location),
                    )
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue

        polylines: List[List[carla.Location]] = []
        minimum_separation = max(0.1, float(sample_spacing) * 0.25)
        for samples in lane_samples.values():
            samples.sort(key=lambda item: item[0])
            polyline: List[carla.Location] = []
            for _, location in samples:
                if polyline and polyline[-1].distance(location) < minimum_separation:
                    continue
                polyline.append(location)
            if len(polyline) >= 2:
                polylines.append(polyline)
        return polylines

    @staticmethod
    def _building_footprint(bounding_box: Any) -> List[carla.Location]:
        transform = carla.Transform(
            bounding_box.location,
            bounding_box.rotation,
        )
        extent = bounding_box.extent
        corners = []
        for x, y in (
            (extent.x, extent.y),
            (-extent.x, extent.y),
            (-extent.x, -extent.y),
            (extent.x, -extent.y),
        ):
            corner = transform.transform(
                carla.Location(x=float(x), y=float(y), z=-float(extent.z))
            )
            corners.append(
                carla.Location(
                    x=float(corner.x),
                    y=float(corner.y),
                    z=float(corner.z),
                )
            )
        return corners

    @staticmethod
    def _polygon_area(points: Sequence[carla.Location]) -> float:
        if len(points) < 3:
            return 0.0
        total = 0.0
        for current, following in zip(points, list(points[1:]) + [points[0]]):
            total += float(current.x) * float(following.y)
            total -= float(current.y) * float(following.x)
        return abs(total) * 0.5

    @staticmethod
    def _sample_polygon_edges(
        points: Sequence[carla.Location],
        spacing: float,
    ) -> List[carla.Location]:
        samples: List[carla.Location] = []
        if len(points) < 2:
            return samples
        for start, end in zip(points, list(points[1:]) + [points[0]]):
            length = math.hypot(float(end.x - start.x), float(end.y - start.y))
            steps = max(1, int(math.ceil(length / max(0.1, spacing))))
            for step in range(steps + 1):
                fraction = float(step) / float(steps)
                samples.append(
                    carla.Location(
                        x=float(start.x + (end.x - start.x) * fraction),
                        y=float(start.y + (end.y - start.y) * fraction),
                        z=float(start.z + (end.z - start.z) * fraction),
                    )
                )
        return samples

    def _build_building_footprints(
        self,
        road_locations: Sequence[carla.Location],
    ) -> List[List[carla.Location]]:
        """Build the same road-adjacent building layer used by the reference map."""
        try:
            environment_objects = self.world.get_environment_objects(
                carla.CityObjectLabel.Buildings
            )
        except Exception as exc:
            LOG.warning("Building footprints unavailable for map selector: %s", exc)
            return []

        cell_size = BUILDING_ROAD_PROXIMITY_M
        road_grid: Dict[Tuple[int, int], List[carla.Location]] = {}
        for location in road_locations:
            key = (
                math.floor(float(location.x) / cell_size),
                math.floor(float(location.y) / cell_size),
            )
            road_grid.setdefault(key, []).append(location)

        maximum_distance_squared = BUILDING_ROAD_PROXIMITY_M ** 2
        footprints: List[List[carla.Location]] = []
        for environment_object in environment_objects:
            try:
                bounding_box = environment_object.bounding_box
                footprint = self._building_footprint(bounding_box)
                area = self._polygon_area(footprint)
                height = float(bounding_box.extent.z) * 2.0
                if (
                    height < MIN_BUILDING_HEIGHT_M
                    or area < MIN_BUILDING_AREA_M2
                    or area * height < MIN_BUILDING_VOLUME_M3
                ):
                    continue

                close_to_road = not road_grid
                for sample in self._sample_polygon_edges(
                    footprint, BUILDING_EDGE_SAMPLE_M
                ):
                    cell_x = math.floor(float(sample.x) / cell_size)
                    cell_y = math.floor(float(sample.y) / cell_size)
                    for offset_x in (-1, 0, 1):
                        for offset_y in (-1, 0, 1):
                            for road_location in road_grid.get(
                                (cell_x + offset_x, cell_y + offset_y), []
                            ):
                                dx = float(sample.x - road_location.x)
                                dy = float(sample.y - road_location.y)
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
                    footprints.append(footprint)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
        return footprints

    def _configure_master_clock(self) -> None:
        """Become the single CARLA world/Traffic Manager clock owner."""
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
            self.world.apply_settings(settings)
            self._clock_configured = True
            traffic_manager = self.client.get_trafficmanager(
                self.traffic_manager_port
            )
            traffic_manager.set_synchronous_mode(True)
            self._next_world_tick_at = time.monotonic()
            LOG.info(
                "Master clock enabled: synchronous_mode=True, fixed_delta_seconds=%.3f",
                self.fixed_delta_seconds,
            )
        except Exception:
            self.release_master_clock()
            raise

    def advance_frame(self) -> None:
        """
        Drive one control period: ego control, world tick, then route gates.

        The route agent and the route gates must run whether or not this client
        owns the CARLA clock.  Gating them behind master-clock ownership left
        ``world.master_clock: false`` with a completely inert vehicle AUTO mode
        and a pedestrian that never received its second destination.
        """
        now = time.monotonic()
        owns_clock = bool(self.master_clock and self._clock_configured)
        if owns_clock:
            if now < self._next_world_tick_at:
                return
            self._update_camera_transforms()
            self._run_vehicle_route_agent()
            self.world.tick()
            # Pace from the pre-tick timestamp so a slow frame catches up
            # instead of permanently halving the simulation rate.
            self._next_world_tick_at = now + self.fixed_delta_seconds
        else:
            # Passive mode: another client owns the clock. Still apply ego
            # control at the configured period so the PID gains stay valid.
            if now < self._next_control_at:
                return
            self._update_camera_transforms()
            self._run_vehicle_route_agent()
            self._next_control_at = now + self.fixed_delta_seconds
        self._advance_vehicle_route()
        self._advance_pedestrian_route()
        self._monitor_vehicle_progress()

    def advance_master_clock(self) -> None:
        """Backward-compatible alias for advance_frame()."""
        self.advance_frame()

    def _update_camera_transforms(self) -> None:
        """Push camera poses once per control period rather than once per UI frame."""
        if self.camera is None:
            return
        try:
            self.camera.update_transform()
        except RuntimeError as exc:
            LOG.debug("Camera transform update skipped: %s", exc)

    def update_camera_keys(
        self,
        keys: Sequence[bool],
        delta_seconds: float,
    ) -> None:
        """
        Accumulate arrow-key camera deltas without an RPC per rendered frame.

        v1 pushed a set_transform() for every UI frame a key was held. With
        three live sensors that tripled the RPC rate for no visual gain, so the
        pose is applied once per control period from advance_frame() instead.
        """
        if self.camera is None:
            return
        yaw_axis = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
        pitch_axis = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
        if not (yaw_axis or pitch_axis):
            return
        delta_seconds = v1.clamp(delta_seconds, 0.0, 0.1)
        selected = self.camera.selected
        selected.yaw = v1.clamp(
            selected.yaw + yaw_axis * 75.0 * delta_seconds, -180.0, 180.0
        )
        selected.pitch = v1.clamp(
            selected.pitch + pitch_axis * 60.0 * delta_seconds, -90.0, 45.0
        )
        if selected.kind == "pole":
            invalidate = getattr(self.camera, "invalidate_pole_transform", None)
            if callable(invalidate):
                invalidate()

    def release_master_clock(self) -> None:
        """Restore clock settings after all owned actors have been destroyed."""
        if not self._clock_configured:
            return
        try:
            traffic_manager = self.client.get_trafficmanager(
                self.traffic_manager_port
            )
            # Only switch TM back to asynchronous operation when that matches
            # the world mode that existed before this UI took ownership.
            if not self._original_synchronous_mode:
                traffic_manager.set_synchronous_mode(False)
        except Exception:
            LOG.exception("Unable to restore Traffic Manager clock mode")
        if self.restore_world_settings:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = self._original_synchronous_mode
                settings.fixed_delta_seconds = self._original_fixed_delta_seconds
                self.world.apply_settings(settings)
                LOG.info(
                    "Restored world clock: synchronous_mode=%s, fixed_delta_seconds=%s",
                    self._original_synchronous_mode,
                    self._original_fixed_delta_seconds,
                )
            except Exception:
                LOG.exception("Unable to restore CARLA world clock settings")
        self._clock_configured = False

    @staticmethod
    def _four_wheel_blueprints(
        blueprints: Sequence[carla.ActorBlueprint],
    ) -> List[carla.ActorBlueprint]:
        four_wheel = []
        for blueprint in blueprints:
            if blueprint.has_attribute("number_of_wheels"):
                try:
                    if int(blueprint.get_attribute("number_of_wheels")) != 4:
                        continue
                except (TypeError, ValueError):
                    pass
            four_wheel.append(blueprint)
        return four_wheel

    def _select_occluder_blueprint(
        self,
        kind: str,
    ) -> carla.ActorBlueprint:
        """
        Resolve an installed heavy vehicle without making scenario setup fatal.

        CARLA 0.10 uses IDs such as ``vehicle.sprinter.mercedes`` and
        ``vehicle.carlacola.actors`` while older CARLA releases use the same
        words in a different order.  Both forms are supported.  A requested
        bus may be represented by the closest installed tall van/truck because
        this package does not necessarily ship a literal bus blueprint.
        """
        requested_kind = str(kind).lower()
        all_vehicles = sorted(
            self.world.get_blueprint_library().filter("vehicle.*"),
            key=lambda blueprint: blueprint.id,
        )
        vehicles = self._four_wheel_blueprints(all_vehicles) or all_vehicles
        if not vehicles:
            raise RuntimeError("the CARLA package has no vehicle blueprints")
        by_id = {str(blueprint.id): blueprint for blueprint in vehicles}

        preferences = {
            "bus": (
                # Literal buses, when available.
                "vehicle.mitsubishi.fusorosa",
                # Confirmed CARLA 0.10 tall-vehicle substitutes.
                "vehicle.sprinter.mercedes",
                "vehicle.ambulance.ford",
                "vehicle.carlacola.actors",
                "vehicle.firetruck.actors",
                # Older CARLA naming variants.
                "vehicle.mercedes.sprinter",
                "vehicle.ford.ambulance",
                "vehicle.carlamotors.carlacola",
                "vehicle.carlamotors.firetruck",
                "vehicle.volkswagen.t2",
                "vehicle.volkswagen.t2_2021",
            ),
            "truck": (
                # Confirmed CARLA 0.10 heavy vehicles.
                "vehicle.carlacola.actors",
                "vehicle.firetruck.actors",
                "vehicle.sprinter.mercedes",
                "vehicle.ambulance.ford",
                # Older CARLA naming variants.
                "vehicle.carlamotors.european_hgv",
                "vehicle.carlamotors.carlacola",
                "vehicle.carlamotors.firetruck",
                "vehicle.mercedes.sprinter",
                "vehicle.ford.ambulance",
                "vehicle.tesla.cybertruck",
            ),
        }
        token_order = {
            "bus": (
                "bus",
                "coach",
                "fusorosa",
                "sprinter",
                "van",
                "ambulance",
                "carlacola",
                "firetruck",
                "truck",
                "hgv",
            ),
            "truck": (
                "truck",
                "hgv",
                "carlacola",
                "firetruck",
                "sprinter",
                "van",
                "ambulance",
                "cybertruck",
                "bus",
                "fusorosa",
            ),
        }
        native_tokens = {
            "bus": ("bus", "coach", "fusorosa"),
            "truck": ("truck", "hgv", "carlacola", "firetruck", "cybertruck"),
        }
        if requested_kind not in preferences:
            raise ValueError("occluder kind must be bus or truck")

        selected = None
        selection_source = "preferred_id"
        for blueprint_id in preferences[requested_kind]:
            if blueprint_id in by_id:
                selected = by_id[blueprint_id]
                break
        if selected is None:
            selection_source = "type_token"
            for token in token_order[requested_kind]:
                matches = [
                    blueprint
                    for blueprint in vehicles
                    if token in str(blueprint.id).lower()
                    or token
                    in " ".join(getattr(blueprint, "tags", [])).lower()
                ]
                if matches:
                    selected = matches[0]
                    break
        if selected is None:
            # Last resort: keep the scenario usable with the first deterministic
            # four-wheel blueprint.  The status and structured log disclose it.
            selection_source = "four_wheel_fallback"
            selected = vehicles[0]

        selected_id = str(selected.id)
        substituted = not any(
            token in selected_id.lower()
            for token in native_tokens[requested_kind]
        )
        self.occluder_blueprint_resolution = {
            "requested_type": requested_kind,
            "blueprint_id": selected_id,
            "selection_source": selection_source,
            "substituted": substituted,
        }
        message = (
            "Occluder request '%s' resolved to '%s' via %s",
            requested_kind,
            selected_id,
            selection_source,
        )
        if substituted:
            LOG.warning(*message)
        else:
            LOG.info(*message)
        self._log_event(
            "occluder_blueprint_resolved",
            requested_type=requested_kind,
            blueprint_id=selected_id,
            selection_source=selection_source,
            substituted=substituted,
        )
        return selected

    @staticmethod
    def _route_indices(
        values: Any,
        point_count: int,
        field_name: str,
    ) -> List[int]:
        if values is None:
            return []
        if not isinstance(values, (list, tuple)):
            raise ValueError("{} must be a list of indices".format(field_name))
        if point_count <= 0 and values:
            raise ValueError("{} has no selectable map points".format(field_name))
        indices = []
        for value in values:
            try:
                index = int(value) % point_count
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise ValueError(
                    "{} contains an invalid index: {!r}".format(field_name, value)
                ) from exc
            if not indices or indices[-1] != index:
                indices.append(index)
        return indices

    @staticmethod
    def _dedupe_route_locations(
        locations: Sequence[carla.Location],
        minimum_distance_m: float = 0.5,
    ) -> List[carla.Location]:
        result: List[carla.Location] = []
        for location in locations:
            copied = v1.copy_location(location)
            if result and result[-1].distance(copied) < minimum_distance_m:
                continue
            result.append(copied)
        return result

    def _prepare_intermediate_routes(self, config: Dict[str, Any]) -> None:
        scenario = config["scenario"]
        vehicle_config = scenario["ego_vehicle"]
        self._configured_vehicle_waypoint_indices = self._route_indices(
            vehicle_config.get("route_waypoint_indices", []),
            len(self.road_preview),
            "scenario.ego_vehicle.route_waypoint_indices",
        )
        self._pending_vehicle_route_locations = [
            v1.copy_location(self.road_preview[index])
            for index in self._configured_vehicle_waypoint_indices
        ]
        self._pending_vehicle_route_segments.clear()
        self._pending_vehicle_route_targets.clear()
        self._pending_vehicle_agent_plan.clear()
        self._vehicle_scripted_route_requested = bool(
            vehicle_config.get("scripted_route", False)
        )
        self._vehicle_waypoint_reach_threshold_m = max(
            2.0,
            float(vehicle_config.get("waypoint_reach_threshold_m", 8.0)),
        )

        pedestrian_config = scenario["ego_pedestrian"]
        navigation = self.navigation_preview(int(scenario["seed"]))
        self._configured_pedestrian_waypoint_indices = self._route_indices(
            pedestrian_config.get("route_waypoint_indices", []),
            len(navigation),
            "scenario.ego_pedestrian.route_waypoint_indices",
        )
        start_index = int(pedestrian_config["start_spawn_index"]) % len(navigation)
        end_index = int(pedestrian_config["end_spawn_index"]) % len(navigation)
        if end_index == start_index:
            end_index = (start_index + 1) % len(navigation)
        route_locations = [navigation[start_index]]
        route_locations.extend(
            navigation[index]
            for index in self._configured_pedestrian_waypoint_indices
        )
        route_locations.append(navigation[end_index])
        normalized = self._dedupe_route_locations(route_locations)
        self._pending_pedestrian_route_locations = normalized[1:]
        self._pedestrian_waypoint_reach_threshold_m = max(
            0.25,
            float(pedestrian_config.get("waypoint_reach_threshold_m", 1.5)),
        )

    def _settle_actor_snapshot(self) -> None:
        """
        Advance one frame so a freshly spawned actor reports its real pose.

        Immediately after try_spawn_actor the client-side actor snapshot is
        still empty and ``actor.get_location()`` returns the world origin. The
        base class plans the ego route from that reading, so on Town10HD the
        whole route started 69 m away from the car: the agent drove
        cross-country toward its own route start, left the carriageway, and
        wedged against roadside furniture. Confirmed from the event log, where
        the pre-tick lane offset equalled the origin-to-lane distance exactly.
        """
        try:
            if self.master_clock and self._clock_configured:
                self.world.tick()
                self._next_world_tick_at = (
                    time.monotonic() + self.fixed_delta_seconds
                )
            else:
                self.world.wait_for_tick(2.0)
        except Exception as exc:
            LOG.warning(
                "Unable to settle the actor snapshot before planning: %s", exc
            )

    def _settled_ego_start(self, requested: carla.Location) -> carla.Location:
        """Re-read the ego pose after a settling tick before route planning."""
        if not v1.actor_alive(self.ego_vehicle):
            return requested
        self._settle_actor_snapshot()
        try:
            settled = self.ego_vehicle.get_location()
        except RuntimeError:
            return requested
        moved = float(settled.distance(requested))
        if moved > 1.0:
            LOG.warning(
                "Ego pose read %.1f m away before the first tick; planning the "
                "route from the settled pose (%.2f, %.2f, %.2f)",
                moved,
                float(settled.x),
                float(settled.y),
                float(settled.z),
            )
            self._log_event(
                "ego_vehicle_pose_settled",
                self.ego_vehicle,
                pre_tick={
                    "x": float(requested.x),
                    "y": float(requested.y),
                    "z": float(requested.z),
                },
                settled={
                    "x": float(settled.x),
                    "y": float(settled.y),
                    "z": float(settled.z),
                },
                correction_m=moved,
            )
        return v1.copy_location(settled)

    def _driving_waypoint(self, location: carla.Location) -> Optional[Any]:
        try:
            return self.map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            return None

    def _project_route_target(
        self,
        previous: Optional[carla.Location],
        target: carla.Location,
    ) -> carla.Location:
        """
        Snap a clicked map point onto the driving lane that matches travel direction.

        ``map.generate_waypoints`` samples both directions of a two-way road, so
        a via-point picked off the map easily lands on the oncoming centerline.
        GlobalRoutePlanner then honors that lane and plans a long detour or a
        U-turn, which reads as "the ego ignored my waypoint".  Choosing between
        the projected lane and its drivable neighbours by heading agreement
        keeps the ordered route pointing the way the user drew it.
        """
        waypoint = self._driving_waypoint(target)
        if waypoint is None:
            return v1.copy_location(target)
        if previous is None:
            return v1.copy_location(waypoint.transform.location)
        heading_x = float(target.x - previous.x)
        heading_y = float(target.y - previous.y)
        norm = math.hypot(heading_x, heading_y)
        if norm < 1.0:
            return v1.copy_location(waypoint.transform.location)
        heading_x /= norm
        heading_y /= norm
        candidates = [waypoint]
        for accessor in ("get_left_lane", "get_right_lane"):
            try:
                neighbour = getattr(waypoint, accessor)()
            except (AttributeError, RuntimeError):
                neighbour = None
            if neighbour is None:
                continue
            try:
                if neighbour.lane_type != carla.LaneType.Driving:
                    continue
            except (AttributeError, RuntimeError):
                continue
            candidates.append(neighbour)
        best = waypoint
        best_score: Optional[float] = None
        for candidate in candidates:
            try:
                forward = candidate.transform.get_forward_vector()
                offset = candidate.transform.location.distance(target)
            except (AttributeError, RuntimeError):
                continue
            alignment = heading_x * float(forward.x) + heading_y * float(forward.y)
            # Prefer heading agreement, but never trade away a much closer lane.
            score = alignment - 0.20 * float(offset)
            if best_score is None or score > best_score:
                best = candidate
                best_score = score
        projected = v1.copy_location(best.transform.location)
        moved = projected.distance(waypoint.transform.location)
        if moved > 1.0:
            LOG.info(
                "Route via-point (%.1f, %.1f) moved %.1f m to the lane heading "
                "toward the next point",
                float(target.x),
                float(target.y),
                float(moved),
            )
        return projected

    def _project_route_targets(
        self,
        targets: Sequence[carla.Location],
    ) -> List[carla.Location]:
        """Direction-align every target after the first, which is the ego pose."""
        if not targets:
            return []
        projected = [v1.copy_location(targets[0])]
        for target in targets[1:]:
            projected.append(self._project_route_target(projected[-1], target))
        return self._dedupe_route_locations(projected)

    def _record_detour(
        self,
        index: int,
        start: carla.Location,
        end: carla.Location,
        path_length_m: float,
    ) -> None:
        """Warn when a via-point forces a route far longer than the direct line."""
        direct = float(start.distance(end))
        if path_length_m <= max(40.0, 4.0 * direct):
            return
        message = "via {} adds a {:.0f} m detour (direct {:.0f} m)".format(
            index, path_length_m, direct
        )
        LOG.warning(
            "Ego route segment to (%.1f, %.1f): %s; check the lane direction "
            "of that via-point",
            float(end.x),
            float(end.y),
            message,
        )
        self._route_detour_warnings.append(message)

    @staticmethod
    def _path_length(path: Sequence[carla.Location]) -> float:
        total = 0.0
        for first, second in zip(path, path[1:]):
            total += float(first.distance(second))
        return total

    def plan_vehicle_route_for_export(
        self,
        start_transform: carla.Transform,
        intermediate_waypoints: Sequence[carla.Location],
        end_transform: carla.Transform,
        resolution: float,
    ) -> Dict[str, Any]:
        """Resolve and trace a selected route without touching live run state.

        Unlike :meth:`_build_route`, this method never advances the world,
        reads an actor pose, or writes any of the pending/active route fields.
        It therefore remains safe to call from the route Save button while a
        scenario is stopped or running.  A fresh planner is deliberately used
        instead of the controller's cached runtime planner.
        """
        sampling_resolution = float(resolution)
        if not math.isfinite(sampling_resolution) or sampling_resolution <= 0.0:
            raise ValueError("route sampling resolution must be greater than zero")
        planner_type = getattr(v1, "GlobalRoutePlanner", None)
        if planner_type is None:
            raise RuntimeError(
                "CARLA GlobalRoutePlanner is unavailable; cannot save a "
                "road-following ego route"
            )

        authoritative_start = v1.copy_transform(start_transform)
        requested_controls = self._dedupe_route_locations(
            [authoritative_start.location]
            + [v1.copy_location(location) for location in intermediate_waypoints]
            + [v1.copy_location(end_transform.location)]
        )
        if len(requested_controls) < 2:
            raise ValueError("ego route start and end must be distinct")
        resolved_controls = self._project_route_targets(requested_controls)
        if len(resolved_controls) < 2:
            raise ValueError("ego route has fewer than two distinct road points")

        try:
            planner = planner_type(self.map, sampling_resolution)
        except Exception as exc:
            raise RuntimeError(
                "unable to initialize CARLA GlobalRoutePlanner: {}".format(exc)
            ) from exc

        planned_path: List[carla.Location] = []
        for segment_number, (segment_start, segment_end) in enumerate(
            zip(resolved_controls, resolved_controls[1:]), start=1
        ):
            try:
                trace = list(planner.trace_route(segment_start, segment_end))
            except Exception as exc:
                raise RuntimeError(
                    "unable to plan route segment {} ending at "
                    "({:.2f}, {:.2f}): {}".format(
                        segment_number,
                        float(segment_end.x),
                        float(segment_end.y),
                        exc,
                    )
                ) from exc
            if not trace:
                raise RuntimeError(
                    "selected ego route contains an unreachable segment {} "
                    "ending at ({:.2f}, {:.2f})".format(
                        segment_number,
                        float(segment_end.x),
                        float(segment_end.y),
                    )
                )
            segment = [
                v1.copy_location(waypoint.transform.location)
                for waypoint, _road_option in trace
            ]
            if not segment or segment[0].distance(segment_start) > 1.0:
                segment.insert(0, v1.copy_location(segment_start))
            if segment[-1].distance(segment_end) > 1.0:
                segment.append(v1.copy_location(segment_end))
            if (
                planned_path
                and segment
                and planned_path[-1].distance(segment[0]) < 0.5
            ):
                segment = segment[1:]
            planned_path.extend(segment)
        planned_path = self._dedupe_route_locations(planned_path, 0.05)
        if len(planned_path) < 2:
            raise RuntimeError("CARLA route planner returned an empty route")

        resolved_end_location = v1.copy_location(resolved_controls[-1])
        resolved_end_rotation = carla.Rotation(
            pitch=end_transform.rotation.pitch,
            yaw=end_transform.rotation.yaw,
            roll=end_transform.rotation.roll,
        )
        terminal_waypoint = self._driving_waypoint(resolved_end_location)
        if terminal_waypoint is not None:
            try:
                rotation = terminal_waypoint.transform.rotation
                resolved_end_rotation = carla.Rotation(
                    pitch=rotation.pitch,
                    yaw=rotation.yaw,
                    roll=rotation.roll,
                )
            except (AttributeError, RuntimeError):
                pass

        return {
            "start_transform": authoritative_start,
            "intermediate_waypoints": [
                v1.copy_location(location) for location in resolved_controls[1:-1]
            ],
            "end_transform": carla.Transform(
                resolved_end_location, resolved_end_rotation
            ),
            "planned_path": [
                v1.copy_location(location) for location in planned_path
            ],
        }

    def _build_route(
        self,
        start: carla.Location,
        end: carla.Location,
        resolution: float,
    ) -> List[carla.Location]:
        """Build one continuous vehicle route through every selected waypoint."""
        self._route_detour_warnings = []
        # The base class passes a pose read before the first tick after spawn.
        start = self._settled_ego_start(start)
        targets = self._project_route_targets(
            self._dedupe_route_locations(
                [start] + self._pending_vehicle_route_locations + [end]
            )
        )
        if len(targets) < 2:
            return [v1.copy_location(start), v1.copy_location(end)]
        route: List[carla.Location] = []
        segments: List[List[carla.Location]] = []
        agent_plan: List[Tuple[Any, Any]] = []
        planner = self._get_route_planner(resolution)
        if self._vehicle_scripted_route_requested and planner is None:
            raise RuntimeError(
                "CARLA GlobalRoutePlanner is required for scripted ego routing"
            )
        for segment_index, (segment_start, segment_end) in enumerate(
            zip(targets, targets[1:]), start=1
        ):
            segment_trace: List[Tuple[Any, Any]] = []
            if planner is not None:
                try:
                    segment_trace = list(
                        planner.trace_route(segment_start, segment_end)
                    )
                except Exception as exc:
                    if self._vehicle_scripted_route_requested:
                        raise RuntimeError(
                            "unable to plan an ego route segment: {}".format(exc)
                        ) from exc
                    LOG.warning("Route planner segment failed: %s", exc)
            if self._vehicle_scripted_route_requested and not segment_trace:
                raise RuntimeError(
                    "selected ego route contains an unreachable segment near "
                    "({:.1f}, {:.1f})".format(segment_end.x, segment_end.y)
                )
            if segment_trace:
                target_waypoint = self.map.get_waypoint(
                    segment_end,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if (
                    target_waypoint is not None
                    and segment_trace[-1][0].transform.location.distance(
                        target_waypoint.transform.location
                    )
                    > 0.5
                ):
                    segment_trace.append(
                        (target_waypoint, segment_trace[-1][1])
                    )
                if (
                    agent_plan
                    and agent_plan[-1][0].transform.location.distance(
                        segment_trace[0][0].transform.location
                    )
                    < 0.5
                ):
                    segment_trace = segment_trace[1:]
                agent_plan.extend(segment_trace)
                segment = [
                    v1.copy_location(waypoint.transform.location)
                    for waypoint, _ in segment_trace
                ]
            else:
                segment = super()._build_route(
                    segment_start,
                    segment_end,
                    resolution,
                )
            segment = self._dedupe_route_locations(segment)
            if not segment or segment[0].distance(segment_start) > 1.0:
                segment.insert(0, v1.copy_location(segment_start))
            if segment[-1].distance(segment_end) > 1.0:
                segment.append(v1.copy_location(segment_end))
            self._record_detour(
                segment_index,
                segment_start,
                segment_end,
                self._path_length(segment),
            )
            segments.append(segment)
            if route and segment and route[-1].distance(segment[0]) < 1.0:
                segment = segment[1:]
            route.extend(segment)
        self._pending_vehicle_route_segments = segments
        self._pending_vehicle_route_targets = [
            v1.copy_location(location) for location in targets[1:]
        ]
        self._pending_vehicle_agent_plan = agent_plan
        return route or [v1.copy_location(start), v1.copy_location(end)]

    def _activate_scripted_ego_vehicle(
        self,
        route_path: Sequence[carla.Location],
        traffic_manager_port: int,
    ) -> None:
        """Reserve v2 ego authority for BasicAgent; TM remains NPC-only."""
        del route_path
        self.ego_vehicle.set_autopilot(False, int(traffic_manager_port))

    def _trace_vehicle_agent_plan(
        self,
        start: carla.Location,
        targets: Sequence[carla.Location],
        resolution: float,
    ) -> List[Tuple[Any, Any]]:
        """Build one BasicAgent plan from the current pose through all targets."""
        planner = self._get_route_planner(resolution)
        if planner is None:
            raise RuntimeError(
                "CARLA GlobalRoutePlanner is required for autonomous vehicle control"
            )
        plan: List[Tuple[Any, Any]] = []
        segment_start = v1.copy_location(start)
        aligned = self._project_route_targets(
            [segment_start] + list(self._dedupe_route_locations(targets))
        )[1:]
        for target in aligned:
            try:
                segment = list(planner.trace_route(segment_start, target))
            except Exception as exc:
                raise RuntimeError(
                    "unable to resume the ego route near ({:.1f}, {:.1f}): {}".format(
                        target.x,
                        target.y,
                        exc,
                    )
                ) from exc
            if not segment:
                raise RuntimeError(
                    "ego route target near ({:.1f}, {:.1f}) is unreachable".format(
                        target.x,
                        target.y,
                    )
                )
            target_waypoint = self.map.get_waypoint(
                target,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if (
                target_waypoint is not None
                and segment[-1][0].transform.location.distance(
                    target_waypoint.transform.location
                )
                > 0.5
            ):
                segment.append((target_waypoint, segment[-1][1]))
            if (
                plan
                and plan[-1][0].transform.location.distance(
                    segment[0][0].transform.location
                )
                < 0.5
            ):
                segment = segment[1:]
            plan.extend(segment)
            segment_start = v1.copy_location(target)
        if len(plan) < 2:
            raise RuntimeError("autonomous vehicle route produced no usable plan")
        return plan

    def _create_camera_director(
        self,
        config: Dict[str, Any],
        pole_views: Sequence[Any],
    ) -> v1.CameraDirector:
        """Install the dual-ego-stream camera layout instead of v1's single stream."""
        return MultiStreamCameraDirector(
            self.world,
            config,
            self.ego_vehicle,
            self.ego_pedestrian,
            pole_views,
        )

    def _occluder_route_waypoint(
        self,
        config: Dict[str, Any],
    ) -> Optional[Tuple[Any, int]]:
        """
        The driving lane and route index for the occluder, avoiding junctions.

        A road cross-section is only well defined outside an intersection: in a
        junction a lateral sweep crosses into the perpendicular road's lanes and
        reports an absurd kerb distance. Parking the occluder just short of the
        intersection is also the realistic placement. The index is returned so
        the caller can pin the base class to the same route sample the lateral
        offset was computed for.
        """
        if not self._vehicle_route:
            return None
        fraction = v1.clamp(
            float(config["scenario"]["occluder"]["route_fraction"]), 0.0, 1.0
        )
        last = len(self._vehicle_route) - 1
        requested = min(last, int(round(fraction * last)))
        fallback: Optional[Tuple[Any, int]] = None
        for offset in range(0, last + 1):
            for index in sorted({requested - offset, requested + offset}):
                if not 0 <= index <= last:
                    continue
                waypoint = self._driving_waypoint(self._vehicle_route[index])
                if waypoint is None:
                    continue
                if fallback is None:
                    fallback = (waypoint, index)
                try:
                    if not bool(waypoint.is_junction):
                        if offset:
                            LOG.info(
                                "Occluder route fraction landed in a junction; "
                                "using route sample %d instead of %d",
                                index,
                                requested,
                            )
                        return waypoint, index
                except (AttributeError, RuntimeError):
                    continue
        return fallback

    def _kerbside_lateral_offset(
        self,
        waypoint: Optional[Any],
        half_width: float,
        clearance: float,
    ) -> Optional[Tuple[float, str]]:
        """
        Find a kerbside offset that occludes without taking a lane out of service.

        Clearing only the *ego* lane pushed the occluder into the adjacent
        driving lane, where Traffic Manager NPCs queued behind the parked,
        physics-disabled vehicle indefinitely and gridlocked the junction the
        ego had to cross (observed with every vehicle within 25 m at 0 m/s).
        Walking outward to the shoulder or a parking lane keeps the sight-line
        break while leaving all driving lanes usable.
        """
        if waypoint is None:
            return None
        try:
            base_location = waypoint.transform.location
            base_right = waypoint.transform.get_right_vector()
            base_lane_width = float(waypoint.lane_width)
            base_road_id = int(waypoint.road_id)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        maximum_offset = 10.0

        # A lateral projection sweep is used rather than
        # get_right_lane()/get_left_lane(): those return None on the junction
        # waypoints that route fractions frequently land on, which silently
        # defeated kerb detection at exactly the interesting locations.
        parkable = (carla.LaneType.Shoulder, carla.LaneType.Parking)
        step = 0.25
        for sign in (1.0, -1.0):
            driving_edge = 0.5 * base_lane_width
            kerb_type: Optional[str] = None
            distance = step
            while distance <= 14.0:
                probe = carla.Location(
                    x=float(base_location.x + base_right.x * sign * distance),
                    y=float(base_location.y + base_right.y * sign * distance),
                    z=float(base_location.z),
                )
                try:
                    candidate = self.map.get_waypoint(
                        probe,
                        project_to_road=True,
                        lane_type=carla.LaneType.Any,
                    )
                except RuntimeError:
                    candidate = None
                distance += step
                if candidate is None:
                    continue
                try:
                    # Only trust the sample when the probe really lies in that
                    # lane rather than being projected in from far away, and
                    # only within this road's own cross-section: a sweep that
                    # wanders onto a crossing road reports a kerb 17 m out.
                    if probe.distance(candidate.transform.location) > 1.5:
                        continue
                    if int(candidate.road_id) != base_road_id:
                        continue
                    lane_type = candidate.lane_type
                    lane_width = float(candidate.lane_width)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    continue
                if lane_type == carla.LaneType.Driving:
                    # Stay clear of every driving lane found on this side, not
                    # just the ego's own: a parked, physics-disabled occluder in
                    # any traffic lane gridlocks Traffic Manager behind it.
                    driving_edge = max(driving_edge, distance + 0.5 * lane_width)
                elif lane_type in parkable and kerb_type is None:
                    kerb_type = str(lane_type).split(".")[-1].lower()
            if kerb_type is None:
                continue
            resolved = driving_edge + half_width + clearance
            if resolved > maximum_offset:
                LOG.warning(
                    "Kerbside occluder offset %.2f m exceeds the %.1f m limit; "
                    "falling back to a lane-edge offset",
                    resolved,
                    maximum_offset,
                )
                continue
            return sign * resolved, "kerb_{}".format(kerb_type)
        return None

    def _spawn_occluder(
        self,
        config: Dict[str, Any],
        rng: Any,
    ) -> Optional[carla.Vehicle]:
        """
        Park the occluder at the kerb unless it is asked to block the lane.

        ``lateral_offset_m: 0.0`` placed the bus exactly on the ego route
        centerline, so the ego agent held an emergency stop behind it for the
        rest of the run.  A kerbside occluder still breaks line of sight to the
        pedestrian, which is what the demo needs, while leaving every driving
        lane usable for the ego and for NPC traffic.  Set
        ``occluder.blocks_ego_lane: true`` to keep the literal offset and
        deliberately block the road.
        """
        occluder_config = config["scenario"]["occluder"]
        if str(occluder_config["type"]).lower() == "none":
            return None
        self.occluder_lateral_offset_resolved = None
        requested = float(occluder_config["lateral_offset_m"])
        if bool(occluder_config.get("blocks_ego_lane", False)):
            return super()._spawn_occluder(config, rng)
        half_width = max(0.5, float(occluder_config.get("half_width_m", 1.30)))
        clearance = max(
            0.0, float(occluder_config.get("min_ego_lane_clearance_m", 0.40))
        )
        selection = self._occluder_route_waypoint(config)
        waypoint = None if selection is None else selection[0]
        route_index = None if selection is None else selection[1]
        lane_width = 3.5
        if waypoint is not None:
            try:
                lane_width = float(waypoint.lane_width)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                lane_width = 3.5
        placement = self._kerbside_lateral_offset(waypoint, half_width, clearance)
        if placement is None:
            resolved = math.copysign(
                0.5 * lane_width + half_width + clearance,
                requested if requested else 1.0,
            )
            source = "ego_lane_edge_only"
            LOG.warning(
                "No shoulder or parking lane found near the occluder point; "
                "offsetting to %.2f m clears the ego lane but NPC traffic in "
                "the adjacent lane may queue behind it",
                resolved,
            )
        else:
            resolved, source = placement
        if abs(requested) >= abs(resolved) and (
            requested == 0.0 or math.copysign(1.0, requested) == math.copysign(1.0, resolved)
        ):
            # An explicit offset already at least as far out is respected.
            return super()._spawn_occluder(config, rng)
        LOG.warning(
            "Occluder lateral_offset_m=%.2f would obstruct a %.2f m driving "
            "lane; using %.2f m (%s) so the ego route and NPC traffic stay clear",
            requested,
            lane_width,
            resolved,
            source,
        )
        patched = copy.deepcopy(config)
        patched["scenario"]["occluder"]["lateral_offset_m"] = resolved
        patched_fraction = float(config["scenario"]["occluder"]["route_fraction"])
        if route_index is not None and len(self._vehicle_route) > 1:
            # The base class recomputes its own route index from the fraction.
            # Pin it to the sample the lateral offset was measured at, or the
            # offset is applied at a different waypoint with a different
            # orientation and the occluder lands back in a traffic lane.
            patched_fraction = route_index / float(len(self._vehicle_route) - 1)
            patched["scenario"]["occluder"]["route_fraction"] = patched_fraction
        occluder = super()._spawn_occluder(patched, rng)
        self.occluder_lateral_offset_resolved = {
            "requested_lateral_offset_m": requested,
            "resolved_lateral_offset_m": resolved,
            "ego_lane_width_m": lane_width,
            "placement": source,
            "route_index": route_index,
            "route_fraction_used": patched_fraction,
        }
        self._log_event(
            "occluder_offset_resolved",
            occluder,
            **self.occluder_lateral_offset_resolved,
        )
        return occluder

    def _create_vehicle_route_agent(
        self,
        config: Dict[str, Any],
        plan: Sequence[Tuple[Any, Any]],
    ) -> Any:
        """Create a conservative 20 Hz agent that holds the lane centerline."""
        if BasicAgent is None:
            raise RuntimeError("CARLA BasicAgent is unavailable")
        vehicle_config = config["scenario"]["ego_vehicle"]
        resolution = max(
            0.5,
            float(vehicle_config.get("route_sampling_resolution_m", 2.0)),
        )
        self._vehicle_cruise_speed_kmh = max(
            1.0,
            float(vehicle_config.get("scripted_speed_kmh", 18.0)),
        )
        self._vehicle_curve_speed_kmh = v1.clamp(
            float(vehicle_config.get("scripted_curve_speed_kmh", 10.0)),
            1.0,
            self._vehicle_cruise_speed_kmh,
        )
        # Measured steady-state PID tracking error is 0.1-0.9 m of the lane
        # centre, so a 0.80 fraction of the 0.73 m usable half-lane fired on
        # every frame of normal driving. The guard is for real excursions
        # (2-3 m, observed when the ego left the carriageway), so the fraction
        # may exceed 1.0.
        self._vehicle_lane_guard_fraction = v1.clamp(
            float(vehicle_config.get("lane_center_guard_fraction", 1.60)),
            0.25,
            4.0,
        )
        self._vehicle_lane_guard_brake_speed_mps = max(
            0.5,
            float(vehicle_config.get("lane_guard_brake_speed_mps", 2.0)),
        )
        self._vehicle_lane_guard_recovery_throttle = v1.clamp(
            float(vehicle_config.get("lane_guard_recovery_throttle", 0.30)),
            0.0,
            0.60,
        )
        self._vehicle_stall_warning_seconds = max(
            1.0,
            float(vehicle_config.get("stall_warning_seconds", 6.0)),
        )
        self._vehicle_stuck_recovery_attempts = max(
            0, int(vehicle_config.get("stuck_recovery_attempts", 3))
        )
        self._vehicle_stuck_recovery_used = 0
        self._vehicle_reverse_until = 0.0
        dt = float(self.fixed_delta_seconds)
        # Look-ahead is the single most important stability parameter here. An
        # earlier revision cut the stock 3.0 + 0.5*speed purge distance down to
        # 1.0 + 0.10*speed to stop the agent clipping the inside of bends. That
        # left the lateral PID chasing a point ~1.5 m ahead at 5 m/s: any small
        # heading error produced a large steering command, the ego snaked, and
        # it eventually left the carriageway altogether (observed driving into a
        # traffic-light pole with full throttle applied). CARLA's own defaults
        # are restored, and every term is configurable.
        purge_base = max(
            0.5, float(vehicle_config.get("waypoint_purge_base_m", 3.0))
        )
        purge_ratio = max(
            0.0, float(vehicle_config.get("waypoint_purge_speed_ratio", 0.5))
        )
        self._vehicle_waypoint_purge_base_m = purge_base
        self._vehicle_waypoint_purge_speed_ratio = purge_ratio
        local_planner_options = {
            "sampling_resolution": resolution,
            "sampling_radius": min(2.0, resolution),
            "dt": dt,
            "base_min_distance": purge_base,
            "distance_ratio": purge_ratio,
            "max_throttle": v1.clamp(
                float(vehicle_config.get("max_throttle", 0.60)), 0.1, 1.0
            ),
            "max_brake": 0.75,
            "max_steering": v1.clamp(
                float(vehicle_config.get("max_steering", 0.70)), 0.1, 1.0
            ),
            "lateral_control_dict": {
                "K_P": float(vehicle_config.get("lateral_k_p", 1.30)),
                "K_I": float(vehicle_config.get("lateral_k_i", 0.03)),
                "K_D": float(vehicle_config.get("lateral_k_d", 0.20)),
                "dt": dt,
            },
            "longitudinal_control_dict": {
                "K_P": 1.00,
                "K_I": 0.02,
                "K_D": 0.05,
                "dt": dt,
            },
            "base_vehicle_threshold": 6.0,
            "detection_speed_ratio": 1.0,
            "use_bbs_detection": True,
        }
        ignored_actor_ids: List[int] = []
        occluder_config = config["scenario"]["occluder"]
        if not bool(occluder_config.get("blocks_ego_lane", False)) and v1.actor_alive(
            self.occluder
        ):
            # The occluder is a parked prop that exists to break line of sight.
            # Left in the obstacle list it holds a permanent emergency stop and
            # the ego never reaches its destination.
            ignored_actor_ids.append(int(self.occluder.id))
        agent = EgoRouteAgent(
            self.ego_vehicle,
            target_speed=self._vehicle_cruise_speed_kmh,
            opt_dict=local_planner_options,
            map_inst=self.map,
            grp_inst=self._get_route_planner(resolution),
            ignored_actor_ids=ignored_actor_ids,
            ignore_lights=bool(
                vehicle_config.get("ignore_traffic_lights", False)
            ),
            ignore_signs=bool(vehicle_config.get("ignore_stop_signs", False)),
        )
        agent.set_global_plan(
            list(plan),
            stop_waypoint_creation=True,
            clean_queue=True,
        )
        self._vehicle_stalled_since = None
        self._vehicle_lane_guard_active = False
        return agent

    def _start_vehicle_agent_route(self, config: Dict[str, Any]) -> None:
        """Give the Python navigation agent one authoritative ordered plan."""
        self._vehicle_route_segment_queue.clear()
        self._vehicle_route_target = None
        self._vehicle_route_target_number = 0
        self._vehicle_route_total_targets = 0
        self._vehicle_route_agent = None
        if not self._vehicle_scripted_route_requested:
            return
        if (
            len(self._pending_vehicle_route_segments)
            != len(self._pending_vehicle_route_targets)
        ):
            raise RuntimeError("vehicle route segment/target count mismatch")
        gated_segments = list(
            zip(
                self._pending_vehicle_route_segments,
                self._pending_vehicle_route_targets,
            )
        )
        if not gated_segments:
            raise RuntimeError("scripted vehicle route produced no route segments")
        if len(self._pending_vehicle_agent_plan) < 2:
            raise RuntimeError("scripted vehicle route produced no agent plan")
        self.ego_vehicle.set_autopilot(
            False,
            int(config["traffic_manager"]["port"]),
        )
        self._vehicle_route_agent = self._create_vehicle_route_agent(
            config,
            self._pending_vehicle_agent_plan,
        )
        _, first_target = gated_segments[0]
        self._vehicle_route_segment_queue = [
            (
                [v1.copy_location(location) for location in segment],
                v1.copy_location(target),
            )
            for segment, target in gated_segments[1:]
        ]
        self._vehicle_route_target = v1.copy_location(first_target)
        self._vehicle_route_target_number = 1
        self._vehicle_route_total_targets = len(gated_segments)
        self._log_event(
            "ego_vehicle_route_agent_started",
            self.ego_vehicle,
            target_number=1,
            route_targets=self._vehicle_route_total_targets,
            route_points=len(self._pending_vehicle_agent_plan),
            target_speed_kmh=self._vehicle_cruise_speed_kmh,
            curve_speed_kmh=self._vehicle_curve_speed_kmh,
            lane_guard_fraction=self._vehicle_lane_guard_fraction,
            waypoint_purge_base_m=self._vehicle_waypoint_purge_base_m,
            waypoint_purge_speed_ratio=self._vehicle_waypoint_purge_speed_ratio,
            waypoint_kind=(
                "via" if self._configured_vehicle_waypoint_indices else "destination"
            ),
            target={
                "x": float(first_target.x),
                "y": float(first_target.y),
                "z": float(first_target.z),
            },
        )

    @staticmethod
    def _normalized_yaw_delta(first: float, second: float) -> float:
        return (float(second) - float(first) + 180.0) % 360.0 - 180.0

    def _vehicle_agent_target_speed(self) -> float:
        """Slow down before a material heading change in the queued route."""
        try:
            local_planner = self._vehicle_route_agent.get_local_planner()
            plan = list(local_planner.get_plan())[:12]
            vehicle_yaw = float(self.ego_vehicle.get_transform().rotation.yaw)
        except (AttributeError, RuntimeError, TypeError):
            return self._vehicle_cruise_speed_kmh
        headings = [vehicle_yaw]
        for waypoint, _ in plan:
            try:
                headings.append(float(waypoint.transform.rotation.yaw))
            except (AttributeError, TypeError, RuntimeError):
                continue
        if len(headings) < 3:
            # A short remaining plan is the end of the route, not a curve.
            # Returning the curve speed here made the ego crawl the last
            # stretch to every destination.
            return self._vehicle_cruise_speed_kmh
        accumulated_turn = sum(
            abs(self._normalized_yaw_delta(first, second))
            for first, second in zip(headings, headings[1:])
        )
        heading_change = max(
            abs(self._normalized_yaw_delta(headings[0], heading))
            for heading in headings[1:]
        )
        if accumulated_turn >= 35.0 or heading_change >= 25.0:
            return self._vehicle_curve_speed_kmh
        return self._vehicle_cruise_speed_kmh

    def _lane_reference(self) -> Optional[Tuple[Any, float]]:
        """Return the ego's driving lane and its signed lateral offset in meters."""
        try:
            transform = self.ego_vehicle.get_transform()
        except (AttributeError, RuntimeError):
            return None
        waypoint = self._driving_waypoint(transform.location)
        if waypoint is None:
            return None
        try:
            center = waypoint.transform.location
            right = waypoint.transform.get_right_vector()
        except (AttributeError, RuntimeError):
            return None
        dx = float(transform.location.x - center.x)
        dy = float(transform.location.y - center.y)
        # Project onto the lane's right vector: the raw 2D distance to the
        # projected waypoint also picks up longitudinal sampling error.
        return waypoint, dx * float(right.x) + dy * float(right.y)

    def _apply_vehicle_lane_guard(self, control: Any) -> Any:
        """
        Damp throttle when the ego drifts toward the lane edge.

        The guard deliberately never latches a brake on a stopped vehicle. The
        previous version braked whenever the center was more than ~0.79 m off
        the lane centerline, including while stationary; with throttle forced to
        zero the lateral PID had no authority to steer back, so a single normal
        junction exit deadlocked the route permanently (confirmed in
        events.jsonl as ego_vehicle_lane_guard repeating at a constant
        lateral_error_m). Junction crossings legitimately leave the centerline
        and are exempt, an agent-level hazard stop always wins, and a slow
        off-center ego is given enough throttle to recover.
        """
        reference = self._lane_reference()
        if reference is None:
            # Off every driving lane: the global plan is the only way back, so
            # do not fight the agent's own recovery steering.
            self._clear_lane_guard_status()
            return control
        waypoint, lateral = reference
        try:
            in_junction = bool(waypoint.is_junction)
            lane_width = float(waypoint.lane_width)
            half_vehicle_width = float(self.ego_vehicle.bounding_box.extent.y)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._clear_lane_guard_status()
            return control
        hazard = getattr(self._vehicle_route_agent, "last_hazard", None)
        if hazard is not None:
            # A red light or a real vehicle ahead already stopped the agent.
            self._clear_lane_guard_status()
            return control
        usable_half_lane = max(
            0.35, 0.5 * lane_width - half_vehicle_width - 0.10
        )
        lateral_error = abs(lateral)
        caution_limit = max(
            0.30, usable_half_lane * self._vehicle_lane_guard_fraction
        )
        emergency_limit = max(caution_limit + 0.25, usable_half_lane + 0.35)
        speed = actor_speed_mps(self.ego_vehicle)
        intervention: Optional[str] = None
        # A junction crossing legitimately leaves the lane centerline, but the
        # exemption must not be unbounded: an ego that has left the carriageway
        # near a junction reported no guard action at all and drove on.
        junction_limit = emergency_limit * 2.5
        if lateral_error < caution_limit or (
            in_junction and lateral_error < junction_limit
        ):
            self._clear_lane_guard_status()
            return control
        if (
            lateral_error >= emergency_limit
            and speed >= self._vehicle_lane_guard_brake_speed_mps
        ):
            # Moving quickly off the lane: scrub speed but keep the steer value
            # so the controller can still turn back toward the centerline.
            control.throttle = 0.0
            control.brake = max(float(control.brake), 0.45)
            intervention = "brake"
        else:
            recovery = (
                self._vehicle_lane_guard_recovery_throttle
                if speed < self._vehicle_lane_guard_brake_speed_mps
                else 0.0
            )
            control.throttle = float(
                v1.clamp(float(control.throttle), recovery, 0.35)
            )
            control.brake = 0.0
            intervention = "throttle_limit"
        if abs(float(control.steer)) >= 0.60:
            control.throttle = min(float(control.throttle), 0.30)
        self._vehicle_lane_guard_active = True
        if time.monotonic() >= self._next_vehicle_safety_log_at:
            self._vehicle_lane_guard_status = (
                "Vehicle AUTO lane guard: {:.2f} m off lane center "
                "({})".format(lateral_error, intervention)
            )
            self.status = self._vehicle_lane_guard_status
            self._log_event(
                "ego_vehicle_lane_guard",
                self.ego_vehicle,
                intervention=intervention,
                lateral_error_m=lateral_error,
                lateral_offset_m=float(lateral),
                caution_limit_m=caution_limit,
                emergency_limit_m=emergency_limit,
                speed_mps=float(speed),
                in_junction=in_junction,
            )
            self._next_vehicle_safety_log_at = time.monotonic() + 1.0
        return control

    def _clear_lane_guard_status(self) -> None:
        """Drop the guard message once the ego is back inside its lane."""
        if not self._vehicle_lane_guard_active:
            return
        self._vehicle_lane_guard_active = False
        if self.status == self._vehicle_lane_guard_status:
            self.status = "Vehicle AUTO recovered lane center"
        self._vehicle_lane_guard_status = ""

    def _monitor_vehicle_progress(self) -> None:
        """Report why an autonomous ego stopped instead of leaving it silent."""
        agent = self._vehicle_route_agent
        if not self.running or agent is None or not v1.actor_alive(self.ego_vehicle):
            self._vehicle_stalled_since = None
            return
        try:
            plan_done = bool(agent.done())
        except (AttributeError, RuntimeError):
            plan_done = False
        if plan_done or actor_speed_mps(self.ego_vehicle) >= 0.3:
            self._vehicle_stalled_since = None
            return
        now = time.monotonic()
        if self._vehicle_stalled_since is None:
            self._vehicle_stalled_since = now
            return
        stalled_for = now - self._vehicle_stalled_since
        if (
            stalled_for < self._vehicle_stall_warning_seconds
            or now < self._next_vehicle_stall_log_at
        ):
            return
        hazard = getattr(agent, "last_hazard", None)
        reasons = {
            "vehicle": "blocked by a vehicle ahead",
            "traffic_light": "holding at a red traffic light",
        }
        reason = reasons.get(
            hazard,
            "lane guard active" if self._vehicle_lane_guard_active else
            "not moving under throttle; obstructed or off route",
        )
        self.status = "Vehicle AUTO stopped {:.0f}s: {}".format(
            stalled_for, reason
        )
        self._log_event(
            "ego_vehicle_route_stalled",
            self.ego_vehicle,
            stalled_seconds=float(stalled_for),
            hazard=hazard,
            lane_guard_active=self._vehicle_lane_guard_active,
            target_number=self._vehicle_route_target_number,
            route_targets=self._vehicle_route_total_targets,
            recovery_attempts_used=self._vehicle_stuck_recovery_used,
        )
        # Waiting behind traffic or at a red light is normal driving, so those
        # repeats are logged sparsely; an unexplained stall stays chatty.
        self._next_vehicle_stall_log_at = now + (15.0 if hazard else 5.0)
        if hazard is None:
            self._recover_stuck_vehicle(now)

    def _recover_stuck_vehicle(self, now: float) -> None:
        """
        Back off and re-plan when the ego is wedged rather than waiting.

        A demo ego that drives into a kerb or a pole sits there under full
        throttle forever, which makes the scenario unusable. A short straight
        reverse pulse frees the vehicle and a fresh plan from the recovered pose
        puts it back on the ordered route. Attempts are bounded and logged so
        the behavior is never silent.
        """
        if self._vehicle_stuck_recovery_used >= self._vehicle_stuck_recovery_attempts:
            return
        self._vehicle_stuck_recovery_used += 1
        self._vehicle_reverse_until = now + 1.2
        self._vehicle_stalled_since = None
        replanned = 0
        error: Optional[str] = None
        targets = self._remaining_vehicle_route_targets()
        if targets and self.last_config is not None:
            resolution = max(
                0.5,
                float(
                    self.last_config["scenario"]["ego_vehicle"].get(
                        "route_sampling_resolution_m", 2.0
                    )
                ),
            )
            try:
                plan = self._trace_vehicle_agent_plan(
                    self.ego_vehicle.get_location(), targets, resolution
                )
                self._vehicle_route_agent.set_global_plan(
                    list(plan),
                    stop_waypoint_creation=True,
                    clean_queue=True,
                )
                replanned = len(plan)
            except Exception as exc:
                error = str(exc)
                LOG.warning("Stuck-recovery re-plan failed: %s", exc)
        self.status = (
            "Vehicle AUTO stuck: reversing and re-planning "
            "(recovery {}/{})".format(
                self._vehicle_stuck_recovery_used,
                self._vehicle_stuck_recovery_attempts,
            )
        )
        self._log_event(
            "ego_vehicle_stuck_recovery",
            self.ego_vehicle,
            attempt=self._vehicle_stuck_recovery_used,
            max_attempts=self._vehicle_stuck_recovery_attempts,
            reverse_seconds=1.2,
            replanned_points=replanned,
            remaining_targets=len(targets),
            error=error,
        )

    def _run_vehicle_route_agent(self) -> None:
        """Apply one non-blocking BasicAgent control before the owned world tick."""
        if (
            not self.running
            or self._vehicle_route_agent is None
            or not v1.actor_alive(self.ego_vehicle)
        ):
            return
        try:
            if time.monotonic() < self._vehicle_reverse_until:
                # Stuck-recovery pulse: back straight out of whatever the ego
                # wedged against before the re-planned route resumes.
                self.ego_vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.40, steer=0.0, brake=0.0, reverse=True
                    )
                )
                return
            self._vehicle_route_agent.set_target_speed(
                self._vehicle_agent_target_speed()
            )
            control = self._vehicle_route_agent.run_step()
            control = self._apply_vehicle_lane_guard(control)
            self.ego_vehicle.apply_control(control)
        except Exception as exc:
            LOG.exception("Ego vehicle route agent failed")
            self.status = "Ego vehicle route agent failed: {}".format(exc)
            self._vehicle_route_agent = None
            self._vehicle_stalled_since = None
            self._vehicle_lane_guard_active = False
            try:
                self.ego_vehicle.apply_control(carla.VehicleControl(brake=1.0))
            except RuntimeError:
                pass

    def vehicle_autonomous(self) -> bool:
        return bool(
            self.last_config is not None
            and self.last_config["scenario"]["ego_vehicle"]["scripted_route"]
        )

    def pedestrian_autonomous(self) -> bool:
        return bool(
            self.last_config is not None
            and self.last_config["scenario"]["ego_pedestrian"]["scripted_route"]
        )

    def _remaining_vehicle_route_targets(self) -> List[carla.Location]:
        if self._vehicle_route_target is not None:
            targets = [self._vehicle_route_target]
            targets.extend(target for _, target in self._vehicle_route_segment_queue)
            return [v1.copy_location(target) for target in targets]
        if self._vehicle_route_total_targets > 0:
            return []
        return [
            v1.copy_location(target)
            for target in self._pending_vehicle_route_targets
        ]

    def _prune_passed_vehicle_targets(
        self,
        targets: Sequence[carla.Location],
    ) -> List[carla.Location]:
        """
        Drop gates the ego has already driven past so a re-plan goes forward.

        Without this, toggling Vehicle autonomous back ON while sitting on top
        of a via-point re-planned from the current pose back to that same point,
        which sent the ego around the block to approach it again.  A target is
        only dropped when it is both behind the ego and within reach distance;
        a genuinely distant target behind the ego is a real route leg.
        """
        remaining = [v1.copy_location(target) for target in targets]
        try:
            transform = self.ego_vehicle.get_transform()
            forward = transform.get_forward_vector()
        except (AttributeError, RuntimeError):
            return remaining
        location = transform.location
        while len(remaining) > 1:
            target = remaining[0]
            dx = float(target.x - location.x)
            dy = float(target.y - location.y)
            ahead = dx * float(forward.x) + dy * float(forward.y)
            if (
                ahead > 0.0
                or math.hypot(dx, dy)
                > self._vehicle_waypoint_reach_threshold_m
            ):
                break
            remaining.pop(0)
        return remaining

    def set_vehicle_control_mode(self, autonomous: bool) -> None:
        """Switch the live ego vehicle between BasicAgent and WASD control."""
        if not self.running or self.last_config is None:
            raise RuntimeError("start the demo before changing vehicle control mode")
        if not v1.actor_alive(self.ego_vehicle):
            raise RuntimeError("ego vehicle is unavailable")
        autonomous = bool(autonomous)
        if not autonomous:
            self._vehicle_route_agent = None
            self._vehicle_scripted_route_requested = False
            self.last_config["scenario"]["ego_vehicle"]["scripted_route"] = False
            self.ego_vehicle.set_autopilot(
                False,
                int(self.last_config["traffic_manager"]["port"]),
            )
            self.ego_vehicle.apply_control(carla.VehicleControl(brake=0.65))
            self.status = "Vehicle MANUAL (WASD); pedestrian {}".format(
                "AUTO" if self.pedestrian_autonomous() else "MANUAL"
            )
            self._log_event(
                "ego_vehicle_control_mode_changed",
                self.ego_vehicle,
                mode="manual",
            )
            return
        if self.vehicle_autonomous() and self._vehicle_route_agent is not None:
            return
        targets = self._prune_passed_vehicle_targets(
            self._remaining_vehicle_route_targets()
        )
        if not targets:
            raise RuntimeError("ego vehicle route is already complete")
        resolution = max(
            0.5,
            float(
                self.last_config["scenario"]["ego_vehicle"].get(
                    "route_sampling_resolution_m",
                    2.0,
                )
            ),
        )
        plan = self._trace_vehicle_agent_plan(
            self.ego_vehicle.get_location(),
            targets,
            resolution,
        )
        agent = self._create_vehicle_route_agent(self.last_config, plan)
        # Keep the ordered gate state consistent with the plan that is actually
        # installed, including any target pruned as already passed.
        self._vehicle_route_target = v1.copy_location(targets[0])
        self._vehicle_route_segment_queue = [
            ([], v1.copy_location(target)) for target in targets[1:]
        ]
        if self._vehicle_route_total_targets <= 0:
            self._vehicle_route_total_targets = len(targets)
            self._vehicle_route_target_number = 1
        else:
            self._vehicle_route_target_number = max(
                1, self._vehicle_route_total_targets - len(targets) + 1
            )
        self.ego_vehicle.set_autopilot(
            False,
            int(self.last_config["traffic_manager"]["port"]),
        )
        self._vehicle_route_agent = agent
        self._vehicle_scripted_route_requested = True
        self.last_config["scenario"]["ego_vehicle"]["scripted_route"] = True
        self.status = "Vehicle AUTO; pedestrian {}".format(
            "AUTO" if self.pedestrian_autonomous() else "MANUAL"
        )
        self._log_event(
            "ego_vehicle_control_mode_changed",
            self.ego_vehicle,
            mode="autonomous",
            remaining_targets=len(targets),
            route_points=len(plan),
            target_speed_kmh=self._vehicle_cruise_speed_kmh,
        )

    def set_vehicle_speed(self, cruise_speed_kmh: float) -> None:
        """
        Change the ego vehicle's speed while the scenario is running.

        In AUTO this is the BasicAgent cruise target, picked up on the next
        control period because _run_vehicle_route_agent() calls
        set_target_speed() every step. In MANUAL it is the speed the throttle is
        allowed to reach, so one control governs both modes. The configured
        curve/cruise relationship is preserved.
        """
        if self.last_config is None:
            return
        cruise = max(1.0, float(cruise_speed_kmh))
        vehicle_config = self.last_config["scenario"]["ego_vehicle"]
        previous = max(1.0, float(vehicle_config.get("scripted_speed_kmh", cruise)))
        ratio = v1.clamp(
            float(vehicle_config.get("scripted_curve_speed_kmh", previous * 0.55))
            / previous,
            0.1,
            1.0,
        )
        curve = max(1.0, cruise * ratio)
        vehicle_config["scripted_speed_kmh"] = cruise
        vehicle_config["scripted_curve_speed_kmh"] = curve
        self._vehicle_cruise_speed_kmh = cruise
        self._vehicle_curve_speed_kmh = curve
        self._log_event(
            "ego_vehicle_speed_changed",
            self.ego_vehicle,
            cruise_speed_kmh=cruise,
            curve_speed_kmh=curve,
            mode="autonomous" if self.vehicle_autonomous() else "manual",
        )

    def set_pedestrian_speed(self, walk_speed_mps: float) -> None:
        """
        Change the ego pedestrian's speed while the scenario is running.

        The manual WASD branch reads walk/run speed from the live config every
        frame, so writing them here takes effect immediately; the AI walker
        controller needs an explicit set_max_speed(). The configured run/walk
        relationship is preserved.
        """
        if self.last_config is None:
            return
        walk = max(0.1, float(walk_speed_mps))
        pedestrian_config = self.last_config["scenario"]["ego_pedestrian"]
        previous = max(0.1, float(pedestrian_config.get("walk_speed_mps", walk)))
        ratio = max(
            1.0,
            float(pedestrian_config.get("run_speed_mps", previous * 2.0)) / previous,
        )
        run = walk * ratio
        pedestrian_config["walk_speed_mps"] = walk
        pedestrian_config["run_speed_mps"] = run
        pedestrian_config["scripted_speed_mps"] = walk
        applied_to_controller = False
        if v1.actor_alive(self.ego_walker_controller):
            try:
                self.ego_walker_controller.set_max_speed(float(walk))
                applied_to_controller = True
            except RuntimeError as exc:
                LOG.warning("Unable to set the walker controller speed: %s", exc)
        self._log_event(
            "ego_pedestrian_speed_changed",
            self.ego_pedestrian,
            walk_speed_mps=walk,
            run_speed_mps=run,
            applied_to_ai_controller=applied_to_controller,
        )

    WALKER_CONTROL_SPEED_SCALE = 20.5
    WALKER_CONTROL_SPEED_CAP_MPS = 2.0

    def _ensure_pedestrian_guide(self, speed: float) -> Optional[carla.Actor]:
        """Lazily create the AI walker controller used to steer fast walking."""
        if v1.actor_alive(self.ego_walker_controller):
            return self.ego_walker_controller
        if not v1.actor_alive(self.ego_pedestrian):
            return None
        try:
            controller = v1.ScenarioController._spawn_walker_controller(
                self, self.ego_pedestrian, self.ego_pedestrian.get_location(), speed
            )
        except RuntimeError as exc:
            LOG.warning("Unable to start the manual pedestrian guide: %s", exc)
            return None
        self.ego_walker_controller = controller
        self.walker_controllers.append(controller)
        self._pedestrian_guide_active = True
        self._log_event(
            "ego_pedestrian_manual_guide_started",
            self.ego_pedestrian,
            target_speed_mps=float(speed),
        )
        return controller

    def _release_pedestrian_guide(self) -> None:
        """Remove the manual steering controller, e.g. when AUTO resumes."""
        if not self._pedestrian_guide_active:
            return
        controller = self.ego_walker_controller
        self._pedestrian_guide_active = False
        self.ego_walker_controller = None
        if controller is None:
            return
        try:
            controller.stop()
        except Exception:
            pass
        self._batch_destroy_owned_group([controller], "ego_pedestrian_manual_guide")
        self.walker_controllers = [
            item
            for item in self.walker_controllers
            if int(item.id) != int(controller.id)
        ]

    def _steer_pedestrian_guide(
        self,
        heading: Optional[carla.Vector3D],
        speed: float,
        pedestrian_config: Dict[str, Any],
    ) -> None:
        """
        Walk the pedestrian fast by aiming CARLA's own walker controller.

        WalkerControl.speed saturates around 2 m/s in this CARLA build, so a
        direct control cannot deliver the speeds this demo wants. The AI walker
        controller reaches roughly 15 m/s, so manual steering hands it a target a
        few metres ahead in the WASD heading and refreshes that target as the
        operator turns. Motion therefore stays on the navigation mesh.
        """
        controller = self._ensure_pedestrian_guide(speed)
        if controller is None:
            return
        now = time.monotonic()
        if heading is not None and now < self._next_pedestrian_guide_at:
            return
        try:
            location = self.ego_pedestrian.get_location()
        except RuntimeError:
            return
        if heading is None:
            target = location
        else:
            lookahead = max(
                2.0, float(pedestrian_config.get("guided_lookahead_m", 12.0))
            )
            target = carla.Location(
                x=float(location.x + heading.x * lookahead),
                y=float(location.y + heading.y * lookahead),
                z=float(location.z),
            )
        try:
            controller.set_max_speed(float(speed))
            controller.go_to_location(target)
        except RuntimeError as exc:
            LOG.debug("Manual pedestrian steering failed: %s", exc)
            return
        # A path recompute per UI frame is wasteful; ~10 Hz tracks WASD closely.
        self._next_pedestrian_guide_at = now + 0.1

    def _update_manual_pedestrian(
        self,
        keys: Sequence[bool],
        delta_seconds: float,
    ) -> None:
        """Manual walking that honours the requested speed despite the CARLA cap."""
        if not v1.actor_alive(self.ego_pedestrian):
            return
        pedestrian_config = self.last_config["scenario"]["ego_pedestrian"]
        delta_seconds = v1.clamp(delta_seconds, 0.0, 0.1)
        turn_axis = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
        self._pedestrian_body_yaw = (
            self._pedestrian_body_yaw + turn_axis * 90.0 * delta_seconds
        ) % 360.0
        move_axis = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
        running = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        desired = float(
            pedestrian_config["run_speed_mps"]
            if running
            else pedestrian_config["walk_speed_mps"]
        )
        heading = carla.Rotation(
            yaw=self._pedestrian_body_yaw
        ).get_forward_vector()
        if move_axis < 0:
            heading = carla.Vector3D(x=-heading.x, y=-heading.y, z=0.0)
        mode = str(
            pedestrian_config.get("manual_control_mode", "auto")
        ).lower()
        cap = max(
            0.1,
            float(
                pedestrian_config.get(
                    "direct_speed_cap_mps", self.WALKER_CONTROL_SPEED_CAP_MPS
                )
            ),
        )
        if mode == "guided" or (mode != "direct" and desired > cap):
            self._steer_pedestrian_guide(
                heading if move_axis else None, desired, pedestrian_config
            )
            return
        self._release_pedestrian_guide()
        # Below the cap a direct control is the most responsive option, but the
        # command must be pre-scaled or the walker moves at speed/20.5.
        scale = max(
            1.0,
            float(
                pedestrian_config.get(
                    "walker_control_speed_scale", self.WALKER_CONTROL_SPEED_SCALE
                )
            ),
        )
        control = carla.WalkerControl()
        control.direction = heading
        if move_axis:
            control.speed = desired * scale
        elif turn_axis:
            control.speed = 0.01 * scale
        else:
            control.speed = 0.0
        control.jump = bool(keys[pygame.K_SPACE])
        try:
            self.ego_pedestrian.apply_control(control)
        except RuntimeError:
            pass

    def update_controls(
        self,
        keys: Sequence[bool],
        delta_seconds: float,
    ) -> None:
        """Apply manual control, compensating for CARLA's walker speed limits."""
        if (
            self.running
            and self.last_config is not None
            and self.manual_target == "pedestrian"
            and not self.pedestrian_autonomous()
        ):
            self._update_manual_pedestrian(keys, delta_seconds)
            return
        super().update_controls(keys, delta_seconds)
        if (
            not self.running
            or self.last_config is None
            or self.manual_target != "vehicle"
            or self.vehicle_autonomous()
            or not v1.actor_alive(self.ego_vehicle)
        ):
            return
        vehicle_config = self.last_config["scenario"]["ego_vehicle"]
        limit_kmh = float(vehicle_config.get("manual_speed_limit_kmh", 0.0))
        if limit_kmh <= 0.0:
            limit_kmh = float(vehicle_config.get("scripted_speed_kmh", 18.0))
        if actor_speed_mps(self.ego_vehicle) * 3.6 <= max(1.0, limit_kmh):
            return
        # Over the requested speed: stop accelerating but keep the driver's
        # steering so the car stays controllable while it slows.
        try:
            self.ego_vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=float(self._vehicle_steer),
                    brake=0.25,
                    hand_brake=bool(keys[pygame.K_SPACE]),
                )
            )
        except RuntimeError:
            pass

    def set_pedestrian_control_mode(self, autonomous: bool) -> None:
        """Switch the live ego pedestrian between AI route and WASD control."""
        if not self.running or self.last_config is None:
            raise RuntimeError("start the demo before changing pedestrian control mode")
        if not v1.actor_alive(self.ego_pedestrian):
            raise RuntimeError("ego pedestrian is unavailable")
        autonomous = bool(autonomous)
        if not autonomous:
            controller = self.ego_walker_controller
            if controller is not None:
                try:
                    controller.stop()
                except Exception:
                    pass
                _, errors = self._batch_destroy_owned_group(
                    [controller],
                    "ego_walker_controller_mode_switch",
                )
                if errors:
                    raise RuntimeError("; ".join(errors))
                self.walker_controllers = [
                    item
                    for item in self.walker_controllers
                    if int(item.id) != int(controller.id)
                ]
                self.ego_walker_controller = None
            self.last_config["scenario"]["ego_pedestrian"]["scripted_route"] = False
            self.ego_pedestrian.apply_control(carla.WalkerControl())
            self.status = "Vehicle {}; pedestrian MANUAL (WASD)".format(
                "AUTO" if self.vehicle_autonomous() else "MANUAL"
            )
            self._log_event(
                "ego_pedestrian_control_mode_changed",
                self.ego_pedestrian,
                mode="manual",
            )
            return
        self._release_pedestrian_guide()
        if self.pedestrian_autonomous() and self.ego_walker_controller is not None:
            return
        if self._pedestrian_route_target is not None:
            targets = [v1.copy_location(self._pedestrian_route_target)]
            targets.extend(
                v1.copy_location(target) for target in self._pedestrian_route_queue
            )
        elif self._pedestrian_route_total_targets > 0:
            targets = []
        else:
            targets = [
                v1.copy_location(target)
                for target in self._pending_pedestrian_route_locations
            ]
        current_location = self.ego_pedestrian.get_location()
        while targets and targets[0].distance(current_location) <= 0.5:
            targets.pop(0)
        if not targets:
            raise RuntimeError("ego pedestrian route is already complete")
        speed = float(
            self.last_config["scenario"]["ego_pedestrian"][
                "scripted_speed_mps"
            ]
        )
        controller = v1.ScenarioController._spawn_walker_controller(
            self,
            self.ego_pedestrian,
            targets[0],
            speed,
        )
        self.ego_walker_controller = controller
        self.walker_controllers.append(controller)
        self._pedestrian_route_target = v1.copy_location(targets[0])
        self._pedestrian_route_queue = [
            v1.copy_location(target) for target in targets[1:]
        ]
        if self._pedestrian_route_total_targets == 0:
            self._pedestrian_route_target_number = 1
            self._pedestrian_route_total_targets = len(targets)
        self.last_config["scenario"]["ego_pedestrian"]["scripted_route"] = True
        self.status = "Vehicle {}; pedestrian AUTO".format(
            "AUTO" if self.vehicle_autonomous() else "MANUAL"
        )
        self._log_event(
            "ego_pedestrian_control_mode_changed",
            self.ego_pedestrian,
            mode="autonomous",
            remaining_targets=len(targets),
            target_speed_mps=speed,
        )

    @staticmethod
    def _passed_waypoint(
        location: carla.Location,
        waypoint: Any,
    ) -> bool:
        """True when a location lies ahead of a waypoint along its own heading."""
        try:
            forward = waypoint.transform.get_forward_vector()
            center = waypoint.transform.location
        except (AttributeError, RuntimeError):
            return False
        dx = float(location.x - center.x)
        dy = float(location.y - center.y)
        return dx * float(forward.x) + dy * float(forward.y) > 0.0

    def _advance_vehicle_route(self) -> None:
        """Record ordered route-gate progress while BasicAgent executes the plan."""
        if (
            not self.running
            or self._vehicle_route_target is None
            or not v1.actor_alive(self.ego_vehicle)
        ):
            return
        try:
            vehicle_location = self.ego_vehicle.get_location()
            distance = vehicle_location.distance(self._vehicle_route_target)
        except RuntimeError:
            return
        if distance > self._vehicle_waypoint_reach_threshold_m:
            return
        if self._vehicle_route_segment_queue:
            vehicle_waypoint = self._driving_waypoint(vehicle_location)
            target_waypoint = self._driving_waypoint(self._vehicle_route_target)
            if vehicle_waypoint is not None and target_waypoint is not None:
                vehicle_lane = (
                    int(vehicle_waypoint.road_id),
                    int(vehicle_waypoint.section_id),
                    int(vehicle_waypoint.lane_id),
                )
                target_lane = (
                    int(target_waypoint.road_id),
                    int(target_waypoint.section_id),
                    int(target_waypoint.lane_id),
                )
                # Requiring an exact lane match stalled route progress whenever
                # the target projected onto a neighbouring or junction lane: the
                # ego drove on but every later gate stayed pending, and a mode
                # toggle then re-planned back to a gate already behind it. A
                # target the ego has driven past counts as reached.
                if vehicle_lane != target_lane and not self._passed_waypoint(
                    vehicle_location, target_waypoint
                ):
                    return
        reached_number = self._vehicle_route_target_number
        reached_target = self._vehicle_route_target
        if not self._vehicle_route_segment_queue:
            self._log_event(
                "ego_vehicle_route_completed",
                self.ego_vehicle,
                route_targets=self._vehicle_route_total_targets,
                final_distance_m=float(distance),
            )
            self._vehicle_route_target = None
            self.status = (
                "Ego vehicle reached its destination ({} route targets); "
                "switch Vehicle autonomous OFF to drive manually"
            ).format(self._vehicle_route_total_targets)
            return
        _, next_target = self._vehicle_route_segment_queue[0]
        self._vehicle_route_segment_queue.pop(0)
        self._vehicle_route_target = next_target
        self._vehicle_route_target_number += 1
        waypoint_kind = (
            "destination"
            if self._vehicle_route_target_number == self._vehicle_route_total_targets
            else "via"
        )
        self._log_event(
            "ego_vehicle_route_target_reached",
            self.ego_vehicle,
            target_number=reached_number,
            route_targets=self._vehicle_route_total_targets,
            distance_m=float(distance),
            target={
                "x": float(reached_target.x),
                "y": float(reached_target.y),
                "z": float(reached_target.z),
            },
        )
        self._log_event(
            "ego_vehicle_route_target_activated",
            self.ego_vehicle,
            target_number=self._vehicle_route_target_number,
            route_targets=self._vehicle_route_total_targets,
            waypoint_kind=waypoint_kind,
            target={
                "x": float(next_target.x),
                "y": float(next_target.y),
                "z": float(next_target.z),
            },
        )

    def _spawn_walker_controller(
        self,
        walker: carla.Walker,
        destination: carla.Location,
        speed: float,
    ) -> carla.Actor:
        destinations = [v1.copy_location(destination)]
        if (
            walker is self.ego_pedestrian
            and self._pending_pedestrian_route_locations
        ):
            destinations = [
                v1.copy_location(location)
                for location in self._pending_pedestrian_route_locations
            ]
            # The walker was spawned moments ago, so its snapshot pose is only
            # valid after a tick. NPC walkers deliberately skip this.
            self._settle_actor_snapshot()
            walker_location = walker.get_location()
            while (
                len(destinations) > 1
                and destinations[0].distance(walker_location) < 0.5
            ):
                destinations.pop(0)
            # If the requested pedestrian start was occupied, the base
            # controller chooses a deterministic fallback and may also move an
            # end point that now equals that fallback. Honor the corrected end.
            if destinations[-1].distance(walker_location) < 0.5:
                destinations[-1] = v1.copy_location(destination)
        controller = super()._spawn_walker_controller(
            walker,
            destinations[0],
            speed,
        )
        if walker is self.ego_pedestrian:
            self._pedestrian_route_target = destinations[0]
            self._pedestrian_route_queue = destinations[1:]
            self._pedestrian_route_target_number = 1
            self._pedestrian_route_total_targets = len(destinations)
        return controller

    def _advance_pedestrian_route(self) -> None:
        """Issue the next walker destination after reaching the active target."""
        if (
            not self.running
            or self.ego_walker_controller is None
            or self._pedestrian_route_target is None
            or not v1.actor_alive(self.ego_pedestrian)
            or not v1.actor_alive(self.ego_walker_controller)
        ):
            return
        try:
            distance = self.ego_pedestrian.get_location().distance(
                self._pedestrian_route_target
            )
        except RuntimeError:
            return
        if distance > self._pedestrian_waypoint_reach_threshold_m:
            return
        if not self._pedestrian_route_queue:
            self._log_event(
                "ego_pedestrian_route_completed",
                self.ego_pedestrian,
                route_targets=self._pedestrian_route_total_targets,
            )
            self._pedestrian_route_target = None
            return
        next_target = self._pedestrian_route_queue[0]
        try:
            self.ego_walker_controller.go_to_location(next_target)
        except RuntimeError as exc:
            LOG.warning("Unable to advance ego pedestrian route: %s", exc)
            return
        self._pedestrian_route_queue.pop(0)
        self._pedestrian_route_target = next_target
        self._pedestrian_route_target_number += 1
        self._log_event(
            "ego_pedestrian_route_advanced",
            self.ego_pedestrian,
            target_number=self._pedestrian_route_target_number,
            route_targets=self._pedestrian_route_total_targets,
            target={
                "x": float(self._pedestrian_route_target.x),
                "y": float(self._pedestrian_route_target.y),
                "z": float(self._pedestrian_route_target.z),
            },
        )

    def start(self, config: Dict[str, Any]) -> None:
        self.occluder_blueprint_resolution = None
        self.occluder_lateral_offset_resolved = None
        self._route_detour_warnings = []
        cleanup_result = self.stop_demo(remove_tick_callback=False)
        if cleanup_result["errors"]:
            raise RuntimeError(
                "cannot start a new demo while owned CARLA actors remain; "
                "press Stop again or run with --cleanup-only"
            )
        self._prepare_intermediate_routes(config)
        try:
            super().start(config)
            if bool(config["scenario"]["ego_vehicle"]["scripted_route"]):
                self._start_vehicle_agent_route(config)
        except Exception:
            self.stop_demo(remove_tick_callback=False)
            raise
        sensor_count = len(
            self.camera.owned_sensors()
            if hasattr(self.camera, "owned_sensors")
            else [self.camera.sensor]
        )
        # The base status text predates the dedicated ego sensors.
        self.status = self.status.replace(
            "1 shared RGB sensor", "{} RGB sensors".format(sensor_count)
        )
        ego_suffix = "Ego: {}".format(self.ego_vehicle.type_id)
        resolution = self.occluder_blueprint_resolution
        if resolution is not None:
            suffix = "Occluder: {}".format(resolution["blueprint_id"])
            if resolution["substituted"]:
                suffix += " ({} substitute)".format(
                    resolution["requested_type"]
                )
            self.status = "{}; {}; {}; via: vehicle {}, pedestrian {}".format(
                self.status,
                ego_suffix,
                suffix,
                len(self._configured_vehicle_waypoint_indices),
                len(self._configured_pedestrian_waypoint_indices),
            )
        else:
            self.status = "{}; {}; via: vehicle {}, pedestrian {}".format(
                self.status,
                ego_suffix,
                len(self._configured_vehicle_waypoint_indices),
                len(self._configured_pedestrian_waypoint_indices),
            )
        if self._vehicle_scripted_route_requested:
            self.status += "; vehicle route: BasicAgent"
        offset_resolution = self.occluder_lateral_offset_resolved
        if offset_resolution is not None:
            self.status += "; occluder moved to {:+.2f} m lateral to clear the ego lane".format(
                offset_resolution["resolved_lateral_offset_m"]
            )
        if self._route_detour_warnings:
            self.status += "; route warning: {}".format(
                "; ".join(self._route_detour_warnings[:2])
            )
        self._log_event(
            "route_waypoints_configured",
            vehicle_waypoint_indices=self._configured_vehicle_waypoint_indices,
            pedestrian_waypoint_indices=self._configured_pedestrian_waypoint_indices,
            vehicle_reach_threshold_m=self._vehicle_waypoint_reach_threshold_m,
            vehicle_route_controller=(
                "basic_agent" if self._vehicle_scripted_route_requested else "manual"
            ),
        )

    @staticmethod
    def _unique_actors(
        actors: Sequence[Optional[carla.Actor]],
        excluded_ids: Optional[set] = None,
    ) -> List[carla.Actor]:
        """Return non-null actors once, without querying their server state."""
        seen = set() if excluded_ids is None else excluded_ids
        unique = []
        for actor in actors:
            if actor is None:
                continue
            try:
                actor_id = int(actor.id)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
            if actor_id in seen:
                continue
            seen.add(actor_id)
            unique.append(actor)
        return unique

    @staticmethod
    def _stop_sensor_callback(sensor: Optional[carla.Sensor]) -> None:
        if sensor is None:
            return
        # is_listening is local to the Python sensor wrapper. A sensor
        # rediscovered through world.get_actors() was never subscribed by that
        # wrapper, and stop() on it emits a misleading warning. CARLA 0.10
        # exposes is_listening as a method, so this must not be a bare bool().
        if sensor_is_listening(sensor) is False:
            return
        try:
            sensor.stop()
        except Exception:
            pass

    @staticmethod
    def _actor_type(actor: carla.Actor) -> str:
        try:
            return str(actor.type_id)
        except (AttributeError, RuntimeError):
            return ""

    @staticmethod
    def _actor_role(actor: carla.Actor) -> str:
        try:
            return str(actor.attributes.get("role_name", ""))
        except (AttributeError, RuntimeError):
            return ""

    @staticmethod
    def _parent_id(actor: carla.Actor) -> Optional[int]:
        try:
            parent = actor.parent
            return None if parent is None else int(parent.id)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    def discover_owned_actor_groups(self) -> Dict[str, List[carla.Actor]]:
        """
        Find actors left by this UI, including actors from an older process.

        Root actors and explicitly named sensors carry ``physical_ai_*`` role
        names. CARLA AI walker controllers do not expose a configurable role,
        so controllers and sensors attached to an owned root are included too.
        Unrelated traffic is never selected.
        """
        actors = list(self.world.get_actors())
        owned_ids = {
            int(actor.id)
            for actor in actors
            if self._actor_role(actor).startswith(OWNED_ROLE_PREFIX)
        }
        owned = []
        for actor in actors:
            try:
                actor_id = int(actor.id)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
            if actor_id in owned_ids or self._parent_id(actor) in owned_ids:
                owned.append(actor)

        sensors = []
        controllers = []
        scenario_actors = []
        for actor in owned:
            type_id = self._actor_type(actor)
            if type_id.startswith("sensor."):
                sensors.append(actor)
            elif type_id == "controller.ai.walker":
                controllers.append(actor)
            else:
                scenario_actors.append(actor)
        return {
            "sensor": sensors,
            "walker_controller": controllers,
            "scenario_actor": scenario_actors,
        }

    def _actor_still_exists(self, actor_id: int) -> Tuple[bool, Optional[str]]:
        try:
            actor = self.world.get_actor(int(actor_id))
            if actor is None:
                return False, None
            try:
                return bool(actor.is_alive), None
            except (AttributeError, RuntimeError):
                return True, None
        except Exception as exc:
            return True, "actor {} confirmation failed: {}".format(actor_id, exc)

    def _remaining_actor_ids(
        self, actor_ids: Sequence[int]
    ) -> Tuple[List[int], List[str]]:
        remaining = []
        errors = []
        for actor_id in actor_ids:
            exists, error = self._actor_still_exists(actor_id)
            if exists:
                remaining.append(int(actor_id))
            if error is not None:
                errors.append(error)
        return remaining, errors

    def _batch_destroy_owned_group(
        self,
        actors: Sequence[carla.Actor],
        group_name: str,
    ) -> Tuple[int, List[str]]:
        """Destroy, tick-flush, confirm, and retry one ownership group."""
        if not actors:
            return 0, []
        actor_ids = [int(actor.id) for actor in actors]
        actor_by_id = {int(actor.id): actor for actor in actors}
        remaining = list(actor_ids)

        # The workspace's reliable CARLA cleanup clients use do_tick=True so
        # destruction is processed before confirmation. Retry once because a
        # dependent controller/sensor can occasionally outlive the first RPC.
        for attempt in (1, 2):
            commands = [
                carla.command.DestroyActor(actor_id) for actor_id in remaining
            ]
            response_errors: Dict[int, str] = {}
            batch_error: Optional[Exception] = None
            try:
                responses = self.client.apply_batch_sync(commands, True)
                for actor_id, response in zip(remaining, responses):
                    response_error = getattr(response, "error", None)
                    if response_error:
                        response_errors[int(actor_id)] = str(response_error)
            except Exception as exc:
                batch_error = exc
            remaining, _ = self._remaining_actor_ids(remaining)
            if not remaining:
                return len(actor_ids), []
            if batch_error is not None:
                LOG.warning(
                    "%s batch destroy attempt %d failed with %d actor(s) "
                    "still present: %s",
                    group_name,
                    attempt,
                    len(remaining),
                    batch_error,
                )
            for actor_id in remaining:
                response_error = response_errors.get(actor_id)
                if response_error is not None:
                    LOG.warning(
                        "%s actor %d destroy attempt %d: %s",
                        group_name,
                        actor_id,
                        attempt,
                        response_error,
                    )

        # Last-resort individual destruction follows patterns used by other
        # clients in this workspace, followed by an explicit server tick.
        for actor_id in remaining:
            actor = actor_by_id.get(actor_id)
            if actor is None:
                continue
            try:
                actor.destroy()
            except Exception as exc:
                LOG.warning(
                    "%s actor %d individual destroy failed: %s",
                    group_name,
                    actor_id,
                    exc,
                )
        try:
            self.world.tick()
        except Exception as exc:
            LOG.warning("Cleanup confirmation tick failed: %s", exc)
        remaining, confirmation_errors = self._remaining_actor_ids(remaining)
        errors = list(confirmation_errors)
        if remaining:
            errors.append(
                "{} actor ids remain in CARLA after retries: {}".format(
                    group_name, remaining
                )
            )
        return len(actor_ids) - len(remaining), errors

    def stop_demo(self, remove_tick_callback: bool = False) -> Dict[str, Any]:
        """
        Stop the demo and delete every actor owned by this UI version.

        Sensor callbacks and AI walker controllers are stopped before actor
        deletion.  Destruction is ordered as sensors, controllers, then moving
        actors. Each batch advances the owned master clock and server-side actor
        lookup confirms deletion. Actors from an earlier process are discovered
        by role name. Errors stay in the UI instead of escaping the event loop.
        """
        was_running = bool(self.running)
        self.running = False
        errors: List[str] = []
        seen_ids: set = set()

        camera_sensors: List[carla.Sensor] = []
        if self.camera is not None:
            owned_sensors = getattr(self.camera, "owned_sensors", None)
            if callable(owned_sensors):
                camera_sensors = list(owned_sensors())
            elif self.camera.sensor is not None:
                camera_sensors = [self.camera.sensor]
        try:
            discovered = self.discover_owned_actor_groups()
        except Exception as exc:
            LOG.exception("Unable to discover owned CARLA actors")
            errors.append("owned-actor discovery failed: {}".format(exc))
            discovered = {
                "sensor": [],
                "walker_controller": [],
                "scenario_actor": [],
            }

        previous_survivors = list(getattr(self, "_cleanup_survivors", []))
        survivor_sensors = [
            actor
            for actor in previous_survivors
            if self._actor_type(actor).startswith("sensor.")
        ]
        survivor_controllers = [
            actor
            for actor in previous_survivors
            if self._actor_type(actor) == "controller.ai.walker"
        ]
        survivor_scenario_actors = [
            actor
            for actor in previous_survivors
            if not self._actor_type(actor).startswith("sensor.")
            and self._actor_type(actor) != "controller.ai.walker"
        ]

        # Tracked wrappers must come first. Sensor subscription state belongs
        # to the wrapper that called listen(); a wrapper returned by discovery
        # for the same actor ID reports is_listening=False.
        tracked_sensors = camera_sensors + [self.radar] + survivor_sensors
        sensors = self._unique_actors(
            tracked_sensors + discovered["sensor"], seen_ids
        )
        for sensor in sensors:
            self._stop_sensor_callback(sensor)
        if self.camera is not None:
            self.camera.listening = False
            mailboxes = getattr(self.camera, "mailboxes", None)
            for mailbox in (
                mailboxes() if callable(mailboxes) else [self.camera.mailbox]
            ):
                mailbox.clear()

        controllers = self._unique_actors(
            list(self.walker_controllers)
            + survivor_controllers
            + discovered["walker_controller"],
            seen_ids,
        )
        for controller in controllers:
            try:
                controller.stop()
            except Exception:
                pass

        moving_actors = self._unique_actors(
            list(self.npc_walkers)
            + list(self.npc_vehicles)
            + [self.ego_pedestrian, self.occluder, self.ego_vehicle]
            + survivor_scenario_actors
            + discovered["scenario_actor"],
            seen_ids,
        )

        # Allow callback unsubscription and controller.stop() to reach the
        # server before issuing DestroyActor. This mirrors the sensor-first,
        # tick, confirm pattern used by the workspace cleanup references.
        if sensors or controllers:
            try:
                self.world.tick()
                self._next_world_tick_at = (
                    time.monotonic() + self.fixed_delta_seconds
                )
            except Exception as exc:
                LOG.warning("Pre-destroy cleanup tick failed: %s", exc)

        groups = (
            (sensors, "sensor"),
            (controllers, "walker_controller"),
            (moving_actors, "scenario_actor"),
        )
        requested = sum(len(actors) for actors, _ in groups)
        destroyed = 0
        for actors, group_name in groups:
            group_destroyed, group_errors = self._batch_destroy_owned_group(
                actors, group_name
            )
            destroyed += group_destroyed
            errors.extend(group_errors)

        attempted_by_id = {
            int(actor.id): actor
            for actors, _ in groups
            for actor in actors
        }
        self._cleanup_survivors = []
        try:
            lingering_groups = self.discover_owned_actor_groups()
            lingering_actors = [
                actor
                for actors in lingering_groups.values()
                for actor in actors
            ]
            lingering_ids = sorted(
                int(actor.id) for actor in lingering_actors
            )
            self._cleanup_survivors = self._unique_actors(
                [
                    attempted_by_id.get(int(actor.id), actor)
                    for actor in lingering_actors
                ]
            )
            if lingering_ids:
                errors.append(
                    "owned actor ids still present after cleanup: {}".format(
                        lingering_ids
                    )
                )
        except Exception as exc:
            errors.append("post-cleanup actor discovery failed: {}".format(exc))
            for actor_id, actor in attempted_by_id.items():
                exists, _ = self._actor_still_exists(actor_id)
                if exists:
                    self._cleanup_survivors.append(actor)
            self._cleanup_survivors = self._unique_actors(
                self._cleanup_survivors
            )

        if was_running or requested:
            try:
                self._log_event(
                    "scenario_stopped",
                    destroy_requested=requested,
                    destroy_succeeded=destroyed,
                    destroy_errors=errors,
                )
            except Exception as exc:
                errors.append("stop-event logging failed: {}".format(exc))

        # Drop normal scenario references. Any actor still confirmed alive is
        # retained in _cleanup_survivors so a live sensor wrapper cannot fall
        # out of scope before the next cleanup retry.
        if self.camera is not None:
            forget_sensors = getattr(self.camera, "forget_sensors", None)
            if callable(forget_sensors):
                forget_sensors()
            else:
                self.camera.sensor = None
        self.camera = None
        self.radar = None
        self.radar_mailbox.clear()
        self.walker_controllers.clear()
        self.npc_walkers.clear()
        self.npc_vehicles.clear()
        self.ego_walker_controller = None
        self.ego_pedestrian = None
        self.occluder = None
        self.ego_vehicle = None
        self._vehicle_route.clear()
        self._vehicle_route_segment_queue.clear()
        self._vehicle_route_target = None
        self._vehicle_route_target_number = 0
        self._vehicle_route_total_targets = 0
        self._vehicle_route_agent = None
        self._vehicle_stalled_since = None
        self._vehicle_stuck_recovery_used = 0
        self._vehicle_reverse_until = 0.0
        self._vehicle_lane_guard_active = False
        self._vehicle_lane_guard_status = ""
        self._pedestrian_guide_active = False
        self._pedestrian_route_queue.clear()
        self._pedestrian_route_target = None
        self._pedestrian_route_target_number = 0
        self._pedestrian_route_total_targets = 0
        self._projection_cache.actors.clear()
        self._projection_cache.next_refresh = 0.0

        if remove_tick_callback and self._tick_callback_id is not None:
            try:
                self.world.remove_on_tick(self._tick_callback_id)
            except Exception as exc:
                errors.append("tick callback removal failed: {}".format(exc))
            self._tick_callback_id = None

        if errors:
            self.status = (
                "Demo stopped locally; deleted {}/{} owned actors; {} cleanup "
                "error(s) logged"
            ).format(destroyed, requested, len(errors))
        else:
            self.status = "Demo stopped; deleted {}/{} owned actors".format(
                destroyed, requested
            )
        return {
            "requested": requested,
            "destroyed": destroyed,
            "errors": errors,
        }

    def cleanup(self, remove_tick_callback: bool = False) -> None:
        """Route every v2 reset, failure, replay, and shutdown through stop_demo."""
        self.stop_demo(remove_tick_callback=remove_tick_callback)

    def shutdown(self) -> Dict[str, Any]:
        """Clean up actors/callbacks, then relinquish the master clock."""
        result = self.stop_demo(remove_tick_callback=True)
        self.release_master_clock()
        return result

    def _navigation_locations(
        self,
        seed: int,
        count: int = 128,
    ) -> List[carla.Location]:
        seed = int(seed)
        if (
            self._navigation_preview_seed == seed
            and len(self._navigation_preview) >= min(2, count)
        ):
            return [v1.copy_location(location) for location in self._navigation_preview]
        locations = super()._navigation_locations(seed, count)
        self._navigation_preview_seed = seed
        self._navigation_preview = [
            v1.copy_location(location) for location in locations
        ]
        return [v1.copy_location(location) for location in locations]

    def navigation_preview(self, seed: int) -> List[carla.Location]:
        """Return the exact seeded list that the next scenario start will use."""
        return self._navigation_locations(int(seed), count=128)


class TopDownMapSelector:
    """Zoomable, pannable Pygame map for indexed route endpoint selection."""

    def __init__(
        self,
        rect: pygame.Rect,
        road_locations: Sequence[carla.Location],
        road_polylines: Sequence[Sequence[carla.Location]],
        building_footprints: Sequence[Sequence[carla.Location]],
        vehicle_points: Sequence[carla.Transform],
        pedestrian_points: Sequence[carla.Location],
        max_zoom: float,
        scale: float = 1.0,
    ) -> None:
        self.scale_factor = max(0.5, float(scale))
        self.rect = rect.copy()
        inset = int(round(24 * self.scale_factor))
        header = int(round(78 * self.scale_factor))
        footer = int(round(24 * self.scale_factor))
        self._inset = inset
        self.plot_rect = pygame.Rect(
            self.rect.left + inset,
            self.rect.top + header,
            self.rect.width - 2 * inset,
            self.rect.height - header - footer,
        )
        self.road_locations = list(road_locations)
        self._road_selection_cell_m = 10.0
        self._road_selection_grid: Dict[
            Tuple[int, int], List[Tuple[int, carla.Location]]
        ] = {}
        for index, location in enumerate(self.road_locations):
            key = (
                math.floor(float(location.x) / self._road_selection_cell_m),
                math.floor(float(location.y) / self._road_selection_cell_m),
            )
            self._road_selection_grid.setdefault(key, []).append(
                (index, location)
            )
        self.road_polylines = [
            self._smooth_polyline(polyline, passes=2)
            for polyline in road_polylines
            if len(polyline) >= 2
        ]
        self.building_footprints = [
            [v1.copy_location(location) for location in footprint]
            for footprint in building_footprints
            if len(footprint) >= 3
        ]
        self.vehicle_points = list(vehicle_points)
        self.pedestrian_points = list(pedestrian_points)
        self.max_zoom = max(1.0, float(max_zoom))
        self.zoom = 1.0
        self.dragging = False
        self.last_drag_position = (0, 0)
        self.hover: Optional[Tuple[int, carla.Location]] = None
        self.endpoint_font = pygame.font.Font(
            pygame.font.get_default_font(),
            max(9, int(round(DESIGN_MAP_LABEL_FONT_SIZE * self.scale_factor))),
        )
        self._static_layer_key: Optional[Tuple[Any, ...]] = None
        self._static_layer_surface: Optional[pygame.Surface] = None
        self._calculate_bounds()
        self.reset_view()

    def _sy(self, value: float) -> int:
        """Scale a design-space pixel offset for the current window size."""
        return int(round(float(value) * self.scale_factor))

    @staticmethod
    def _smooth_polyline(
        points: Sequence[carla.Location],
        passes: int,
    ) -> List[carla.Location]:
        """Apply lightweight Chaikin smoothing once at UI construction."""
        result = [v1.copy_location(point) for point in points]
        for _ in range(max(0, int(passes))):
            if len(result) < 3:
                break
            smoothed = [v1.copy_location(result[0])]
            for first, second in zip(result, result[1:]):
                smoothed.append(
                    carla.Location(
                        x=0.75 * first.x + 0.25 * second.x,
                        y=0.75 * first.y + 0.25 * second.y,
                        z=0.75 * first.z + 0.25 * second.z,
                    )
                )
                smoothed.append(
                    carla.Location(
                        x=0.25 * first.x + 0.75 * second.x,
                        y=0.25 * first.y + 0.75 * second.y,
                        z=0.25 * first.z + 0.75 * second.z,
                    )
                )
            smoothed.append(v1.copy_location(result[-1]))
            result = smoothed
        return result

    def _calculate_bounds(self) -> None:
        locations = list(self.road_locations)
        locations.extend(
            location
            for footprint in self.building_footprints
            for location in footprint
        )
        locations.extend(point_location(point) for point in self.vehicle_points)
        locations.extend(self.pedestrian_points)
        if not locations:
            locations = [carla.Location()]
        x_values = [float(location.x) for location in locations]
        y_values = [float(location.y) for location in locations]
        self.min_x = min(x_values)
        self.max_x = max(x_values)
        self.min_y = min(y_values)
        self.max_y = max(y_values)
        span_x = max(10.0, self.max_x - self.min_x)
        span_y = max(10.0, self.max_y - self.min_y)
        padding = max(10.0, 0.04 * max(span_x, span_y))
        self.min_x -= padding
        self.max_x += padding
        self.min_y -= padding
        self.max_y += padding

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.center_x = (self.min_x + self.max_x) / 2.0
        self.center_y = (self.min_y + self.max_y) / 2.0

    def set_pedestrian_points(
        self,
        points: Sequence[carla.Location],
    ) -> None:
        self.pedestrian_points = list(points)
        self._calculate_bounds()
        self.reset_view()

    @property
    def base_scale(self) -> float:
        span_x = max(1.0, self.max_x - self.min_x)
        span_y = max(1.0, self.max_y - self.min_y)
        return min(self.plot_rect.width / span_x, self.plot_rect.height / span_y)

    @property
    def scale(self) -> float:
        return self.base_scale * self.zoom

    def world_to_screen(self, location: carla.Location) -> Tuple[int, int]:
        x = self.plot_rect.centerx + (float(location.x) - self.center_x) * self.scale
        # Match traffic_lights_map.png: CARLA +Y is drawn downward.
        y = self.plot_rect.centery + (float(location.y) - self.center_y) * self.scale
        return int(round(x)), int(round(y))

    def screen_to_world(self, position: Tuple[int, int]) -> Tuple[float, float]:
        x = self.center_x + (position[0] - self.plot_rect.centerx) / self.scale
        y = self.center_y + (position[1] - self.plot_rect.centery) / self.scale
        return x, y

    def _points_for_mode(self, mode: str) -> Sequence[Any]:
        if mode == "vehicle_waypoints":
            return self.road_locations
        if mode.startswith("vehicle_"):
            return self.vehicle_points
        return self.pedestrian_points

    def _nearest_point(
        self,
        position: Tuple[int, int],
        mode: str,
        maximum_pixels: float = 18.0,
    ) -> Optional[Tuple[int, carla.Location]]:
        best = None
        best_distance = maximum_pixels * maximum_pixels
        candidates: Sequence[Tuple[int, Any]]
        if mode == "vehicle_waypoints":
            world_x, world_y = self.screen_to_world(position)
            radius_m = maximum_pixels / max(self.scale, 1e-6)
            cell_radius = max(
                1,
                int(math.ceil(radius_m / self._road_selection_cell_m)),
            )
            cell_x = math.floor(world_x / self._road_selection_cell_m)
            cell_y = math.floor(world_y / self._road_selection_cell_m)
            road_candidates: List[Tuple[int, carla.Location]] = []
            for offset_x in range(-cell_radius, cell_radius + 1):
                for offset_y in range(-cell_radius, cell_radius + 1):
                    road_candidates.extend(
                        self._road_selection_grid.get(
                            (cell_x + offset_x, cell_y + offset_y), []
                        )
                    )
            candidates = road_candidates
        else:
            candidates = list(enumerate(self._points_for_mode(mode)))
        for index, point in candidates:
            location = point_location(point)
            screen_position = self.world_to_screen(location)
            dx = screen_position[0] - position[0]
            dy = screen_position[1] - position[1]
            distance = dx * dx + dy * dy
            if distance <= best_distance:
                best = (index, location)
                best_distance = distance
        return best

    def handle_event(
        self,
        event: pygame.event.Event,
        mode: str,
    ) -> Optional[int]:
        mouse_position = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEWHEEL and self.plot_rect.collidepoint(mouse_position):
            before_x, before_y = self.screen_to_world(mouse_position)
            factor = 1.25 if event.y > 0 else 1.0 / 1.25
            self.zoom = v1.clamp(self.zoom * factor, 1.0, self.max_zoom)
            new_scale = self.scale
            self.center_x = before_x - (
                mouse_position[0] - self.plot_rect.centerx
            ) / new_scale
            self.center_y = before_y - (
                mouse_position[1] - self.plot_rect.centery
            ) / new_scale
            return None
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button in (2, 3)
            and self.plot_rect.collidepoint(event.pos)
        ):
            self.dragging = True
            self.last_drag_position = event.pos
            return None
        if event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
            self.dragging = False
            return None
        if event.type == pygame.MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.last_drag_position[0]
            dy = event.pos[1] - self.last_drag_position[1]
            self.center_x -= dx / self.scale
            self.center_y -= dy / self.scale
            self.last_drag_position = event.pos
            return None
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.plot_rect.collidepoint(event.pos)
        ):
            nearest = self._nearest_point(event.pos, mode)
            return None if nearest is None else nearest[0]
        return None

    @staticmethod
    def _nice_grid_spacing(raw_spacing: float) -> float:
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
    def _offset(
        position: Tuple[int, int],
        origin: Tuple[int, int],
    ) -> Tuple[int, int]:
        return position[0] + origin[0], position[1] + origin[1]

    def _draw_grid(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        origin: Tuple[int, int] = (0, 0),
    ) -> None:
        spacing = self._nice_grid_spacing(85.0 / self.scale)
        left_world, top_world = self.screen_to_world(self.plot_rect.topleft)
        right_world, bottom_world = self.screen_to_world(self.plot_rect.bottomright)
        minimum_x, maximum_x = sorted((left_world, right_world))
        minimum_y, maximum_y = sorted((top_world, bottom_world))
        start_x = math.floor(minimum_x / spacing) * spacing
        start_y = math.floor(minimum_y / spacing) * spacing
        top = self.plot_rect.top + origin[1]
        bottom = self.plot_rect.bottom + origin[1]
        left = self.plot_rect.left + origin[0]
        right = self.plot_rect.right + origin[0]
        value = start_x
        while value <= maximum_x:
            x, _ = self.world_to_screen(carla.Location(x=value, y=self.center_y))
            x += origin[0]
            pygame.draw.line(screen, COLOR_GRID, (x, top), (x, bottom), 1)
            label = font.render("{:.0f}".format(value), True, v1.COLOR_MUTED)
            screen.blit(label, (x + 3, bottom - self._sy(17)))
            value += spacing
        value = start_y
        while value <= maximum_y:
            _, y = self.world_to_screen(carla.Location(x=self.center_x, y=value))
            y += origin[1]
            pygame.draw.line(screen, COLOR_GRID, (left, y), (right, y), 1)
            label = font.render("{:.0f}".format(value), True, v1.COLOR_MUTED)
            screen.blit(label, (left + 3, y - self._sy(16)))
            value += spacing

    def _draw_endpoint(
        self,
        screen: pygame.Surface,
        location: Optional[carla.Location],
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        if location is None:
            return
        position = self.world_to_screen(location)
        pygame.draw.circle(screen, v1.COLOR_BG, position, 10)
        pygame.draw.circle(screen, color, position, 9, 3)
        text = self.endpoint_font.render(label, True, color, v1.COLOR_BG)
        screen.blit(text, (position[0] + 11, position[1] - 9))

    def _draw_route_waypoint(
        self,
        screen: pygame.Surface,
        location: carla.Location,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        position = self.world_to_screen(location)
        pygame.draw.circle(screen, v1.COLOR_BG, position, 8)
        pygame.draw.circle(screen, color, position, 6)
        pygame.draw.circle(screen, COLOR_WAYPOINT_TEXT, position, 6, 1)
        text = self.endpoint_font.render(label, True, COLOR_WAYPOINT_TEXT, v1.COLOR_BG)
        screen.blit(text, (position[0] + 8, position[1] - 8))

    def _static_map_layer(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
    ) -> pygame.Surface:
        """Cache the grid, buildings, and lane geometry until the view moves."""
        cache_key = (
            screen.get_size(),
            self.plot_rect.x,
            self.plot_rect.y,
            self.plot_rect.width,
            self.plot_rect.height,
            round(self.zoom, 6),
            round(self.center_x, 6),
            round(self.center_y, 6),
        )
        if (
            self._static_layer_surface is not None
            and self._static_layer_key == cache_key
        ):
            return self._static_layer_surface

        # An opaque surface the size of the plot, not a full-window SRCALPHA
        # one: this is blitted every frame, and a per-pixel alpha blend of the
        # whole window cost enough to hold the simulation clock below 8 Hz.
        layer = pygame.Surface(self.plot_rect.size)
        layer.fill(COLOR_PLOT_BG)
        layer_origin = (-self.plot_rect.x, -self.plot_rect.y)
        self._draw_grid(layer, font, layer_origin)
        for footprint in self.building_footprints:
            screen_points = [
                self._offset(self.world_to_screen(location), layer_origin)
                for location in footprint
            ]
            pygame.draw.polygon(layer, COLOR_BUILDING_FILL, screen_points)
            pygame.draw.polygon(layer, COLOR_BUILDING_EDGE, screen_points, 1)
        for polyline in self.road_polylines:
            screen_points = [
                self._offset(self.world_to_screen(location), layer_origin)
                for location in polyline
            ]
            if len(screen_points) >= 2:
                pygame.draw.lines(
                    layer,
                    COLOR_LANE_CENTERLINE,
                    False,
                    screen_points,
                    2,
                )
        self._static_layer_key = cache_key
        self._static_layer_surface = layer
        return layer

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        mode: str,
        indices: Dict[str, int],
        vehicle_waypoint_indices: Sequence[int],
        pedestrian_waypoint_indices: Sequence[int],
        vehicle_route_preview: Sequence[carla.Location] = (),
    ) -> None:
        pygame.draw.rect(screen, v1.COLOR_BG, self.rect)
        title = font.render(
            "Top-down route waypoint selector", True, v1.COLOR_TEXT
        )
        screen.blit(title, (self.rect.left + self._inset, self.rect.top + self._sy(16)))
        click_action = (
            "left-click appends waypoint"
            if mode.endswith("_waypoints")
            else "left-click selects point"
        )
        instruction = small_font.render(
            "Active: {} | {} | wheel zoom | right-drag pan".format(
                MAP_MODE_LABELS[mode], click_action
            ),
            True,
            v1.COLOR_MUTED,
        )
        screen.blit(
            instruction, (self.rect.left + self._inset, self.rect.top + self._sy(45))
        )
        zoom_text = small_font.render(
            "Zoom {:.1f}x".format(self.zoom), True, v1.COLOR_ACCENT
        )
        screen.blit(
            zoom_text,
            (
                self.rect.right - zoom_text.get_width() - self._inset,
                self.rect.top + self._sy(18),
            ),
        )

        old_clip = screen.get_clip()
        screen.set_clip(self.plot_rect)
        screen.blit(
            self._static_map_layer(screen, small_font), self.plot_rect.topleft
        )
        for point in self.vehicle_points:
            position = self.world_to_screen(point.location)
            if self.plot_rect.collidepoint(position):
                pygame.draw.circle(screen, COLOR_VEHICLE_POINT, position, 3)
        for location in self.pedestrian_points:
            position = self.world_to_screen(location)
            if self.plot_rect.collidepoint(position):
                pygame.draw.circle(screen, COLOR_PEDESTRIAN_POINT, position, 2)

        selected_locations: Dict[str, Optional[carla.Location]] = {}
        for key, index in indices.items():
            points = self._points_for_mode(key)
            selected_locations[key] = (
                point_location(points[index % len(points)]) if points else None
            )
        vehicle_start = selected_locations["vehicle_start"]
        vehicle_end = selected_locations["vehicle_end"]
        pedestrian_start = selected_locations["pedestrian_start"]
        pedestrian_end = selected_locations["pedestrian_end"]
        vehicle_waypoints = [
            self.road_locations[int(index) % len(self.road_locations)]
            for index in vehicle_waypoint_indices
        ] if self.road_locations else []
        pedestrian_waypoints = [
            self.pedestrian_points[int(index) % len(self.pedestrian_points)]
            for index in pedestrian_waypoint_indices
        ] if self.pedestrian_points else []
        if len(vehicle_route_preview) >= 2:
            preview_points = [
                self.world_to_screen(location) for location in vehicle_route_preview
            ]
            pygame.draw.lines(screen, v1.COLOR_BG, False, preview_points, 6)
            pygame.draw.lines(
                screen, COLOR_ROUTE_VEHICLE, False, preview_points, 3
            )
        elif vehicle_start is not None and vehicle_end is not None:
            pygame.draw.lines(
                screen,
                COLOR_ROUTE_VEHICLE,
                False,
                [
                    self.world_to_screen(location)
                    for location in [vehicle_start] + vehicle_waypoints + [vehicle_end]
                ],
                2,
            )
        if pedestrian_start is not None and pedestrian_end is not None:
            pygame.draw.lines(
                screen,
                COLOR_ROUTE_PEDESTRIAN,
                False,
                [
                    self.world_to_screen(location)
                    for location in [pedestrian_start]
                    + pedestrian_waypoints
                    + [pedestrian_end]
                ],
                2,
            )
        for order, location in enumerate(vehicle_waypoints, start=1):
            self._draw_route_waypoint(
                screen, location, "V{}".format(order), COLOR_ROUTE_VEHICLE
            )
        for order, location in enumerate(pedestrian_waypoints, start=1):
            self._draw_route_waypoint(
                screen, location, "P{}".format(order), COLOR_ROUTE_PEDESTRIAN
            )
        self._draw_endpoint(screen, vehicle_start, "V-START", COLOR_START)
        self._draw_endpoint(screen, vehicle_end, "V-END", COLOR_END)
        self._draw_endpoint(screen, pedestrian_start, "P-START", COLOR_START)
        self._draw_endpoint(screen, pedestrian_end, "P-END", COLOR_END)

        mouse_position = pygame.mouse.get_pos()
        self.hover = (
            self._nearest_point(mouse_position, mode, 14.0)
            if self.plot_rect.collidepoint(mouse_position)
            else None
        )
        if self.hover is not None:
            hover_index, hover_location = self.hover
            position = self.world_to_screen(hover_location)
            pygame.draw.circle(screen, v1.COLOR_TEXT, position, 8, 2)
            hover_text = small_font.render(
                "#{} {}".format(hover_index, format_location(hover_location)),
                True,
                v1.COLOR_TEXT,
                v1.COLOR_PANEL,
            )
            tooltip_x = min(
                position[0] + 12,
                self.plot_rect.right - hover_text.get_width() - 5,
            )
            tooltip_y = max(self.plot_rect.top + 4, position[1] - 24)
            screen.blit(hover_text, (tooltip_x, tooltip_y))
        screen.set_clip(old_clip)
        pygame.draw.rect(screen, v1.COLOR_BORDER, self.plot_rect, 1)

        # CARLA uses x/y world axes. Match the reference map's +Y-down view.
        origin = (self.plot_rect.right - self._sy(95), self.plot_rect.top + self._sy(35))
        arm = self._sy(30)
        pygame.draw.line(screen, COLOR_START, origin, (origin[0] + arm, origin[1]), 2)
        pygame.draw.line(screen, COLOR_END, origin, (origin[0], origin[1] + arm), 2)
        screen.blit(
            small_font.render("+X", True, COLOR_START),
            (origin[0] + arm + self._sy(3), origin[1] - self._sy(8)),
        )
        screen.blit(
            small_font.render("+Y", True, COLOR_END),
            (origin[0] - self._sy(9), origin[1] + arm + self._sy(4)),
        )


class ScenarioUIV2(v1.ScenarioUI):
    """V2 layout with coordinate readouts and embedded map interaction."""

    def __init__(
        self,
        controller: ScenarioControllerV2,
        base_config: Dict[str, Any],
        route_config_path: Path,
    ) -> None:
        self.route_config_path = Path(route_config_path).resolve()
        self.route_status = "Route file: {}".format(self.route_config_path)
        self.vehicle_route_preview: List[carla.Location] = []
        self.loaded_route_config: Optional[Dict[str, Any]] = None
        self.vehicle_points = list(controller.vehicle_spawn_preview)
        self.pedestrian_points: List[carla.Location] = []
        self.preview_error = ""
        try:
            self.pedestrian_points = controller.navigation_preview(
                int(base_config["scenario"]["seed"])
            )
        except Exception as exc:
            LOG.exception("Unable to build pedestrian map preview")
            self.preview_error = "Pedestrian preview unavailable: {}".format(exc)
        vehicle_config = base_config["scenario"]["ego_vehicle"]
        pedestrian_config = base_config["scenario"]["ego_pedestrian"]
        self.vehicle_waypoint_indices = self._normalize_ui_indices(
            vehicle_config.get("route_waypoint_indices", []),
            len(controller.road_preview),
        )
        self.pedestrian_waypoint_indices = self._normalize_ui_indices(
            pedestrian_config.get("route_waypoint_indices", []),
            len(self.pedestrian_points),
        )
        self.map_mode = "vehicle_start"
        ui_config = base_config.get("ui", {})
        # Resolve geometry before the base __init__ runs: it calls set_mode()
        # and _build_controls(), both of which need the final size and scale.
        window_width, window_height, self._window_flags, self.ui_scale, self.window_mode = (
            resolve_window_geometry(ui_config)
        )
        base_config = copy.deepcopy(base_config)
        base_config["ui"]["window_size"] = [window_width, window_height]
        # v1 lays the video and bottom bar out from these module constants, so
        # scaling them here scales every inherited drawing path too.
        v1.PANEL_WIDTH = self._scaled(DESIGN_PANEL_WIDTH)
        v1.BOTTOM_HEIGHT = self._scaled(DESIGN_BOTTOM_HEIGHT)
        self.hand_over_control = bool(
            ui_config.get("control_switch_takes_manual", True)
        )
        self.redraw_hz = max(1.0, float(ui_config.get("redraw_hz", 10.0)))
        self.vehicle_speed_range = self._speed_range(
            ui_config, "vehicle_speed_range_kmh", (5.0, 90.0, 2.0)
        )
        self.ped_speed_range = self._speed_range(
            ui_config, "ped_speed_range_mps", (0.5, 20.0, 0.5)
        )
        requested_video_size = ui_config.get(
            "video_window_size", list(DEFAULT_VIDEO_WINDOW_SIZE)
        )
        try:
            self.video_window_size = (
                int(requested_video_size[0]),
                int(requested_video_size[1]),
            )
        except (TypeError, ValueError, IndexError):
            LOG.warning(
                "Unsupported ui.video_window_size %r; using %dx%d",
                requested_video_size,
                *DEFAULT_VIDEO_WINDOW_SIZE,
            )
            self.video_window_size = DEFAULT_VIDEO_WINDOW_SIZE
        self.video_window_positions = self._resolve_video_positions(
            ui_config, self.video_window_size
        )
        # Video feeds live in their own desktop windows; only the last annotated
        # frame per stream is retained so a window keeps its picture between
        # sensor ticks.
        self.video_windows: Optional[VideoWindowBank] = None
        self.last_frames: Dict[str, Any] = {}
        self.frame_numbers: Dict[str, int] = {}
        super().__init__(controller, base_config)
        if self._window_flags:
            # The base class opens a plain window; re-open it with the
            # requested borderless/fullscreen flags at the same size.
            self.screen = pygame.display.set_mode(
                (self.width, self.height), self._window_flags
            )
        self.font = self._scaled_font(DESIGN_FONT_SIZE)
        self.small_font = self._scaled_font(DESIGN_SMALL_FONT_SIZE)
        self.title_font = self._scaled_font(DESIGN_TITLE_FONT_SIZE)
        self.error_message = self.preview_error
        video_rect = pygame.Rect(
            v1.PANEL_WIDTH,
            0,
            self.width - v1.PANEL_WIDTH,
            self.height - v1.BOTTOM_HEIGHT,
        )
        self.map_selector = TopDownMapSelector(
            video_rect,
            controller.road_preview,
            controller.road_polylines_preview,
            controller.building_footprints_preview,
            self.vehicle_points,
            self.pedestrian_points,
            float(base_config["ui"].get("map_max_zoom", 8.0)),
            self.ui_scale,
        )
        self._update_map_mode_labels()
        self._update_target_button()
        if self.route_config_path.is_file():
            self._guard(self._load_vehicle_route)
        pygame.display.set_caption(
            "CARLA Physical AI Scenario Controller v2"
        )

    @staticmethod
    def _speed_range(
        ui_config: Dict[str, Any],
        key: str,
        default: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Read a [minimum, maximum, step] speed-stepper range from the config."""
        requested = ui_config.get(key, list(default))
        try:
            minimum, maximum, step = (float(value) for value in requested)
        except (TypeError, ValueError):
            LOG.warning("Unsupported ui.%s %r; using %s", key, requested, default)
            return default
        if not 0.0 < minimum < maximum or step <= 0.0:
            LOG.warning(
                "ui.%s must be an increasing positive [min, max, step]; got %r",
                key,
                requested,
            )
            return default
        return minimum, maximum, step

    def _resolve_video_positions(
        self,
        ui_config: Dict[str, Any],
        video_size: Tuple[int, int],
    ) -> Dict[str, Tuple[int, int]]:
        """
        Where each video window first appears.

        Two 1080p windows cannot both fit un-overlapped on one laptop panel, so
        they are pushed to the right edge and cascaded downward: that keeps the
        scenario sidebar -- speeds, autonomy switches, Start/Stop -- uncovered,
        and each title bar stays grabbable. Drag them anywhere, or set explicit
        ui.video_window_positions to send one to a second display.
        """
        area_x, area_y, area_width, area_height = desktop_work_area()
        screen_width, _ = desktop_size()
        configured = ui_config.get("video_window_positions", {}) or {}
        step_x = max(0, int(ui_config.get("video_window_cascade_x", 0)))
        step_y = max(0, int(ui_config.get("video_window_cascade_y", 300)))
        # Right-align so the sidebar stays visible, without running off screen.
        base_x = max(area_x, min(screen_width - video_size[0], area_x + area_width - 200))
        defaults = {
            EGO_VEHICLE_VIEW_KEY: (base_x, area_y),
            EGO_PEDESTRIAN_VIEW_KEY: (base_x + step_x, area_y + step_y),
            "pole": (base_x + 2 * step_x, area_y + 2 * step_y),
        }
        positions: Dict[str, Tuple[int, int]] = {}
        for key, default in defaults.items():
            value = configured.get(key, default)
            try:
                positions[key] = (int(value[0]), int(value[1]))
            except (TypeError, ValueError, IndexError):
                LOG.warning(
                    "Unsupported ui.video_window_positions[%r]=%r; using %s",
                    key,
                    value,
                    default,
                )
                positions[key] = default
        # Keep the first window inside the work area even on a small desktop.
        positions[EGO_VEHICLE_VIEW_KEY] = (
            max(0, min(positions[EGO_VEHICLE_VIEW_KEY][0], area_x + area_width - 200)),
            max(0, min(positions[EGO_VEHICLE_VIEW_KEY][1], area_y + area_height - 120)),
        )
        return positions

    def _scaled(self, value: float) -> int:
        """Scale a design-space pixel value for the current window."""
        return int(round(float(value) * self.ui_scale))

    def _scaled_font(self, design_size: int) -> pygame.font.Font:
        return pygame.font.Font(
            pygame.font.get_default_font(),
            max(10, self._scaled(design_size)),
        )

    @staticmethod
    def _ellipsize_middle(
        text: str,
        font: pygame.font.Font,
        maximum_width: int,
    ) -> str:
        """Fit a path-like status while retaining both action and filename."""
        if font.size(text)[0] <= maximum_width:
            return text
        low, high = 0, len(text)
        best = "..."
        while low <= high:
            keep = (low + high) // 2
            left_count = (keep + 1) // 2
            right_count = keep // 2
            candidate = "{}...{}".format(
                text[:left_count], text[len(text) - right_count :]
                if right_count
                else "",
            )
            if font.size(candidate)[0] <= maximum_width:
                best = candidate
                low = keep + 1
            else:
                high = keep - 1
        return best

    @staticmethod
    def _normalize_ui_indices(values: Any, point_count: int) -> List[int]:
        if not isinstance(values, (list, tuple)) or point_count <= 0:
            return []
        result = []
        for value in values:
            try:
                index = int(value) % point_count
            except (TypeError, ValueError):
                continue
            if not result or result[-1] != index:
                result.append(index)
        return result

    # Design-space panel rows (see DESIGN_PANEL_HEIGHT). Keeping them in one
    # table makes the vertical rhythm checkable and keeps _draw_panel in step.
    PANEL_ROWS = {
        "title": 14,
        "subtitle": 44,
        "head_actors": 64,
        "seed": 82,
        "npc_vehicles": 116,
        "npc_pedestrians": 150,
        "head_vehicle": 190,
        "vehicle_start": 210,
        "vehicle_start_coord": 242,
        "vehicle_end": 262,
        "vehicle_end_coord": 294,
        "vehicle_scripted": 314,
        "vehicle_speed": 348,
        "head_pedestrian": 388,
        "ped_start": 408,
        "ped_start_coord": 440,
        "ped_end": 460,
        "ped_end_coord": 492,
        "ped_scripted": 512,
        "ped_speed": 546,
        "head_occlusion": 586,
        "occluder": 606,
        "occluder_fraction": 640,
        "occluder_lateral": 674,
        "ground_truth": 708,
        "start": 752,
        "stop": 796,
        "replay": 838,
        "status_label": 880,
        "status_text": 898,
        "hint_1": 968,
        "hint_2": 984,
    }

    def _row(self, name: str) -> int:
        return self._scaled(self.PANEL_ROWS[name])

    def _panel_rect(self, x: float, y: float, width: float, height: float) -> pygame.Rect:
        return pygame.Rect(
            self._scaled(x), self._scaled(y), self._scaled(width), self._scaled(height)
        )

    def _make_stepper(
        self,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        row: str,
        integer: bool = True,
        value_width: float = 52,
    ) -> ScaledStepper:
        """
        Build a stepper whose rects are right-aligned inside the scaled panel.

        The +/- and value rects are laid out from the panel's right edge so a
        wide value (an eight-digit seed, a speed in km/h) simply pushes the minus
        button left instead of overflowing the panel.
        """
        y = self.PANEL_ROWS[row]
        stepper = ScaledStepper(label, value, minimum, maximum, step, 0, integer)
        plus_x = DESIGN_PANEL_WIDTH - 17 - 36
        value_x = plus_x - 1 - value_width
        minus_x = value_x - 1 - 36
        stepper.minus = self._panel_rect(minus_x, y, 36, 28)
        stepper.value_rect = self._panel_rect(value_x, y, value_width, 28)
        stepper.plus = self._panel_rect(plus_x, y, 36, 28)
        stepper.label_x = self._scaled(18)
        return stepper

    def _make_toggle(self, label: str, value: bool, row: str) -> ScaledToggle:
        toggle = ScaledToggle(label, value, 0)
        toggle.rect = self._panel_rect(
            DESIGN_PANEL_WIDTH - 17 - 85, self.PANEL_ROWS[row], 85, 28
        )
        toggle.label_x = self._scaled(18)
        return toggle

    def _build_controls(self) -> None:
        scenario = self.base_config["scenario"]
        vehicle_max = max(0, len(self.vehicle_points) - 1)
        pedestrian_max = max(0, len(self.pedestrian_points) - 1)
        vehicle = scenario["ego_vehicle"]
        pedestrian = scenario["ego_pedestrian"]
        occluder = scenario["occluder"]
        # Preserve the configured run/walk and curve/cruise relationships so the
        # single speed control per actor scales the whole speed profile.
        walk_speed = max(0.1, float(pedestrian.get("walk_speed_mps", 2.5)))
        self.pedestrian_run_ratio = max(
            1.0, float(pedestrian.get("run_speed_mps", 5.0)) / walk_speed
        )
        cruise_speed = max(1.0, float(vehicle.get("scripted_speed_kmh", 18.0)))
        self.vehicle_curve_ratio = v1.clamp(
            float(vehicle.get("scripted_curve_speed_kmh", 10.0)) / cruise_speed,
            0.1,
            1.0,
        )

        self.seed = self._make_stepper(
            "Random seed", scenario["seed"], 0, 99999999, 1, "seed",
            value_width=84,
        )
        self.npc_vehicles = self._make_stepper(
            "NPC vehicles", scenario["npc_vehicles"], 0, 100, 1, "npc_vehicles"
        )
        self.npc_pedestrians = self._make_stepper(
            "NPC pedestrians", scenario["npc_pedestrians"], 0, 200, 1,
            "npc_pedestrians",
        )
        self.vehicle_start = self._make_stepper(
            "Vehicle start index", vehicle["start_spawn_index"], 0, vehicle_max,
            1, "vehicle_start",
        )
        self.vehicle_end = self._make_stepper(
            "Vehicle end index", vehicle["end_spawn_index"], 0, vehicle_max, 1,
            "vehicle_end",
        )
        self.vehicle_scripted = self._make_toggle(
            "Vehicle autonomous", vehicle["scripted_route"], "vehicle_scripted"
        )
        vehicle_minimum, vehicle_maximum, vehicle_step = self.vehicle_speed_range
        self.vehicle_speed = self._make_stepper(
            "Vehicle speed (km/h)", cruise_speed, vehicle_minimum, vehicle_maximum,
            vehicle_step, "vehicle_speed", False, value_width=66,
        )
        self.ped_start = self._make_stepper(
            "Pedestrian start index", pedestrian["start_spawn_index"], 0,
            pedestrian_max, 1, "ped_start",
        )
        self.ped_end = self._make_stepper(
            "Pedestrian end index", pedestrian["end_spawn_index"], 0,
            pedestrian_max, 1, "ped_end",
        )
        self.ped_scripted = self._make_toggle(
            "Ped autonomous", pedestrian["scripted_route"], "ped_scripted"
        )
        ped_minimum, ped_maximum, ped_step = self.ped_speed_range
        self.ped_speed = self._make_stepper(
            "Ped speed (m/s)", walk_speed, ped_minimum, ped_maximum, ped_step,
            "ped_speed", False, value_width=66,
        )
        # Indices depend on the loaded map and the speeds on the configured
        # ranges, so every stepper starts clamped to its own bounds.
        for stepper in (
            self.vehicle_start,
            self.vehicle_end,
            self.ped_start,
            self.ped_end,
            self.vehicle_speed,
            self.ped_speed,
        ):
            stepper.value = v1.clamp(
                stepper.value, stepper.minimum, stepper.maximum
            )
        self.occluder = ScaledCycleField(
            "Occluder type", occluder["type"], ["none", "bus", "truck"], 0
        )
        self.occluder.rect = self._panel_rect(
            247, self.PANEL_ROWS["occluder"], DESIGN_PANEL_WIDTH - 17 - 247, 28
        )
        self.occluder.label_x = self._scaled(18)
        self.occluder_fraction = self._make_stepper(
            "Route fraction", occluder["route_fraction"], 0.0, 1.0, 0.05,
            "occluder_fraction", False,
        )
        self.occluder_lateral = self._make_stepper(
            "Lateral offset (m)", occluder["lateral_offset_m"], -10.0, 10.0, 0.5,
            "occluder_lateral", False,
        )
        self.ground_truth = self._make_toggle(
            "GT boxes + LOS estimate",
            scenario["ground_truth"]["enabled"],
            "ground_truth",
        )
        self.config_controls = [
            self.seed,
            self.npc_vehicles,
            self.npc_pedestrians,
            self.vehicle_start,
            self.vehicle_end,
            self.vehicle_scripted,
            self.vehicle_speed,
            self.ped_start,
            self.ped_end,
            self.ped_scripted,
            self.ped_speed,
            self.occluder,
            self.occluder_fraction,
            self.occluder_lateral,
            self.ground_truth,
        ]
        button_width = DESIGN_PANEL_WIDTH - 35
        half_width = (button_width - 15) / 2.0
        self.start_button = v1.Button(
            "Start current config",
            self._panel_rect(18, self.PANEL_ROWS["start"], button_width, 36),
            self._start,
        )
        self.stop_button = v1.Button(
            "Stop demo + delete owned actors",
            self._panel_rect(18, self.PANEL_ROWS["stop"], button_width, 34),
            self._stop_demo,
        )
        self.replay_button = v1.Button(
            "Replay last launch",
            self._panel_rect(18, self.PANEL_ROWS["replay"], half_width, 34),
            self._replay,
        )
        self.target_button = v1.Button(
            "WASD: VEH",
            self._panel_rect(
                18 + half_width + 15, self.PANEL_ROWS["replay"], half_width, 34
            ),
            self._switch_manual_target,
        )

        # --- bottom bar: map selection on the left, camera aim on the right ---
        bottom_y = self.height - v1.BOTTOM_HEIGHT
        map_button_width = self._scaled(BOTTOM_BUTTON_WIDTH)
        map_left = v1.PANEL_WIDTH + self._scaled(25)
        gap = self._scaled(8)
        self.map_mode_buttons: Dict[str, v1.Button] = {}
        for offset, key in enumerate(MAP_MODE_LABELS):
            row = offset // 3
            column = offset % 3
            self.map_mode_buttons[key] = v1.Button(
                MAP_MODE_BUTTON_LABELS[key],
                pygame.Rect(
                    map_left + column * (map_button_width + gap),
                    bottom_y + self._scaled(BOTTOM_ROWS["buttons_1"] + row * 42),
                    map_button_width,
                    self._scaled(34),
                ),
                lambda selected=key: self._set_map_mode(selected),
            )
        action_y = bottom_y + self._scaled(BOTTOM_ROWS["buttons_3"])
        action_gap = self._scaled(5)
        action_row_width = 3 * map_button_width + 2 * gap
        action_width = (action_row_width - 4 * action_gap) // 5
        self.reset_map_button = v1.Button(
            "Reset",
            pygame.Rect(map_left, action_y, action_width, self._scaled(34)),
            self._reset_map_view,
        )
        self.undo_waypoint_button = v1.Button(
            "Undo",
            pygame.Rect(
                map_left + action_width + action_gap, action_y, action_width,
                self._scaled(34),
            ),
            self._undo_active_waypoint,
        )
        self.clear_waypoints_button = v1.Button(
            "Clear",
            pygame.Rect(
                map_left + 2 * (action_width + action_gap), action_y, action_width,
                self._scaled(34),
            ),
            self._clear_active_waypoints,
        )
        self.load_route_button = v1.Button(
            "Load route",
            pygame.Rect(
                map_left + 3 * (action_width + action_gap), action_y, action_width,
                self._scaled(34),
            ),
            lambda: self._guard(self._load_vehicle_route),
        )
        self.save_route_button = v1.Button(
            "Save route",
            pygame.Rect(
                map_left + 4 * (action_width + action_gap), action_y, action_width,
                self._scaled(34),
            ),
            lambda: self._guard(self._save_vehicle_route),
        )

        self.camera_column_x = map_left + self._scaled(
            3 * BOTTOM_BUTTON_WIDTH + 2 * 8 + 40
        )
        view_width = self._scaled(150)
        camera_row = bottom_y + self._scaled(BOTTOM_ROWS["buttons_1"])
        self.prev_view = v1.Button(
            "< Previous",
            pygame.Rect(
                self.camera_column_x, camera_row, view_width, self._scaled(34)
            ),
            lambda: self._cycle_view(-1),
        )
        self.next_view = v1.Button(
            "Next camera >",
            pygame.Rect(
                self.camera_column_x + view_width + gap, camera_row, view_width,
                self._scaled(34),
            ),
            lambda: self._cycle_view(1),
        )
        self.reset_view = v1.Button(
            "Recentre aim",
            pygame.Rect(
                self.camera_column_x + 2 * (view_width + gap), camera_row,
                view_width, self._scaled(34),
            ),
            self._reset_view,
        )
        slider_width = max(
            self._scaled(240),
            self.width - self.camera_column_x - self._scaled(30),
        )
        self.yaw_slider = ScaledSlider(
            "Camera yaw",
            pygame.Rect(
                self.camera_column_x,
                bottom_y + self._scaled(BOTTOM_ROWS["yaw_track"]),
                slider_width,
                self._scaled(10),
            ),
            -180.0,
            180.0,
            0.0,
        )
        self.pitch_slider = ScaledSlider(
            "Camera pitch",
            pygame.Rect(
                self.camera_column_x,
                bottom_y + self._scaled(BOTTOM_ROWS["pitch_track"]),
                slider_width,
                self._scaled(10),
            ),
            -90.0,
            45.0,
            0.0,
        )
        for slider in (self.yaw_slider, self.pitch_slider):
            slider.label_gap = self._scaled(26)
            slider.track_width = max(4, self._scaled(6))
            slider.knob_radius = max(7, self._scaled(9))
        self.camera_buttons = [self.prev_view, self.next_view, self.reset_view]
        self.common_buttons = [
            self.start_button,
            self.stop_button,
            self.replay_button,
            self.target_button,
        ]

    @property
    def typed_controller(self) -> ScenarioControllerV2:
        return self.controller

    def _location_at(
        self,
        points: Sequence[Any],
        index: int,
    ) -> Optional[carla.Location]:
        if not points:
            return None
        return point_location(points[int(index) % len(points)])

    def _selected_locations(self) -> Dict[str, Optional[carla.Location]]:
        return {
            "vehicle_start": self._location_at(
                self.vehicle_points, self.vehicle_start.get()
            ),
            "vehicle_end": self._location_at(
                self.vehicle_points, self.vehicle_end.get()
            ),
            "pedestrian_start": self._location_at(
                self.pedestrian_points, self.ped_start.get()
            ),
            "pedestrian_end": self._location_at(
                self.pedestrian_points, self.ped_end.get()
            ),
        }

    def _selection_indices(self) -> Dict[str, int]:
        return {
            "vehicle_start": self.vehicle_start.get(),
            "vehicle_end": self.vehicle_end.get(),
            "pedestrian_start": self.ped_start.get(),
            "pedestrian_end": self.ped_end.get(),
        }

    @staticmethod
    def _location_payload(location: carla.Location) -> Dict[str, float]:
        return {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }

    @classmethod
    def _transform_payload(cls, transform: carla.Transform) -> Dict[str, Any]:
        return {
            "location": cls._location_payload(transform.location),
            "rotation": {
                "pitch": float(transform.rotation.pitch),
                "yaw": float(transform.rotation.yaw),
                "roll": float(transform.rotation.roll),
            },
        }

    @staticmethod
    def _payload_location(payload: Dict[str, Any]) -> carla.Location:
        return carla.Location(
            x=float(payload["x"]),
            y=float(payload["y"]),
            z=float(payload["z"]),
        )

    @classmethod
    def _payload_transform(cls, payload: Dict[str, Any]) -> carla.Transform:
        location = cls._payload_location(payload["location"])
        rotation = payload["rotation"]
        return carla.Transform(
            location,
            carla.Rotation(
                pitch=float(rotation["pitch"]),
                yaw=float(rotation["yaw"]),
                roll=float(rotation["roll"]),
            ),
        )

    @staticmethod
    def _nearest_catalog_index(
        location: carla.Location,
        points: Sequence[Any],
        index_hint: Any = None,
    ) -> Tuple[int, float]:
        """Map a saved coordinate to a catalog, treating indices as hints only."""
        if not points:
            raise ValueError("the loaded CARLA map has no matching route catalog")
        distances = [
            float(point_location(point).distance(location)) for point in points
        ]
        nearest_index = min(range(len(distances)), key=distances.__getitem__)
        nearest_distance = distances[nearest_index]
        # Preserve an exact/duplicate catalog identity when its saved index is
        # still valid. Never modulo a stale index or prefer it over a closer
        # coordinate, because catalogs can change between CARLA builds.
        if not isinstance(index_hint, bool):
            try:
                hinted_index = int(index_hint)
            except (TypeError, ValueError):
                hinted_index = -1
            if (
                0 <= hinted_index < len(points)
                and distances[hinted_index] <= nearest_distance + 1e-3
            ):
                nearest_index = hinted_index
                nearest_distance = distances[hinted_index]
        return nearest_index, nearest_distance

    def _route_sampling_resolution(self) -> float:
        return float(
            self.base_config["scenario"]["ego_vehicle"].get(
                "route_sampling_resolution_m", 2.0
            )
        )

    def _invalidate_vehicle_route_preview(self) -> None:
        self.vehicle_route_preview = []
        self.loaded_route_config = None
        self.route_status = "Vehicle route changed; Save route to rebuild preview"

    def _save_vehicle_route(self) -> None:
        """Plan current vehicle controls and atomically write the shared JSON."""
        start_index = int(self.vehicle_start.get())
        end_index = int(self.vehicle_end.get())
        if not 0 <= start_index < len(self.vehicle_points):
            raise ValueError("vehicle start selection is unavailable")
        if not 0 <= end_index < len(self.vehicle_points):
            raise ValueError("vehicle end selection is unavailable")
        if not self.typed_controller.road_preview:
            raise ValueError("loaded CARLA map has no selectable road waypoints")
        invalid_via = [
            index
            for index in self.vehicle_waypoint_indices
            if not 0 <= int(index) < len(self.typed_controller.road_preview)
        ]
        if invalid_via:
            raise ValueError(
                "vehicle route contains stale waypoint indices: {}".format(
                    invalid_via
                )
            )

        start_transform = v1.copy_transform(self.vehicle_points[start_index])
        end_transform = v1.copy_transform(self.vehicle_points[end_index])
        selected_vias = [
            v1.copy_location(self.typed_controller.road_preview[int(index)])
            for index in self.vehicle_waypoint_indices
        ]
        resolution = self._route_sampling_resolution()
        planned = self.typed_controller.plan_vehicle_route_for_export(
            start_transform,
            selected_vias,
            end_transform,
            resolution,
        )
        map_name = str(self.typed_controller.map.name)
        route_data = {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "type": ROUTE_CONFIG_TYPE,
            "name": "{} ego vehicle route".format(Path(map_name).name),
            "map": map_name,
            "coordinate_system": ROUTE_COORDINATE_SYSTEM,
            "route_sampling_resolution_m": resolution,
            # The selected spawn transform is deliberately authoritative. It
            # retains its exact coordinates and heading rather than replacing
            # it with the planner's first lane sample.
            "start": self._transform_payload(planned["start_transform"]),
            "intermediate_waypoints": [
                self._location_payload(location)
                for location in planned["intermediate_waypoints"]
            ],
            "end": self._transform_payload(planned["end_transform"]),
            "planned_path": [
                self._location_payload(location)
                for location in planned["planned_path"]
            ],
            "ui_selection": {
                "producer": Path(__file__).name,
                "selection_basis": (
                    "coordinates_authoritative_catalog_indices_are_hints"
                ),
                "vehicle_start_spawn_index": start_index,
                "vehicle_end_spawn_index": end_index,
                "vehicle_waypoint_indices": [
                    int(index) for index in self.vehicle_waypoint_indices
                ],
                "vehicle_spawn_catalog_size": len(self.vehicle_points),
                "road_waypoint_catalog_size": len(
                    self.typed_controller.road_preview
                ),
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        normalized = save_route_config(self.route_config_path, route_data)
        self.loaded_route_config = normalized
        self.vehicle_route_preview = [
            v1.copy_location(location) for location in planned["planned_path"]
        ]
        self.route_status = "Saved {} road points to {}".format(
            len(self.vehicle_route_preview), self.route_config_path
        )
        self.controller.status = self.route_status

    def _load_vehicle_route(self) -> None:
        """Load a shared route and restore UI selections by coordinates."""
        route = load_route_config(self.route_config_path)
        current_map = str(self.typed_controller.map.name)
        if not maps_match(route["map"], current_map):
            raise ValueError(
                "route map {!r} does not match loaded CARLA map {!r}".format(
                    route["map"], current_map
                )
            )
        hints = route.get("ui_selection", {})
        if not isinstance(hints, dict):
            hints = {}
        start_transform = self._payload_transform(route["start"])
        end_transform = self._payload_transform(route["end"])
        start_index, start_error = self._nearest_catalog_index(
            start_transform.location,
            self.vehicle_points,
            hints.get("vehicle_start_spawn_index"),
        )
        end_index, end_error = self._nearest_catalog_index(
            end_transform.location,
            self.vehicle_points,
            hints.get("vehicle_end_spawn_index"),
        )
        via_hints = hints.get("vehicle_waypoint_indices", [])
        if not isinstance(via_hints, list):
            via_hints = []
        waypoint_indices: List[int] = []
        mapping_errors = [start_error, end_error]
        for order, waypoint_payload in enumerate(
            route["intermediate_waypoints"]
        ):
            waypoint = self._payload_location(waypoint_payload)
            index_hint = via_hints[order] if order < len(via_hints) else None
            index, mapping_error = self._nearest_catalog_index(
                waypoint,
                self.typed_controller.road_preview,
                index_hint,
            )
            mapping_errors.append(mapping_error)
            if not waypoint_indices or waypoint_indices[-1] != index:
                waypoint_indices.append(index)

        largest_mapping_error = max(mapping_errors) if mapping_errors else 0.0
        if largest_mapping_error > 12.0:
            raise ValueError(
                "saved route controls are up to {:.1f} m from this map's "
                "selection catalogs; refusing to apply ambiguous indices".format(
                    largest_mapping_error
                )
            )

        self.vehicle_start.value = float(start_index)
        self.vehicle_end.value = float(end_index)
        self.vehicle_waypoint_indices = waypoint_indices
        self.base_config["scenario"]["ego_vehicle"][
            "route_sampling_resolution_m"
        ] = float(route["route_sampling_resolution_m"])

        preview_payload = route.get("planned_path", [])
        preview = [
            self._payload_location(location) for location in preview_payload
        ]
        if len(preview) >= 2 and (
            preview[0].distance(start_transform.location) > 12.0
            or preview[-1].distance(end_transform.location) > 12.0
        ):
            LOG.warning(
                "Stored planned_path endpoints do not match its route controls; "
                "replanning the top-down preview"
            )
            preview = []
        if len(preview) >= 2:
            search_index = 0
            for waypoint_payload in route["intermediate_waypoints"]:
                control = self._payload_location(waypoint_payload)
                remaining = preview[search_index:]
                if not remaining:
                    preview = []
                    break
                relative_index = min(
                    range(len(remaining)),
                    key=lambda index: remaining[index].distance(control),
                )
                if remaining[relative_index].distance(control) > 5.0:
                    LOG.warning(
                        "Stored planned_path misses an intermediate route "
                        "control; replanning the top-down preview"
                    )
                    preview = []
                    break
                search_index += relative_index
        if len(preview) < 2:
            planned = self.typed_controller.plan_vehicle_route_for_export(
                start_transform,
                [
                    self._payload_location(location)
                    for location in route["intermediate_waypoints"]
                ],
                end_transform,
                float(route["route_sampling_resolution_m"]),
            )
            preview = list(planned["planned_path"])
        self.loaded_route_config = route
        self.vehicle_route_preview = [
            v1.copy_location(location) for location in preview
        ]
        if largest_mapping_error > 5.0:
            LOG.warning(
                "Loaded route catalog mapping is as far as %.1f m from a "
                "saved control coordinate",
                largest_mapping_error,
            )
        self.route_status = "Loaded {} road points from {}".format(
            len(self.vehicle_route_preview), self.route_config_path
        )
        self.controller.status = self.route_status
        self._update_map_mode_labels()

    def _active_waypoint_indices(self) -> Optional[List[int]]:
        if self.map_mode == "vehicle_waypoints":
            return self.vehicle_waypoint_indices
        if self.map_mode == "pedestrian_waypoints":
            return self.pedestrian_waypoint_indices
        return None

    def _set_map_mode(self, mode: str) -> None:
        self.map_mode = mode
        self._update_map_mode_labels()

    def _update_map_mode_labels(self) -> None:
        if not hasattr(self, "map_mode_buttons"):
            return
        for key, button in self.map_mode_buttons.items():
            button.label = (
                "[{}]".format(MAP_MODE_BUTTON_LABELS[key])
                if key == self.map_mode
                else MAP_MODE_BUTTON_LABELS[key]
            )
        active_waypoints = self._active_waypoint_indices()
        waypoint_actions_enabled = active_waypoints is not None
        self.undo_waypoint_button.enabled = bool(
            waypoint_actions_enabled and active_waypoints
        )
        self.clear_waypoints_button.enabled = bool(
            waypoint_actions_enabled and active_waypoints
        )

    def _undo_active_waypoint(self) -> None:
        active_waypoints = self._active_waypoint_indices()
        if active_waypoints:
            active_waypoints.pop()
            if self.map_mode == "vehicle_waypoints":
                self._invalidate_vehicle_route_preview()
        self._update_map_mode_labels()

    def _clear_active_waypoints(self) -> None:
        active_waypoints = self._active_waypoint_indices()
        if active_waypoints:
            active_waypoints.clear()
            if self.map_mode == "vehicle_waypoints":
                self._invalidate_vehicle_route_preview()
        self._update_map_mode_labels()

    def _reset_map_view(self) -> None:
        if hasattr(self, "map_selector"):
            self.map_selector.reset_view()

    def _cycle_view(self, offset: int) -> None:
        """Move the camera-aim focus, opening/closing the pole window to match."""
        super()._cycle_view(offset)
        self._sync_pole_stream()

    def _apply_map_selection(self, index: int) -> None:
        if self.map_mode == "vehicle_waypoints":
            if not self.vehicle_waypoint_indices or self.vehicle_waypoint_indices[-1] != index:
                self.vehicle_waypoint_indices.append(int(index))
                self._invalidate_vehicle_route_preview()
            self._update_map_mode_labels()
            return
        if self.map_mode == "pedestrian_waypoints":
            if not self.pedestrian_waypoint_indices or self.pedestrian_waypoint_indices[-1] != index:
                self.pedestrian_waypoint_indices.append(int(index))
            self._update_map_mode_labels()
            return
        target = {
            "vehicle_start": self.vehicle_start,
            "vehicle_end": self.vehicle_end,
            "pedestrian_start": self.ped_start,
            "pedestrian_end": self.ped_end,
        }[self.map_mode]
        selected_index = v1.clamp(index, target.minimum, target.maximum)
        if target.value != selected_index:
            target.value = selected_index
            if self.map_mode in ("vehicle_start", "vehicle_end"):
                self._invalidate_vehicle_route_preview()

    def _current_config(self) -> Dict[str, Any]:
        config = super()._current_config()
        scenario = config["scenario"]
        scenario["ego_vehicle"]["route_waypoint_indices"] = list(
            self.vehicle_waypoint_indices
        )
        scenario["ego_pedestrian"]["route_waypoint_indices"] = list(
            self.pedestrian_waypoint_indices
        )
        # One speed control per actor drives the whole speed profile, keeping
        # the configured curve/cruise and run/walk relationships.
        cruise = float(self.vehicle_speed.get())
        scenario["ego_vehicle"]["scripted_speed_kmh"] = cruise
        scenario["ego_vehicle"]["scripted_curve_speed_kmh"] = max(
            1.0, cruise * self.vehicle_curve_ratio
        )
        walk = float(self.ped_speed.get())
        scenario["ego_pedestrian"]["walk_speed_mps"] = walk
        scenario["ego_pedestrian"]["run_speed_mps"] = walk * self.pedestrian_run_ratio
        scenario["ego_pedestrian"]["scripted_speed_mps"] = walk
        return config

    def _start(self) -> None:
        self._close_video_windows()
        super()._start()
        if self.controller.running:
            self.vehicle_scripted.value = self.typed_controller.vehicle_autonomous()
            self.ped_scripted.value = self.typed_controller.pedestrian_autonomous()
            # Start WASD on whichever ego actor is not driving itself so the
            # keys do something the moment the scenario comes up.
            if self.vehicle_scripted.value and not self.ped_scripted.value:
                self.controller.manual_target = "pedestrian"
            elif self.ped_scripted.value and not self.vehicle_scripted.value:
                self.controller.manual_target = "vehicle"
            self._update_target_button()
            self._sync_pole_stream()

    def _stop_demo(self) -> None:
        self.error_message = ""
        try:
            result = self.typed_controller.stop_demo(remove_tick_callback=False)
            if result["errors"]:
                self.error_message = self.controller.status
        except Exception as exc:
            # A cleanup problem must not escape the Pygame button callback.
            LOG.exception("Unexpected stop-demo cleanup failure")
            self.error_message = "Stop cleanup failed without closing UI: {}".format(
                exc
            )
        self._close_video_windows()
        self._update_target_button()

    def _reset(self) -> None:
        """Backward-compatible alias; the UI no longer exposes unsafe reset."""
        self._stop_demo()

    def _update_target_button(self) -> None:
        target = self.controller.manual_target
        autonomous = (
            self.typed_controller.vehicle_autonomous()
            if target == "vehicle"
            else self.typed_controller.pedestrian_autonomous()
        )
        # Abbreviated so the label always fits the 170 px button; the full
        # "Control: PEDESTRIAN (AUTO)" form overflowed onto the Replay button.
        self.target_button.label = "WASD: {}{}".format(
            "VEH" if target == "vehicle" else "PED",
            " AUTO" if autonomous else "",
        )

    def _switch_manual_target(self) -> None:
        """
        Hand WASD to the other ego actor without disturbing its counterpart.

        Switching the manual target alone was not enough to switch control: if
        the newly selected actor was still autonomous, WASD silently did nothing
        because update_controls() returns early for a scripted actor. With
        ui.control_switch_takes_manual the newly selected actor is therefore
        taken off AUTO.

        Only that actor's mode is touched. The actor being released keeps
        whatever mode it already had, so both ego actors can be manual at the
        same time and passing WASD back and forth never silently restarts a
        route the operator had switched off. Use F2 and F3 to put an actor back
        on its route.
        """
        target = (
            "pedestrian"
            if self.controller.manual_target == "vehicle"
            else "vehicle"
        )
        self.controller.manual_target = target
        # Before Start the toggles are the pending scenario configuration, so
        # only a live scenario hands control over.
        if self.hand_over_control and self.controller.running:
            self.error_message = ""
            if target == "vehicle":
                self.vehicle_scripted.value = False
                message = self._push_vehicle_mode()
            else:
                self.ped_scripted.value = False
                message = self._push_pedestrian_mode()
            if message:
                self.error_message = "Mode change failed: {}".format(message)
        if self.controller.running:
            # A handover between two already-manual actors changes no mode, so
            # nothing else refreshes the status line; without this it keeps
            # claiming the released actor still holds WASD.
            self.controller.status = self._describe_control_state()
        self._update_target_button()

    def _describe_control_state(self) -> str:
        """One line naming both ego modes and which actor WASD drives."""
        return "Vehicle {}; pedestrian {}; WASD -> {}".format(
            "AUTO" if self.typed_controller.vehicle_autonomous() else "MANUAL",
            "AUTO" if self.typed_controller.pedestrian_autonomous() else "MANUAL",
            self.controller.manual_target.upper(),
        )

    def _push_vehicle_mode(self) -> str:
        if not self.controller.running:
            return ""
        if self.vehicle_scripted.value == self.typed_controller.vehicle_autonomous():
            return ""
        try:
            self.typed_controller.set_vehicle_control_mode(
                self.vehicle_scripted.value
            )
        except Exception as exc:
            LOG.exception("Unable to change ego vehicle control mode")
            self.vehicle_scripted.value = (
                self.typed_controller.vehicle_autonomous()
            )
            return "vehicle mode: {}".format(exc)
        return ""

    def _push_pedestrian_mode(self) -> str:
        if not self.controller.running:
            return ""
        if self.ped_scripted.value == self.typed_controller.pedestrian_autonomous():
            return ""
        try:
            self.typed_controller.set_pedestrian_control_mode(
                self.ped_scripted.value
            )
        except Exception as exc:
            LOG.exception("Unable to change ego pedestrian control mode")
            self.ped_scripted.value = (
                self.typed_controller.pedestrian_autonomous()
            )
            return "pedestrian mode: {}".format(exc)
        return ""

    def _apply_control_modes(self, driven_first: str = "vehicle") -> None:
        """
        Push both toggle states, applying the manually driven actor first.

        Taking manual control must succeed even when returning the other actor
        to AUTO fails, for example because its route is already complete.
        """
        self.error_message = ""
        order = (
            (self._push_vehicle_mode, self._push_pedestrian_mode)
            if driven_first == "vehicle"
            else (self._push_pedestrian_mode, self._push_vehicle_mode)
        )
        problems = [message for message in (push() for push in order) if message]
        if problems:
            self.error_message = "Mode change failed: {}".format(
                "; ".join(problems)
            )
        self._update_target_button()

    def _apply_vehicle_control_mode(self) -> None:
        self._apply_control_modes(driven_first="vehicle")

    def _apply_pedestrian_control_mode(self) -> None:
        self._apply_control_modes(driven_first="pedestrian")

    def _toggle_vehicle_control_mode(self) -> None:
        self.vehicle_scripted.value = not self.vehicle_scripted.value
        if not self.vehicle_scripted.value:
            self.controller.manual_target = "vehicle"
        self._apply_control_modes(driven_first="vehicle")

    def _toggle_pedestrian_control_mode(self) -> None:
        self.ped_scripted.value = not self.ped_scripted.value
        if not self.ped_scripted.value:
            self.controller.manual_target = "pedestrian"
        self._apply_control_modes(driven_first="pedestrian")

    def _replay(self) -> None:
        replay_config = copy.deepcopy(self.controller.last_config)
        self._close_video_windows()
        super()._replay()
        if self.controller.running:
            if replay_config is not None:
                self._invalidate_vehicle_route_preview()
                self.vehicle_waypoint_indices = self._normalize_ui_indices(
                    replay_config["scenario"]["ego_vehicle"].get(
                        "route_waypoint_indices", []
                    ),
                    len(self.typed_controller.road_preview),
                )
                self.pedestrian_waypoint_indices = self._normalize_ui_indices(
                    replay_config["scenario"]["ego_pedestrian"].get(
                        "route_waypoint_indices", []
                    ),
                    len(self.pedestrian_points),
                )
                self._update_map_mode_labels()
            self.vehicle_scripted.value = (
                self.typed_controller.vehicle_autonomous()
            )
            self.ped_scripted.value = (
                self.typed_controller.pedestrian_autonomous()
            )
            self._update_target_button()
            self._sync_pole_stream()

    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            mods = getattr(event, "mod", pygame.key.get_mods())
            if event.key == pygame.K_ESCAPE or (
                event.key == pygame.K_q and mods & pygame.KMOD_CTRL
            ):
                return False
            if event.key == pygame.K_s and mods & pygame.KMOD_CTRL:
                self._guard(self._save_vehicle_route)
            elif event.key == pygame.K_o and mods & pygame.KMOD_CTRL:
                self._guard(self._load_vehicle_route)
            elif event.key == pygame.K_TAB:
                self._cycle_view(1)
            elif event.key == pygame.K_F1:
                self._switch_manual_target()
            elif event.key == pygame.K_F2:
                self._toggle_vehicle_control_mode()
            elif event.key == pygame.K_F3:
                self._toggle_pedestrian_control_mode()
            elif event.key == pygame.K_b:
                self.ground_truth.value = not self.ground_truth.value
                self.controller.set_ground_truth(self.ground_truth.value)
            elif event.key == pygame.K_r and not (mods & pygame.KMOD_CTRL):
                self._reset_view()

        # The scenario window always shows the map, so its controls and the
        # camera controls are both live at all times.
        selected_index = self.map_selector.handle_event(event, self.map_mode)
        if selected_index is not None:
            self._apply_map_selection(selected_index)
        for button in self.map_mode_buttons.values():
            button.handle_event(event)
        self.reset_map_button.handle_event(event)
        self.undo_waypoint_button.handle_event(event)
        self.clear_waypoints_button.handle_event(event)
        self.load_route_button.handle_event(event)
        self.save_route_button.handle_event(event)

        for control in self.config_controls:
            changed = control.handle_event(event)
            if not changed:
                continue
            if control is self.seed:
                try:
                    self._refresh_pedestrian_preview()
                except Exception as exc:
                    LOG.exception("Unable to refresh seeded pedestrian preview")
                    self.error_message = "Pedestrian preview unavailable: {}".format(exc)
            if control in (self.vehicle_start, self.vehicle_end):
                self._invalidate_vehicle_route_preview()
            if control is self.ground_truth and self.controller.running:
                self.controller.set_ground_truth(self.ground_truth.value)
            elif control is self.vehicle_scripted:
                if not self.vehicle_scripted.value:
                    self.controller.manual_target = "vehicle"
                self._apply_control_modes(driven_first="vehicle")
            elif control is self.ped_scripted:
                if not self.ped_scripted.value:
                    self.controller.manual_target = "pedestrian"
                self._apply_control_modes(driven_first="pedestrian")
            elif control is self.vehicle_speed and self.controller.running:
                self.typed_controller.set_vehicle_speed(self.vehicle_speed.get())
            elif control is self.ped_speed and self.controller.running:
                self.typed_controller.set_pedestrian_speed(self.ped_speed.get())

        for button in self.common_buttons:
            button.handle_event(event)
        for button in self.camera_buttons:
            button.handle_event(event)
        changed = self.yaw_slider.handle_event(event)
        changed = self.pitch_slider.handle_event(event) or changed
        if changed and self.controller.camera is not None:
            self.controller.camera.set_orientation(
                self.yaw_slider.value, self.pitch_slider.value
            )
        return True

    def _draw_coordinate(
        self,
        location: Optional[carla.Location],
        row: str,
    ) -> None:
        self.screen.blit(
            self.small_font.render(format_location(location), True, v1.COLOR_MUTED),
            (self._scaled(28), self._row(row)),
        )

    def _draw_panel(self) -> None:
        left = self._scaled(18)
        pygame.draw.rect(
            self.screen, v1.COLOR_PANEL, (0, 0, v1.PANEL_WIDTH, self.height)
        )
        self.screen.blit(
            self.title_font.render("Physical AI Scenario", True, v1.COLOR_TEXT),
            (left, self._row("title")),
        )
        self.screen.blit(
            self.small_font.render(
                "Coordinate-aware deterministic controller v2",
                True,
                v1.COLOR_MUTED,
            ),
            (left, self._row("subtitle")),
        )
        for text, row in (
            ("ACTOR POPULATION", "head_actors"),
            ("EGO VEHICLE ROUTE", "head_vehicle"),
            ("EGO PEDESTRIAN ROUTE", "head_pedestrian"),
            ("OCCLUSION", "head_occlusion"),
        ):
            self.screen.blit(
                self.small_font.render(text, True, v1.COLOR_ACCENT),
                (left, self._row(row)),
            )
        for count, row in (
            (len(self.vehicle_waypoint_indices), "head_vehicle"),
            (len(self.pedestrian_waypoint_indices), "head_pedestrian"),
        ):
            via = self.small_font.render(
                "VIA {}".format(count), True, v1.COLOR_MUTED
            )
            self.screen.blit(
                via,
                (v1.PANEL_WIDTH - self._scaled(17) - via.get_width(), self._row(row)),
            )
        for control in self.config_controls:
            control.draw(self.screen, self.font)
        locations = self._selected_locations()
        self._draw_coordinate(locations["vehicle_start"], "vehicle_start_coord")
        self._draw_coordinate(locations["vehicle_end"], "vehicle_end_coord")
        self._draw_coordinate(locations["pedestrian_start"], "ped_start_coord")
        self._draw_coordinate(locations["pedestrian_end"], "ped_end_coord")
        for button in self.common_buttons:
            button.draw(self.screen, self.font)

        mode = "running" if self.controller.running else "stopped"
        status_color = v1.COLOR_GREEN if self.controller.running else v1.COLOR_MUTED
        self.screen.blit(
            self.small_font.render("STATUS [{}]".format(mode), True, status_color),
            (left, self._row("status_label")),
        )
        status_text = self.error_message or self.controller.status
        self._draw_wrapped(
            status_text,
            left,
            self._row("status_text"),
            v1.PANEL_WIDTH - 2 * left,
            v1.COLOR_RED if self.error_message else v1.COLOR_TEXT,
            self.small_font,
            3,
        )
        for text, row in (
            ("F1 WASD | F2 vehicle | F3 ped | B boxes", "hint_1"),
            ("Ctrl+S/O route | Tab camera | arrows aim | Esc", "hint_2"),
        ):
            self.screen.blit(
                self.small_font.render(text, True, v1.COLOR_MUTED),
                (left, self._row(row)),
            )

    def _sync_pole_stream(self) -> None:
        """Subscribe the shared pole sensor only while its window is on screen."""
        camera = self.controller.camera
        setter = getattr(camera, "set_pole_stream_enabled", None)
        if not callable(setter):
            return
        setter(camera.selected.kind == "pole")

    def _video_views(
        self,
        camera: MultiStreamCameraDirector,
    ) -> List[v1.CameraView]:
        """
        The views that get their own desktop window.

        Both ego feeds are always present. A traffic-light-pole feed joins them
        only while that pole is the focused camera, so the demo keeps its
        infrastructure viewpoint without paying for a third render target the
        rest of the time.
        """
        views = [
            view
            for view in (
                camera.view_by_key(EGO_VEHICLE_VIEW_KEY),
                camera.view_by_key(EGO_PEDESTRIAN_VIEW_KEY),
            )
            if view is not None
        ]
        selected = camera.selected
        if selected.kind == "pole":
            views.append(selected)
        return views

    def _mailbox_for_view(
        self,
        view: v1.CameraView,
        camera: MultiStreamCameraDirector,
    ) -> Optional[v1.LatestMailbox]:
        stream = camera.stream_for(view.key)
        if stream is not None:
            return stream.mailbox
        if view.kind == "pole" and camera.sensor is not None:
            return camera.mailbox
        return None

    def _camera_transform_for_view(
        self,
        view: v1.CameraView,
        camera: MultiStreamCameraDirector,
    ) -> carla.Transform:
        if view.kind == "pole":
            return camera.transform_for(view)
        return camera.transform_of(view.key)

    def _pane_excluded_ids(self, view: v1.CameraView) -> List[int]:
        """
        Hide only the actor this camera is mounted on.

        v1 excluded both ego actors from every view, so the pedestrian feed had
        no box on the approaching car and the infrastructure poles had no box on
        either ego actor -- exactly the objects this demo needs highlighted.
        """
        if view.kind == "pole" or view.actor is None:
            return []
        try:
            return [int(view.actor.id)]
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return []

    def _annotate_frame(
        self,
        view: v1.CameraView,
        camera: MultiStreamCameraDirector,
        image: Any,
    ) -> Any:
        """Copy a sensor frame and draw the HUD line and ground-truth boxes."""
        frame = image.copy()
        if self.controller.gt_enabled and self.controller.last_config is not None:
            ground_truth = self.controller.last_config["scenario"]["ground_truth"]
            try:
                draw_ground_truth_boxes_bgr(
                    frame,
                    self.controller._projection_cache.get(),
                    self._camera_transform_for_view(view, camera),
                    float(self.controller.last_config["camera"]["fov_deg"]),
                    float(ground_truth["max_distance_m"]),
                    float(ground_truth["nlos_overlap_threshold"]),
                    self._pane_excluded_ids(view),
                    None
                    if self.controller.occluder is None
                    else self.controller.occluder.id,
                    line_scale=frame.shape[1] / 960.0,
                )
            except (RuntimeError, ValueError) as exc:
                LOG.debug("Ground-truth overlay skipped for %s: %s", view.key, exc)
        badge = ""
        if view.key in EGO_VIEW_KEYS:
            autonomous = (
                self.typed_controller.vehicle_autonomous()
                if view.key == EGO_VEHICLE_VIEW_KEY
                else self.typed_controller.pedestrian_autonomous()
            )
            target_key = (
                EGO_VEHICLE_VIEW_KEY
                if self.controller.manual_target == "vehicle"
                else EGO_PEDESTRIAN_VIEW_KEY
            )
            if autonomous:
                badge = "  [AUTO]"
            elif view.key == target_key:
                badge = "  [MANUAL - WASD]"
            else:
                badge = "  [MANUAL - idle]"
        focused = view.key == camera.selected.key
        hud = "{}{} | frame {} | GT {}{}".format(
            view.label,
            badge,
            self.controller.server_frame,
            "ON" if self.controller.gt_enabled else "OFF",
            "  <- camera controls" if focused else "",
        )
        overlay_height = int(round(34 * frame.shape[1] / 960.0))
        cv2.rectangle(
            frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1
        )
        cv2.putText(
            frame,
            hud,
            (10, int(overlay_height * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * max(0.7, frame.shape[1] / 960.0),
            tuple(reversed(v1.COLOR_ACCENT if focused else v1.COLOR_TEXT)),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _update_video_windows(self) -> None:
        """Push one new frame per live stream into its own desktop window."""
        camera = self.controller.camera
        if not isinstance(camera, MultiStreamCameraDirector):
            if self.video_windows is not None:
                self.video_windows.close_all()
            return
        if self.video_windows is None:
            self.video_windows = VideoWindowBank(
                self.video_window_size, self.video_window_positions
            )
        views = self._video_views(camera)
        visible = {view.key for view in views}
        for view_key in list(self.video_windows.open_windows):
            if view_key not in visible:
                self.video_windows.close(view_key)
        for view in views:
            mailbox = self._mailbox_for_view(view, camera)
            if mailbox is None:
                continue
            image = mailbox.pop()
            if image is None:
                # Nothing new this iteration. The window keeps the last frame it
                # was given, and pump() keeps it responsive, so re-uploading a
                # 1080p texture here only steals time from the simulation clock.
                continue
            try:
                frame = self._annotate_frame(view, camera, carla_image_to_bgr(image))
            except ValueError as exc:
                LOG.debug("Dropped a malformed %s frame: %s", view.key, exc)
                continue
            self.last_frames[view.key] = frame
            self.frame_numbers[view.key] = int(image.frame)
            self.video_windows.show(view, frame, image.frame)
        self.video_windows.pump()

    def _close_video_windows(self) -> None:
        if self.video_windows is not None:
            self.video_windows.close_all()
        self.last_frames.clear()
        self.frame_numbers.clear()

    def _draw_video(self) -> None:
        """The scenario window always shows the top-down selector."""
        self.map_selector.draw(
            self.screen,
            self.font,
            self.small_font,
            self.map_mode,
            self._selection_indices(),
            self.vehicle_waypoint_indices,
            self.pedestrian_waypoint_indices,
            self.vehicle_route_preview,
        )


    def _draw_bottom_frame(self) -> int:
        """Paint the bottom bar background and return its top edge."""
        top = self.height - v1.BOTTOM_HEIGHT
        pygame.draw.rect(
            self.screen,
            v1.COLOR_PANEL,
            (v1.PANEL_WIDTH, top, self.width - v1.PANEL_WIDTH, v1.BOTTOM_HEIGHT),
        )
        pygame.draw.line(
            self.screen,
            v1.COLOR_BORDER,
            (v1.PANEL_WIDTH, top),
            (self.width, top),
            1,
        )
        pygame.draw.line(
            self.screen,
            v1.COLOR_BORDER,
            (self.camera_column_x - self._scaled(16), top + self._scaled(10)),
            (
                self.camera_column_x - self._scaled(16),
                self.height - self._scaled(10),
            ),
            1,
        )
        return top

    def _draw_map_controls(self, top: int) -> None:
        """Left column of the bottom bar: route point selection."""
        left = v1.PANEL_WIDTH + self._scaled(25)
        self.screen.blit(
            self.font.render("MAP SELECTION TARGET", True, v1.COLOR_TEXT),
            (left, top + self._scaled(BOTTOM_ROWS["title"])),
        )
        for button in self.map_mode_buttons.values():
            button.draw(self.screen, self.font)
        self.reset_map_button.draw(self.screen, self.font)
        self.undo_waypoint_button.draw(self.screen, self.font)
        self.clear_waypoints_button.draw(self.screen, self.font)
        self.load_route_button.draw(self.screen, self.font)
        self.save_route_button.draw(self.screen, self.font)
        active_waypoints = self._active_waypoint_indices()
        if active_waypoints is not None:
            index_text = " -> ".join("#{}".format(index) for index in active_waypoints)
            if len(index_text) > 56:
                index_text = index_text[:53] + "..."
            active_text = "{} ({}): {}".format(
                MAP_MODE_LABELS[self.map_mode],
                len(active_waypoints),
                index_text or "left-click map points to append",
            )
        else:
            selected = self._selected_locations()[self.map_mode]
            active_text = "{} #{}  {}".format(
                MAP_MODE_LABELS[self.map_mode],
                self._selection_indices()[self.map_mode],
                format_location(selected),
            )
        self.screen.blit(
            self.small_font.render(active_text, True, v1.COLOR_ACCENT),
            (left, top + self._scaled(BOTTOM_ROWS["text_1"])),
        )
        route_status = self._ellipsize_middle(
            self.route_status,
            self.small_font,
            self.camera_column_x - left - self._scaled(25),
        )
        self.screen.blit(
            self.small_font.render(route_status, True, v1.COLOR_MUTED),
            (left, top + self._scaled(BOTTOM_ROWS["text_2"])),
        )

    def _draw_camera_controls(self, top: int) -> None:
        """
        Right column of the bottom bar: live camera aim and stream status.

        These stay reachable at all times because the video feeds are separate
        desktop windows now, so the operator never switches the scenario window
        away from the map to aim a camera.
        """
        left = self.camera_column_x
        camera = self.controller.camera
        self.screen.blit(
            self.font.render("CAMERA CONTROL", True, v1.COLOR_TEXT),
            (left, top + self._scaled(BOTTOM_ROWS["title"])),
        )
        view_text = (
            camera.selected.label if camera is not None else "no active camera"
        )
        self.screen.blit(
            self.small_font.render(
                "Aiming: {}".format(view_text), True, v1.COLOR_ACCENT
            ),
            (left, top + self._scaled(BOTTOM_ROWS["subtitle"])),
        )
        for button in (self.prev_view, self.next_view, self.reset_view):
            button.enabled = camera is not None
            button.draw(self.screen, self.font)
        self.yaw_slider.draw(self.screen, self.font)
        self.pitch_slider.draw(self.screen, self.font)
        windows = (
            [] if self.video_windows is None else sorted(self.video_windows.open_windows)
        )
        radar = self.controller.radar_mailbox.get()
        nearest = (
            "--"
            if radar["nearest_m"] is None
            else "{:.1f} m".format(radar["nearest_m"])
        )
        self.screen.blit(
            self.small_font.render(
                "Video windows: {} | ego radar {} pts, nearest {}".format(
                    len(windows) or "none", radar["points"], nearest
                ),
                True,
                v1.COLOR_MUTED,
            ),
            (left, top + self._scaled(BOTTOM_ROWS["text_1"])),
        )
        self.screen.blit(
            self.small_font.render(
                "Green/orange = LOS*/NLOS* estimate; boxes are CARLA ground truth",
                True,
                v1.COLOR_MUTED,
            ),
            (left, top + self._scaled(BOTTOM_ROWS["text_2"])),
        )

    def _draw_bottom(self) -> None:
        top = self._draw_bottom_frame()
        self._draw_map_controls(top)
        self._draw_camera_controls(top)

    def run(self) -> None:
        """
        Drive control at the simulation period and redraw the panel more slowly.

        Repainting and flipping a window this large costs tens of milliseconds,
        and doing it every iteration held the synchronous clock well under its
        20 Hz target, which made every actor -- most visibly the pedestrian --
        move in slow motion. Control, ticking and the video windows now run
        every iteration while the scenario window repaints at ui.redraw_hz.
        """
        keep_running = True
        next_redraw = 0.0
        redraw_period = 1.0 / max(1.0, self.redraw_hz)
        try:
            while keep_running:
                delta_seconds = self.clock.tick(self.fps) / 1000.0
                for event in pygame.event.get():
                    if not self._handle_event(event):
                        keep_running = False
                        break
                # Manual motion and camera aim are live whenever the scenario
                # window has keyboard focus; the map no longer gates them.
                keys = pygame.key.get_pressed()
                self.controller.update_controls(keys, delta_seconds)
                self.controller.update_camera_keys(keys, delta_seconds)
                self._sync_camera_sliders()
                # Camera poses and ego control are pushed from advance_frame()
                # so they stay on the control period, not the render rate.
                self.typed_controller.advance_frame()
                self._update_video_windows()
                now = time.monotonic()
                if now >= next_redraw:
                    self.screen.fill(v1.COLOR_BG)
                    self._draw_video()
                    self._draw_panel()
                    self._draw_bottom()
                    pygame.display.flip()
                    next_redraw = now + redraw_period
        finally:
            self._close_video_windows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument("--timeout", type=float, default=10.0, help="RPC timeout seconds")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="scenario YAML"
    )
    parser.add_argument(
        "--traffic-light-data",
        type=Path,
        default=DEFAULT_TRAFFIC_LIGHT_DATA,
        help="traffic-light metadata JSON",
    )
    parser.add_argument(
        "--route-config",
        type=Path,
        default=DEFAULT_ROUTE_CONFIG,
        help=(
            "ego-route JSON used by Load/Save and loaded automatically when "
            "it exists (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="delete stale physical_ai_* actors, restore the clock, and exit",
    )
    parser.add_argument(
        "--force-async-on-exit",
        action="store_true",
        help=(
            "always restore asynchronous mode on exit instead of the clock "
            "state found at startup; use after a previous master-clock client "
            "left the world synchronous with nothing ticking it"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = v1.load_yaml_config(args.config.resolve())
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    world_config = config.get("world", {})
    controller = ScenarioControllerV2(
        client,
        world,
        args.traffic_light_data.resolve(),
        float(config["ui"].get("map_waypoint_spacing_m", 4.0)),
        master_clock=bool(world_config.get("master_clock", True)),
        fixed_delta_seconds=float(
            world_config.get("fixed_delta_seconds", 0.05)
        ),
        traffic_manager_port=int(config["traffic_manager"]["port"]),
        restore_world_settings=bool(
            world_config.get("restore_settings_on_exit", True)
        ),
        force_async_on_exit=bool(args.force_async_on_exit),
    )
    LOG.info(
        "Connected to %s:%d; loaded map=%s; master_clock=%s",
        args.host,
        args.port,
        world.get_map().name,
        controller.master_clock,
    )
    try:
        startup_cleanup = controller.stop_demo(remove_tick_callback=False)
        LOG.info(
            "Startup cleanup confirmed %d/%d owned actor deletions",
            startup_cleanup["destroyed"],
            startup_cleanup["requested"],
        )
        if args.cleanup_only:
            if startup_cleanup["errors"]:
                for error in startup_cleanup["errors"]:
                    LOG.error("Cleanup: %s", error)
                return 1
            return 0
        ui = ScenarioUIV2(controller, config, args.route_config.resolve())
        ui.run()
    except KeyboardInterrupt:
        LOG.info("Shutdown requested; deleting owned actors")
    finally:
        try:
            shutdown_result = controller.shutdown()
            if shutdown_result["errors"]:
                for error in shutdown_result["errors"]:
                    LOG.error("Shutdown cleanup: %s", error)
        except Exception:
            LOG.exception("Unexpected shutdown cleanup failure")
        finally:
            pygame.quit()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    raise SystemExit(main())
