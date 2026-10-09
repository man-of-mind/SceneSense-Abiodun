#!/usr/bin/env python3

"""
CARLA traffic-light-pole RGB camera selector and pan/tilt UI (v1).

The client reads traffic-light metadata from ``traffic_lights_data.json``,
reconciles it with the traffic-light actors in the already loaded CARLA world,
and exposes a camera viewpoint at the top of every live traffic-light pole.

For GPU safety, the viewpoints share one physical CARLA RGB sensor actor. The
sensor is relocated to the selected pole instead of allocating one Vulkan
render target per pole. Select a pole from the left panel, then drag the Yaw
and Pitch sliders below the video. Each viewpoint remembers its own pan/tilt.

This is a passive client. It never starts CARLA, loads/reloads a map, changes
world settings, or calls ``world.tick()``. Camera frames follow the clock of
the server's existing master client (for example, ``generate_traffic.py``).

Controls
--------
    Left-click pole       select its RGB camera
    Mouse wheel           scroll the pole list
    Drag Yaw slider       rotate selected camera left/right
    Drag Pitch slider     rotate selected camera up/down
    Reset View button     restore selected camera's initial orientation
    R                     restore selected camera's initial orientation
    Esc / Q               quit and destroy the shared camera

Example
-------
    python3 traffic_light_pole_camera_ui_client_v1.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import carla
import numpy as np

try:
    import pygame
except ImportError as exc:
    raise RuntimeError(
        "pygame is required for the camera UI. Install it in the CARLA "
        "Python environment with 'python3 -m pip install pygame'."
    ) from exc


LOG = logging.getLogger("traffic_light_pole_camera_ui")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAFFIC_LIGHT_DATA = SCRIPT_DIR / "traffic_lights_data.json"

DEFAULT_CAMERA_HEIGHT_M = 5.0
DEFAULT_CAMERA_YAW_OFFSET_DEG = 90.0
DEFAULT_CAMERA_PITCH_DEG = -20.0
YAW_RANGE_DEG = (-180.0, 180.0)
PITCH_RANGE_DEG = (-90.0, 30.0)

SIDEBAR_WIDTH = 280
CONTROLS_HEIGHT = 180
LIST_MARGIN = 12
LIST_ITEM_HEIGHT = 34

COLOR_BACKGROUND = (18, 21, 27)
COLOR_PANEL = (28, 33, 42)
COLOR_PANEL_ALT = (35, 41, 52)
COLOR_TEXT = (235, 239, 245)
COLOR_MUTED = (155, 166, 181)
COLOR_ACCENT = (49, 156, 255)
COLOR_ACCENT_DARK = (31, 94, 153)
COLOR_BORDER = (72, 82, 99)
COLOR_ERROR = (235, 87, 87)


def parse_resolution(value: str) -> Tuple[int, int]:
    """Parse a WIDTHxHEIGHT camera/display resolution."""
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "resolution must have the form WIDTHxHEIGHT, for example 960x540"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def normalize_yaw(yaw: float) -> float:
    """Normalize a yaw angle to the UI slider's [-180, 180] range."""
    normalized = (float(yaw) + 180.0) % 360.0 - 180.0
    return 180.0 if math.isclose(normalized, -180.0) and yaw > 0.0 else normalized


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def location_distance_3d(first: carla.Location, second: carla.Location) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


@dataclass(frozen=True)
class TrafficLightRecord:
    """Validated entry from traffic_lights_data.json."""

    metadata_id: int
    location: carla.Location


def _finite_number(value, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be numeric, not boolean".format(field_name))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric".format(field_name)) from exc
    if not math.isfinite(number):
        raise ValueError("{} must be finite".format(field_name))
    return number


def load_traffic_light_records(path: Path) -> List[TrafficLightRecord]:
    """Load and validate the traffic-light metadata file."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "traffic-light metadata file was not found: {}".format(path)
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            "traffic-light metadata is not valid JSON: {}:{}".format(
                path, exc.lineno
            )
        ) from exc

    if not isinstance(payload, list) or not payload:
        raise ValueError("traffic-light metadata must be a non-empty JSON list")

    records = []
    seen_ids = set()
    for index, item in enumerate(payload):
        prefix = "traffic-light record [{}]".format(index)
        if not isinstance(item, dict):
            raise ValueError("{} must be an object".format(prefix))
        if "id" not in item or "location" not in item:
            raise ValueError("{} must contain id and location".format(prefix))
        try:
            metadata_id = int(item["id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("{} id must be an integer".format(prefix)) from exc
        if metadata_id in seen_ids:
            raise ValueError("duplicate traffic-light id {}".format(metadata_id))

        location = item["location"]
        if not isinstance(location, dict):
            raise ValueError("{} location must be an object".format(prefix))
        missing_axes = [axis for axis in ("x", "y", "z") if axis not in location]
        if missing_axes:
            raise ValueError(
                "{} location is missing {}".format(prefix, ", ".join(missing_axes))
            )

        seen_ids.add(metadata_id)
        records.append(
            TrafficLightRecord(
                metadata_id=metadata_id,
                location=carla.Location(
                    x=_finite_number(location["x"], "{}.location.x".format(prefix)),
                    y=_finite_number(location["y"], "{}.location.y".format(prefix)),
                    z=_finite_number(location["z"], "{}.location.z".format(prefix)),
                ),
            )
        )

    return records


@dataclass(frozen=True)
class ResolvedPole:
    """One live traffic-light actor and its optional JSON metadata match."""

    actor: carla.Actor
    metadata: Optional[TrafficLightRecord]
    match_method: str

    @property
    def display_id(self) -> int:
        return (
            self.metadata.metadata_id
            if self.metadata is not None
            else int(self.actor.id)
        )

    @property
    def label(self) -> str:
        if self.metadata is None:
            return "Actor {} (live only)".format(self.actor.id)
        if int(self.actor.id) == self.metadata.metadata_id:
            return "Pole {}".format(self.metadata.metadata_id)
        return "Pole {} (actor {})".format(
            self.metadata.metadata_id, self.actor.id
        )


def resolve_live_poles(
    world: carla.World,
    records: Sequence[TrafficLightRecord],
    match_tolerance_m: float,
) -> List[ResolvedPole]:
    """
    Reconcile JSON records with every live traffic-light actor.

    Exact actor-ID plus location matches are preferred. Remaining actors are
    matched to the nearest unused JSON location. Live actors without metadata
    are retained so every available pole still receives a camera.
    """
    actors = sorted(
        world.get_actors().filter("traffic.traffic_light"),
        key=lambda actor: int(actor.id),
    )
    if not actors:
        raise RuntimeError(
            "the loaded CARLA world has no traffic.traffic_light actors"
        )

    records_by_id = {record.metadata_id: record for record in records}
    unmatched_records = {record.metadata_id: record for record in records}
    resolved_by_actor_id = {}

    for actor in actors:
        record = records_by_id.get(int(actor.id))
        if record is None:
            continue
        distance = location_distance_3d(actor.get_location(), record.location)
        if distance <= match_tolerance_m:
            resolved_by_actor_id[int(actor.id)] = ResolvedPole(
                actor=actor,
                metadata=record,
                match_method="actor_id",
            )
            unmatched_records.pop(record.metadata_id, None)
        else:
            LOG.warning(
                "Actor id %d exists in JSON but is %.2f m from its saved "
                "location; trying location-based matching",
                actor.id,
                distance,
            )

    for actor in actors:
        if int(actor.id) in resolved_by_actor_id or not unmatched_records:
            continue
        actor_location = actor.get_location()
        nearest = min(
            unmatched_records.values(),
            key=lambda record: location_distance_3d(
                actor_location, record.location
            ),
        )
        distance = location_distance_3d(actor_location, nearest.location)
        if distance <= match_tolerance_m:
            resolved_by_actor_id[int(actor.id)] = ResolvedPole(
                actor=actor,
                metadata=nearest,
                match_method="location",
            )
            unmatched_records.pop(nearest.metadata_id, None)

    for actor in actors:
        if int(actor.id) not in resolved_by_actor_id:
            resolved_by_actor_id[int(actor.id)] = ResolvedPole(
                actor=actor,
                metadata=None,
                match_method="live_only",
            )
            LOG.warning(
                "Live traffic-light actor %d has no JSON match within %.2f m; "
                "it will still receive a camera",
                actor.id,
                match_tolerance_m,
            )

    for record in sorted(
        unmatched_records.values(), key=lambda item: item.metadata_id
    ):
        LOG.warning(
            "JSON traffic-light id %d has no matching live actor and cannot "
            "receive a camera",
            record.metadata_id,
        )

    resolved = list(resolved_by_actor_id.values())
    resolved.sort(key=lambda pole: (pole.display_id, int(pole.actor.id)))
    return resolved


def transform_relative_location(
    base_transform: carla.Transform,
    relative_location: carla.Location,
) -> carla.Location:
    """Transform a pole-local sensor offset into CARLA world coordinates."""
    matrix = np.asarray(base_transform.get_matrix(), dtype=np.float64)
    point = np.asarray(
        [
            relative_location.x,
            relative_location.y,
            relative_location.z,
            1.0,
        ],
        dtype=np.float64,
    )
    x, y, z, _ = matrix @ point
    return carla.Location(x=float(x), y=float(y), z=float(z))


class LatestFrame:
    """One-frame mailbox: stale images are replaced, never queued."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image = None

    def push(self, image: carla.Image) -> None:
        with self._lock:
            self._image = image

    def pop(self):
        with self._lock:
            image = self._image
            self._image = None
        return image

    def clear(self) -> None:
        with self._lock:
            self._image = None


@dataclass
class PoleCamera:
    """A pole-top camera viewpoint with independent pan/tilt state."""

    pole: ResolvedPole
    location: carla.Location
    default_yaw: float
    default_pitch: float
    yaw: float
    pitch: float

    @property
    def label(self) -> str:
        return self.pole.label

    @property
    def actor_id(self) -> int:
        return int(self.pole.actor.id)

    def set_orientation(self, yaw: float, pitch: float) -> None:
        self.yaw = clamp(yaw, *YAW_RANGE_DEG)
        self.pitch = clamp(pitch, *PITCH_RANGE_DEG)

    def reset_orientation(self) -> None:
        self.set_orientation(self.default_yaw, self.default_pitch)

    def transform(self) -> carla.Transform:
        return carla.Transform(
            carla.Location(
                x=self.location.x,
                y=self.location.y,
                z=self.location.z,
            ),
            carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=0.0),
        )


class CameraBank:
    """Move one physical RGB sensor among all pole-top viewpoints."""

    def __init__(
        self,
        cameras: Sequence[PoleCamera],
        sensor: carla.Sensor,
    ) -> None:
        if not cameras:
            raise ValueError("CameraBank requires at least one camera")
        self.cameras = list(cameras)
        self.sensor = sensor
        self.frames = LatestFrame()
        self.selected_index = -1
        self.listening = False

    @property
    def selected(self) -> PoleCamera:
        return self.cameras[self.selected_index]

    def _restart_stream_at_selected_view(self) -> None:
        # stop() releases the callback but retains this single sensor/render
        # target. Restarting after relocation also prevents an in-flight frame
        # from the previous pole appearing in the newly selected view.
        if self.listening:
            self.sensor.stop()
            self.listening = False
        self.frames.clear()
        self.sensor.set_transform(self.selected.transform())
        self.sensor.listen(self.frames.push)
        self.listening = True

    def select(self, index: int) -> bool:
        index = int(index)
        if not 0 <= index < len(self.cameras):
            raise IndexError("camera selection is out of range")
        if index == self.selected_index:
            return False
        self.selected_index = index
        self._restart_stream_at_selected_view()
        LOG.info(
            "Selected %s: traffic-light actor=%d, shared camera actor=%d",
            self.selected.label,
            self.selected.actor_id,
            self.sensor.id,
        )
        return True

    def set_selected_orientation(self, yaw: float, pitch: float) -> None:
        self.selected.set_orientation(yaw, pitch)
        self.sensor.set_transform(self.selected.transform())

    def reset_selected_orientation(self) -> None:
        self.selected.reset_orientation()
        self.sensor.set_transform(self.selected.transform())

    def find_initial_index(self, requested_id: Optional[int]) -> int:
        if requested_id is None:
            return 0
        for index, camera in enumerate(self.cameras):
            metadata = camera.pole.metadata
            if (
                camera.actor_id == requested_id
                or (
                    metadata is not None
                    and metadata.metadata_id == requested_id
                )
            ):
                return index
        available = ", ".join(
            str(camera.pole.display_id) for camera in self.cameras
        )
        raise ValueError(
            "initial pole id {} is unavailable; available ids: {}".format(
                requested_id, available
            )
        )

    def destroy(self) -> None:
        try:
            if self.listening:
                self.sensor.stop()
                self.listening = False
        except RuntimeError:
            pass
        self.frames.clear()
        try:
            if self.sensor.is_alive:
                self.sensor.destroy()
        except RuntimeError:
            pass


def camera_blueprint(
    world: carla.World,
    width: int,
    height: int,
    fov: float,
    gamma: float,
    sensor_tick: float,
    role_name: str,
) -> carla.ActorBlueprint:
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", str(sensor_tick))
    if blueprint.has_attribute("gamma"):
        blueprint.set_attribute("gamma", str(gamma))
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
    return blueprint


def build_pole_camera_views(
    poles: Sequence[ResolvedPole],
    args: argparse.Namespace,
) -> List[PoleCamera]:
    """Build all pole-top viewpoints without allocating CARLA sensors."""
    cameras = []
    for index, pole in enumerate(poles):
        pole_transform = pole.actor.get_transform()
        location = transform_relative_location(
            pole_transform,
            carla.Location(
                x=args.camera_x,
                y=args.camera_y,
                z=args.camera_height,
            ),
        )
        yaw = normalize_yaw(
            pole_transform.rotation.yaw + args.initial_yaw_offset
        )
        pitch = clamp(
            args.initial_pitch, PITCH_RANGE_DEG[0], PITCH_RANGE_DEG[1]
        )
        camera = PoleCamera(
            pole=pole,
            location=location,
            default_yaw=yaw,
            default_pitch=pitch,
            yaw=yaw,
            pitch=pitch,
        )
        cameras.append(camera)
        LOG.info(
            "Prepared viewpoint %d/%d: %s, traffic-light actor=%d, "
            "location=(%.2f, %.2f, %.2f), yaw=%.1f, pitch=%.1f, "
            "metadata_match=%s",
            index + 1,
            len(poles),
            pole.label,
            pole.actor.id,
            location.x,
            location.y,
            location.z,
            yaw,
            pitch,
            pole.match_method,
        )
    return cameras


def spawn_shared_rgb_camera(
    world: carla.World,
    initial_view: PoleCamera,
    args: argparse.Namespace,
) -> carla.Sensor:
    """Allocate the only server-side RGB sensor/render target used by the UI."""
    sensor = world.spawn_actor(
        camera_blueprint(
            world=world,
            width=args.resolution[0],
            height=args.resolution[1],
            fov=args.fov,
            gamma=args.gamma,
            sensor_tick=args.sensor_tick,
            role_name="shared_pole_rgb_camera",
        ),
        initial_view.transform(),
    )
    LOG.info(
        "Spawned one shared RGB camera actor=%d at initial viewpoint %s",
        sensor.id,
        initial_view.label,
    )
    return sensor


class Slider:
    """Small dependency-free horizontal pygame slider."""

    HANDLE_RADIUS = 10

    def __init__(
        self,
        label: str,
        rect: pygame.Rect,
        minimum: float,
        maximum: float,
        value: float,
    ) -> None:
        self.label = label
        self.rect = rect
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.value = clamp(value, self.minimum, self.maximum)
        self.dragging = False

    def set_value(self, value: float) -> None:
        self.value = clamp(value, self.minimum, self.maximum)

    def _set_from_x(self, x_position: int) -> bool:
        fraction = clamp(
            (x_position - self.rect.left) / max(1, self.rect.width),
            0.0,
            1.0,
        )
        new_value = self.minimum + fraction * (self.maximum - self.minimum)
        changed = not math.isclose(new_value, self.value, abs_tol=1e-4)
        self.value = new_value
        return changed

    def handle_event(self, event: pygame.event.Event) -> bool:
        interaction_rect = self.rect.inflate(
            self.HANDLE_RADIUS * 2, self.HANDLE_RADIUS * 3
        )
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and interaction_rect.collidepoint(event.pos)
        ):
            self.dragging = True
            return self._set_from_x(event.pos[0])
        if event.type == pygame.MOUSEMOTION and self.dragging:
            return self._set_from_x(event.pos[0])
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        return False

    def draw(
        self,
        display: pygame.Surface,
        font: pygame.font.Font,
    ) -> None:
        label_surface = font.render(
            "{}: {:+.1f} deg".format(self.label, self.value),
            True,
            COLOR_TEXT,
        )
        display.blit(label_surface, (self.rect.left, self.rect.top - 30))
        pygame.draw.line(
            display,
            COLOR_BORDER,
            (self.rect.left, self.rect.centery),
            (self.rect.right, self.rect.centery),
            6,
        )
        fraction = (self.value - self.minimum) / (
            self.maximum - self.minimum
        )
        handle_x = round(self.rect.left + fraction * self.rect.width)
        pygame.draw.line(
            display,
            COLOR_ACCENT,
            (self.rect.left, self.rect.centery),
            (handle_x, self.rect.centery),
            6,
        )
        pygame.draw.circle(
            display,
            COLOR_TEXT,
            (handle_x, self.rect.centery),
            self.HANDLE_RADIUS + 2,
        )
        pygame.draw.circle(
            display,
            COLOR_ACCENT,
            (handle_x, self.rect.centery),
            self.HANDLE_RADIUS,
        )


class CameraListPanel:
    """Scrollable list for selecting one pole camera."""

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.scroll_offset = 0

    def visible_count(self) -> int:
        return max(1, self.rect.height // LIST_ITEM_HEIGHT)

    def _clamp_scroll(self, item_count: int) -> None:
        maximum = max(0, item_count - self.visible_count())
        self.scroll_offset = max(0, min(maximum, self.scroll_offset))

    def ensure_visible(self, selected_index: int, item_count: int) -> None:
        if selected_index < self.scroll_offset:
            self.scroll_offset = selected_index
        elif selected_index >= self.scroll_offset + self.visible_count():
            self.scroll_offset = selected_index - self.visible_count() + 1
        self._clamp_scroll(item_count)

    def handle_event(
        self,
        event: pygame.event.Event,
        item_count: int,
    ) -> Optional[int]:
        if event.type == pygame.MOUSEWHEEL:
            mouse_position = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_position):
                self.scroll_offset -= event.y
                self._clamp_scroll(item_count)
            return None
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        ):
            local_y = event.pos[1] - self.rect.top
            index = self.scroll_offset + local_y // LIST_ITEM_HEIGHT
            if 0 <= index < item_count:
                return int(index)
        return None

    def draw(
        self,
        display: pygame.Surface,
        font: pygame.font.Font,
        cameras: Sequence[PoleCamera],
        selected_index: int,
    ) -> None:
        pygame.draw.rect(display, COLOR_BACKGROUND, self.rect)
        pygame.draw.rect(display, COLOR_BORDER, self.rect, 1)
        self._clamp_scroll(len(cameras))
        end_index = min(
            len(cameras), self.scroll_offset + self.visible_count()
        )

        old_clip = display.get_clip()
        display.set_clip(self.rect)
        for visible_row, index in enumerate(
            range(self.scroll_offset, end_index)
        ):
            item_rect = pygame.Rect(
                self.rect.left + 3,
                self.rect.top + visible_row * LIST_ITEM_HEIGHT + 2,
                self.rect.width - 6,
                LIST_ITEM_HEIGHT - 4,
            )
            color = (
                COLOR_ACCENT_DARK
                if index == selected_index
                else COLOR_PANEL_ALT
            )
            pygame.draw.rect(display, color, item_rect, border_radius=5)
            if index == selected_index:
                pygame.draw.rect(
                    display, COLOR_ACCENT, item_rect, 2, border_radius=5
                )
            label = cameras[index].label
            maximum_chars = max(8, (item_rect.width - 20) // 8)
            if len(label) > maximum_chars:
                label = label[: maximum_chars - 1] + "…"
            text = font.render(label, True, COLOR_TEXT)
            display.blit(
                text,
                (
                    item_rect.left + 10,
                    item_rect.centery - text.get_height() // 2,
                ),
            )
        display.set_clip(old_clip)

        if self.scroll_offset > 0:
            up_text = font.render("▲", True, COLOR_MUTED)
            display.blit(
                up_text,
                (self.rect.right - 22, self.rect.top + 4),
            )
        if end_index < len(cameras):
            down_text = font.render("▼", True, COLOR_MUTED)
            display.blit(
                down_text,
                (self.rect.right - 22, self.rect.bottom - 22),
            )


def image_to_surface(image: carla.Image) -> pygame.Surface:
    """Convert one CARLA BGRA camera image to a pygame RGB surface."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    expected_size = image.width * image.height * 4
    if array.size != expected_size:
        raise ValueError(
            "camera frame has {} bytes; expected {}".format(
                array.size, expected_size
            )
        )
    bgra = array.reshape((image.height, image.width, 4))
    rgb = bgra[:, :, :3][:, :, ::-1]
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def draw_sidebar(
    display: pygame.Surface,
    fonts,
    list_panel: CameraListPanel,
    camera_bank: CameraBank,
) -> None:
    pygame.draw.rect(
        display,
        COLOR_PANEL,
        pygame.Rect(0, 0, SIDEBAR_WIDTH, display.get_height()),
    )
    title = fonts["title"].render("Pole Cameras", True, COLOR_TEXT)
    display.blit(title, (LIST_MARGIN, 12))
    subtitle = fonts["small"].render(
        "{} pole viewpoints • click to select".format(
            len(camera_bank.cameras)
        ),
        True,
        COLOR_MUTED,
    )
    display.blit(subtitle, (LIST_MARGIN, 43))
    list_panel.draw(
        display,
        fonts["body"],
        camera_bank.cameras,
        camera_bank.selected_index,
    )


def draw_video(
    display: pygame.Surface,
    video_rect: pygame.Rect,
    surface: Optional[pygame.Surface],
    fonts,
    camera: PoleCamera,
    shared_sensor_id: int,
    frame_id: Optional[int],
    client_fps: float,
) -> None:
    if surface is None:
        pygame.draw.rect(display, (4, 5, 7), video_rect)
        waiting = fonts["body"].render(
            "Waiting for the master clock / first camera frame…",
            True,
            COLOR_TEXT,
        )
        display.blit(
            waiting,
            (
                video_rect.centerx - waiting.get_width() // 2,
                video_rect.centery - waiting.get_height() // 2,
            ),
        )
    else:
        if surface.get_size() != video_rect.size:
            surface = pygame.transform.smoothscale(surface, video_rect.size)
        display.blit(surface, video_rect.topleft)

    overlay = pygame.Surface(
        (video_rect.width, 58), pygame.SRCALPHA
    )
    overlay.fill((0, 0, 0, 150))
    display.blit(overlay, video_rect.topleft)
    label = fonts["title"].render(camera.label, True, COLOR_TEXT)
    status = fonts["small"].render(
        "TL actor {} • camera {} • frame {} • UI {:.0f} FPS".format(
            camera.actor_id,
            shared_sensor_id,
            "--" if frame_id is None else frame_id,
            client_fps,
        ),
        True,
        COLOR_MUTED,
    )
    display.blit(label, (video_rect.left + 12, video_rect.top + 7))
    display.blit(status, (video_rect.left + 13, video_rect.top + 35))


def draw_controls(
    display: pygame.Surface,
    controls_rect: pygame.Rect,
    fonts,
    yaw_slider: Slider,
    pitch_slider: Slider,
    reset_button: pygame.Rect,
    camera: PoleCamera,
) -> None:
    pygame.draw.rect(display, COLOR_PANEL, controls_rect)
    pygame.draw.line(
        display,
        COLOR_BORDER,
        controls_rect.topleft,
        controls_rect.topright,
        1,
    )
    yaw_slider.draw(display, fonts["body"])
    pitch_slider.draw(display, fonts["body"])

    pygame.draw.rect(
        display, COLOR_ACCENT_DARK, reset_button, border_radius=5
    )
    pygame.draw.rect(
        display, COLOR_ACCENT, reset_button, 1, border_radius=5
    )
    reset_text = fonts["body"].render("Reset View", True, COLOR_TEXT)
    display.blit(
        reset_text,
        (
            reset_button.centerx - reset_text.get_width() // 2,
            reset_button.centery - reset_text.get_height() // 2,
        ),
    )
    location_text = fonts["small"].render(
        "Mount: ({:.1f}, {:.1f}, {:.1f}) m".format(
            camera.location.x,
            camera.location.y,
            camera.location.z,
        ),
        True,
        COLOR_MUTED,
    )
    display.blit(
        location_text,
        (reset_button.left, reset_button.bottom + 9),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("-p", "--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="CARLA RPC timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--traffic-lights-json",
        type=Path,
        default=DEFAULT_TRAFFIC_LIGHT_DATA,
        help="traffic-light metadata JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=3.0,
        help="maximum JSON-to-live-actor location match in meters",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(960, 540),
        metavar="WIDTHxHEIGHT",
        help="camera/video resolution (default: 960x540)",
    )
    parser.add_argument("--fov", type=float, default=90.0, help="RGB camera FOV")
    parser.add_argument("--gamma", type=float, default=2.2, help="RGB camera gamma")
    parser.add_argument(
        "--sensor-tick",
        type=float,
        default=0.0,
        help="camera interval in simulation seconds; 0 captures every master tick",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=DEFAULT_CAMERA_HEIGHT_M,
        help="camera height above each pole transform in meters",
    )
    parser.add_argument(
        "--camera-x",
        type=float,
        default=0.0,
        help="pole-local camera x offset in meters",
    )
    parser.add_argument(
        "--camera-y",
        type=float,
        default=0.0,
        help="pole-local camera y offset in meters",
    )
    parser.add_argument(
        "--initial-yaw-offset",
        type=float,
        default=DEFAULT_CAMERA_YAW_OFFSET_DEG,
        help="initial yaw offset from each traffic-light rotation",
    )
    parser.add_argument(
        "--initial-pitch",
        type=float,
        default=DEFAULT_CAMERA_PITCH_DEG,
        help="initial camera pitch; negative looks downward",
    )
    parser.add_argument(
        "--initial-pole-id",
        type=int,
        default=None,
        help="JSON pole id or live actor id selected at startup",
    )
    args = parser.parse_args()

    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.match_tolerance <= 0.0:
        parser.error("--match-tolerance must be positive")
    if args.sensor_tick < 0.0:
        parser.error("--sensor-tick cannot be negative")
    if args.camera_height <= 0.0:
        parser.error("--camera-height must be positive")
    if not 1.0 <= args.fov <= 179.0:
        parser.error("--fov must be between 1 and 179 degrees")
    if args.gamma <= 0.0:
        parser.error("--gamma must be positive")
    if not PITCH_RANGE_DEG[0] <= args.initial_pitch <= PITCH_RANGE_DEG[1]:
        parser.error(
            "--initial-pitch must be between {} and {}".format(
                *PITCH_RANGE_DEG
            )
        )
    return args


def run_ui(camera_bank: CameraBank, args: argparse.Namespace) -> None:
    """Run the responsive selector/video/pan-tilt pygame UI."""
    camera_width, camera_height = args.resolution
    display_width = SIDEBAR_WIDTH + camera_width
    display_height = max(560, camera_height + CONTROLS_HEIGHT)
    video_rect = pygame.Rect(
        SIDEBAR_WIDTH, 0, camera_width, camera_height
    )
    controls_rect = pygame.Rect(
        SIDEBAR_WIDTH,
        camera_height,
        camera_width,
        display_height - camera_height,
    )
    list_rect = pygame.Rect(
        LIST_MARGIN,
        70,
        SIDEBAR_WIDTH - 2 * LIST_MARGIN,
        display_height - 82,
    )

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode(
        (display_width, display_height),
        pygame.HWSURFACE | pygame.DOUBLEBUF,
    )
    pygame.display.set_caption("CARLA Traffic-Light Pole Cameras")
    fonts = {
        "title": pygame.font.Font(pygame.font.get_default_font(), 20),
        "body": pygame.font.Font(pygame.font.get_default_font(), 17),
        "small": pygame.font.Font(pygame.font.get_default_font(), 14),
    }

    slider_left = controls_rect.left + 32
    reset_width = 120
    slider_width = max(180, controls_rect.width - reset_width - 95)
    yaw_slider = Slider(
        "Yaw",
        pygame.Rect(slider_left, controls_rect.top + 54, slider_width, 8),
        YAW_RANGE_DEG[0],
        YAW_RANGE_DEG[1],
        camera_bank.selected.yaw,
    )
    pitch_slider = Slider(
        "Pitch",
        pygame.Rect(slider_left, controls_rect.top + 122, slider_width, 8),
        PITCH_RANGE_DEG[0],
        PITCH_RANGE_DEG[1],
        camera_bank.selected.pitch,
    )
    reset_button = pygame.Rect(
        controls_rect.right - reset_width - 25,
        controls_rect.top + 42,
        reset_width,
        38,
    )
    list_panel = CameraListPanel(list_rect)
    list_panel.ensure_visible(
        camera_bank.selected_index, len(camera_bank.cameras)
    )

    clock = pygame.time.Clock()
    latest_surface = None
    latest_frame_id = None
    running = True

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            if event.type == pygame.KEYUP:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    continue
                if event.key == pygame.K_r:
                    camera_bank.reset_selected_orientation()
                    yaw_slider.set_value(camera_bank.selected.yaw)
                    pitch_slider.set_value(camera_bank.selected.pitch)

            selected_index = list_panel.handle_event(
                event, len(camera_bank.cameras)
            )
            if selected_index is not None:
                if camera_bank.select(selected_index):
                    latest_surface = None
                    latest_frame_id = None
                    yaw_slider.set_value(camera_bank.selected.yaw)
                    pitch_slider.set_value(camera_bank.selected.pitch)
                    list_panel.ensure_visible(
                        selected_index, len(camera_bank.cameras)
                    )

            orientation_changed = yaw_slider.handle_event(event)
            orientation_changed = (
                pitch_slider.handle_event(event) or orientation_changed
            )
            if orientation_changed:
                camera_bank.set_selected_orientation(
                    yaw_slider.value, pitch_slider.value
                )

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and reset_button.collidepoint(event.pos)
            ):
                camera_bank.reset_selected_orientation()
                yaw_slider.set_value(camera_bank.selected.yaw)
                pitch_slider.set_value(camera_bank.selected.pitch)

        if not running:
            break
        if not camera_bank.sensor.is_alive:
            raise RuntimeError(
                "shared camera actor {} was destroyed".format(
                    camera_bank.sensor.id
                )
            )

        image = camera_bank.frames.pop()
        if image is not None:
            latest_surface = image_to_surface(image)
            latest_frame_id = int(image.frame)

        display.fill(COLOR_BACKGROUND)
        draw_sidebar(display, fonts, list_panel, camera_bank)
        draw_video(
            display,
            video_rect,
            latest_surface,
            fonts,
            camera_bank.selected,
            camera_bank.sensor.id,
            latest_frame_id,
            clock.get_fps(),
        )
        draw_controls(
            display,
            controls_rect,
            fonts,
            yaw_slider,
            pitch_slider,
            reset_button,
            camera_bank.selected,
        )
        pygame.display.flip()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    camera_bank = None
    cameras = []
    shared_sensor = None
    pygame_started = False
    try:
        records = load_traffic_light_records(
            args.traffic_lights_json.expanduser().resolve()
        )
        LOG.info(
            "Loaded %d traffic-light records from %s",
            len(records),
            args.traffic_lights_json,
        )

        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        # Attach only to the world that is already loaded.
        world = client.get_world()
        settings = world.get_settings()
        LOG.info(
            "Connected to map %s; synchronous_mode=%s, fixed_delta_seconds=%s",
            world.get_map().name,
            settings.synchronous_mode,
            settings.fixed_delta_seconds,
        )
        LOG.info(
            "Passive client: no map load, world settings update, or world.tick()"
        )

        poles = resolve_live_poles(
            world,
            records,
            match_tolerance_m=args.match_tolerance,
        )
        LOG.info(
            "Resolved %d live traffic-light poles; preparing selectable viewpoints",
            len(poles),
        )
        cameras = build_pole_camera_views(poles, args)
        shared_sensor = spawn_shared_rgb_camera(world, cameras[0], args)
        camera_bank = CameraBank(cameras, shared_sensor)
        initial_index = camera_bank.find_initial_index(args.initial_pole_id)
        camera_bank.select(initial_index)

        pygame_started = True
        run_ui(camera_bank, args)
    finally:
        if camera_bank is not None:
            camera_bank.destroy()
            LOG.info("Destroyed the shared pole camera actor")
        elif shared_sensor is not None:
            try:
                shared_sensor.stop()
            except RuntimeError:
                pass
            try:
                if shared_sensor.is_alive:
                    shared_sensor.destroy()
            except RuntimeError:
                pass
        if pygame_started:
            pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
