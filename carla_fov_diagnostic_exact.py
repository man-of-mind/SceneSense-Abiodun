#!/usr/bin/env python3
"""
Experiment 3: FOV-position diagnostic — REUSES EXACT VALIDATED INFERENCE.

This script reuses the exact inference block from carla_fusion_staleness_scenario.py:
- Model forward + decode_objects call (lines ~1100-1150)
- Sensor preprocessing: RGB normalization + radar tensor preparation
- camera_matrix = actor_world_matrix(camera) passed to decode_objects
- Validated ego camera mount: x=1.8, z=1.55, pitch=-4°

Setup: ego on-lane behind fixed target; FOV sweep by lateral offset.
Mandatory sanity checks BEFORE writing findings:
1. Captured frame must show target car clearly (not black)
2. Center-offset loc error must be ~1–1.5 m (validated floor)
"""

import argparse
import csv
import math
import queue
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import carla

sys.path.insert(0, str(Path(__file__).resolve().parent / "pole_lraspp_multimodal_fusion"))
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp
from pole_lraspp_multimodal_fusion.object_targets import decode_objects
from pole_lraspp_multimodal_fusion.radar_fusion import (
    StationaryTrackAccumulator,
    build_radar_sample,
    radar_raw_to_alt_az_depth_velocity,
)
from pole_lraspp_multimodal_fusion.split_runtime import MultimodalLRASPPSplitModel

try:
    import cv2
except ImportError:
    cv2 = None

# ====== VALIDATED CONSTANTS from carla_fusion_staleness_scenario.py ======
DEFAULT_EGO_CAMERA_X = 1.8
DEFAULT_EGO_CAMERA_Z = 1.55
DEFAULT_EGO_CAMERA_PITCH = -4.0
DEFAULT_EGO_CAMERA_FOV = 120.0
DEFAULT_EGO_RADAR_X = 2.0
DEFAULT_EGO_RADAR_Y = 0.0
DEFAULT_EGO_RADAR_Z = 1.0

BASELINE_CHECKPOINT = Path(
    __file__
).parent / "experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"

# Radar config from fusion_full_run.yaml
RADAR_RANGE = 120.0
RADAR_HFOV = 100.0
RADAR_VFOV = 30.0
RADAR_PPS = 5000
STATIONARY_VEL_MPS = 0.35
PARKED_THRESHOLD_S = 5.0
ASSOC_GRID_M = 1.5
MAX_STALE_S = 2.0

# Model config
NUM_CLASSES = 3
OBJECT_SCORE_THRESHOLD = 0.05
OBJECT_NMS_RADIUS_PX = 4
TOPK_OBJECTS = 80
OBJECT_HIDDEN_CHANNELS = 128

# Diagnostic config
LATERAL_OFFSETS = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
EGO_DISTANCE_M = 15.0  # target is placed this far ahead of the ego on the ego's lane (centered in FOV)
FRAMES_PER_OFFSET = 15
MATCH_GATE_M = 5.0
FIXED_DELTA_SECONDS = 0.05
WARMUP_FRAMES = int(math.ceil(PARKED_THRESHOLD_S / FIXED_DELTA_SECONDS)) + 5

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "fov_diagnostic_exact_v2"


def actor_world_matrix(actor: carla.Actor) -> np.ndarray:
    """VALIDATED: from carla_fusion_staleness_scenario.py line 812"""
    return np.array(actor.get_transform().get_matrix(), dtype=np.float64)


def actor_world_inverse_matrix(actor: carla.Actor) -> np.ndarray:
    return np.array(actor.get_transform().get_inverse_matrix(), dtype=np.float64)


def intrinsics_at(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = (float(width) / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    return np.array(
        [[f, 0.0, float(width) / 2.0], [0.0, f, float(height) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def load_baseline_model(device: torch.device) -> Tuple[MultimodalLRASPPSplitModel, Tuple[int, int]]:
    """Load baseline model with all checkpoint parameters, wrapped in split model"""
    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {BASELINE_CHECKPOINT}")

    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location=device)
    radar_ch = int(checkpoint.get("radar_channels", 4))
    obj_ch = int(checkpoint.get("object_channels", 14))
    fuse_low = bool(checkpoint.get("fuse_low_into_object_head", True))
    head_arch = str(checkpoint.get("object_head_arch", "shared"))
    head_depth = int(checkpoint.get("object_head_depth", 2))
    use_cc = bool(checkpoint.get("object_use_coordconv", False))
    predict_bbox2d = bool(checkpoint.get("object_predict_bbox2d", False))
    use_gp = bool(checkpoint.get("object_use_groundplane_prior", False))
    gp_params = checkpoint.get("object_groundplane_params", {})
    input_size = tuple(checkpoint.get("input_size", [768, 432]))

    model = build_multitask_fusion_lraspp(
        num_classes=NUM_CLASSES,
        radar_channels=radar_ch,
        pretrained=False,
        object_channels=obj_ch,
        object_hidden_channels=OBJECT_HIDDEN_CHANNELS,
        fuse_low_into_object_head=fuse_low,
        head_arch=head_arch,
        head_depth=head_depth,
        use_coordconv=use_cc,
        predict_bbox2d=predict_bbox2d,
        use_groundplane_prior=use_gp,
        groundplane_params=gp_params,
        device=device,
    ).to(device)

    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Wrap in split model (same pattern as validated scenario)
    split_model = MultimodalLRASPPSplitModel(model, device, input_size=input_size)
    split_model.object_predict_bbox2d = predict_bbox2d
    split_model.object_class_names = ["vehicle", "person"]
    return split_model, input_size


def prepare_fusion_input(
    frame_bgr: np.ndarray,
    radar_tensor: np.ndarray,
    model_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """EXACT from carla_fusion_staleness_scenario.py"""
    model_w, model_h = int(model_size[0]), int(model_size[1])

    # RGB preprocessing: BGR → RGB, resize, normalize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if (rgb.shape[1], rgb.shape[0]) != (model_w, model_h):
        rgb = cv2.resize(rgb, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    rgb_tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rgb_tensor = (rgb_tensor - rgb_mean) / rgb_std

    # Radar preprocessing: ensure correct size
    radar = radar_tensor.astype(np.float32)
    if radar.shape[1] != model_h or radar.shape[2] != model_w:
        resized = []
        for idx, channel in enumerate(radar):
            interp = cv2.INTER_NEAREST if idx == 0 else cv2.INTER_LINEAR
            resized.append(cv2.resize(channel, (model_w, model_h), interpolation=interp))
        radar = np.stack(resized, axis=0).astype(np.float32)
    radar_tensor_torch = torch.from_numpy(np.ascontiguousarray(radar)).unsqueeze(0).to(device=device)

    # Concatenate [RGB(3) + radar(4)]
    return torch.cat([rgb_tensor, radar_tensor_torch], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--spawn-index", type=int, default=11)
    parser.add_argument("--frames-per-offset", type=int, default=FRAMES_PER_OFFSET)
    parser.add_argument("--warmup-frames", type=int, default=WARMUP_FRAMES)
    parser.add_argument("--match-gate-m", type=float, default=MATCH_GATE_M)
    parser.add_argument("--center-sanity-max-m", type=float, default=3.0)
    parser.add_argument("--sensor-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--sweep-mode",
        choices=("lateral", "target-angle"),
        default="lateral",
        help="Move the ego laterally, or move the target on a fixed-range arc around a stationary ego.",
    )
    parser.add_argument("--offsets", default=",".join(str(v) for v in LATERAL_OFFSETS))
    parser.add_argument("--angles-deg", default="-45,-30,-20,-10,0,10,20,30,45")
    parser.add_argument(
        "--target-orientation",
        choices=("fixed", "radial"),
        default="fixed",
        help=(
            "Target pose during a target-angle sweep. 'radial' rotates the target with "
            "its arc position so the ego sees the same rear-facing aspect at every angle."
        ),
    )
    parser.add_argument(
        "--motion-mode",
        choices=("static", "convoy"),
        default="static",
        help=(
            "Keep both actors static, or move ego and target at the same exact world velocity. "
            "Convoy mode is a centered, matched-geometry control and requires one 0-degree target angle."
        ),
    )
    parser.add_argument("--convoy-speed-mps", type=float, default=8.9)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def put_latest(items: queue.Queue, item: object) -> None:
    try:
        items.put_nowait(item)
    except queue.Full:
        try:
            items.get_nowait()
        except queue.Empty:
            pass
        items.put_nowait(item)


def wait_for_frame(items: queue.Queue, frame_id: int, timeout_s: float) -> Optional[object]:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            item = items.get(timeout=max(0.01, deadline - time.time()))
        except queue.Empty:
            return None
        item_frame = int(getattr(item, "frame", -1))
        if item_frame < int(frame_id):
            continue
        if item_frame == int(frame_id):
            return item
        return None
    return None


def camera_image_to_bgr(image: carla.Image) -> np.ndarray:
    """CARLA RGB raw data is BGRA; dropping alpha already yields BGR."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    return np.ascontiguousarray(array.reshape((image.height, image.width, 4))[:, :, :3])


def target_bbox_center_world(actor: carla.Actor) -> np.ndarray:
    center_local = actor.bounding_box.location
    point = np.array([center_local.x, center_local.y, center_local.z, 1.0], dtype=np.float64)
    return (actor_world_matrix(actor) @ point)[:3]


def project_world_point(
    point_world: np.ndarray,
    camera_inverse: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> Tuple[float, float, float, bool]:
    point = np.concatenate([np.asarray(point_world, dtype=np.float64), np.ones(1, dtype=np.float64)])
    local = (camera_inverse @ point)[:3]
    depth = float(local[0])
    if depth <= 0.05:
        return float("nan"), float("nan"), depth, False
    u = float(intrinsics[0, 2] + (local[1] / depth) * intrinsics[0, 0])
    v = float(intrinsics[1, 2] - (local[2] / depth) * intrinsics[1, 1])
    visible = 0.0 <= u < float(width) and 0.0 <= v < float(height)
    return u, v, depth, visible


def radar_points_inside_actor(points_world: np.ndarray, actor: carla.Actor, margin_m: float = 0.5) -> int:
    if points_world.size == 0:
        return 0
    inverse = np.asarray(actor.get_transform().get_inverse_matrix(), dtype=np.float64)
    homogeneous = np.concatenate(
        [points_world.astype(np.float64), np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1
    )
    local = (inverse @ homogeneous.T).T[:, :3]
    bbox = actor.bounding_box
    local -= np.asarray([bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float64)[None, :]
    inside = (
        (np.abs(local[:, 0]) <= float(bbox.extent.x) + margin_m)
        & (np.abs(local[:, 1]) <= float(bbox.extent.y) + margin_m)
        & (np.abs(local[:, 2]) <= float(bbox.extent.z) + margin_m)
    )
    return int(np.sum(inside))


def destroy_actor(actor: Optional[carla.Actor]) -> None:
    if actor is None:
        return
    try:
        if "sensor." in str(actor.type_id):
            actor.stop()
    except RuntimeError:
        pass
    try:
        actor.destroy()
    except RuntimeError:
        pass


def parse_offsets(raw: str) -> Tuple[float, ...]:
    offsets = tuple(float(value.strip()) for value in str(raw).split(",") if value.strip())
    if not offsets:
        raise ValueError("--offsets must contain at least one value")
    return offsets


def annotate_frame(
    frame_bgr: np.ndarray,
    *,
    target_u_model: float,
    target_v_model: float,
    prediction: Optional[Dict[str, float]],
    model_size: Tuple[int, int],
    label: str,
) -> np.ndarray:
    preview = frame_bgr.copy()
    scale_x = float(preview.shape[1]) / float(model_size[0])
    scale_y = float(preview.shape[0]) / float(model_size[1])
    if np.isfinite(target_u_model) and np.isfinite(target_v_model):
        gt_xy = (int(round(target_u_model * scale_x)), int(round(target_v_model * scale_y)))
        cv2.drawMarker(preview, gt_xy, (0, 255, 0), cv2.MARKER_CROSS, 28, 3)
    if prediction is not None:
        pred_xy = (
            int(round(float(prediction["center_x_px"]) * scale_x)),
            int(round(float(prediction["center_y_px"]) * scale_y)),
        )
        cv2.drawMarker(preview, pred_xy, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 28, 3)
    cv2.putText(preview, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return preview


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the FOV diagnostic")
    sweep_mode = str(args.sweep_mode)
    sweep_values = parse_offsets(args.angles_deg if sweep_mode == "target-angle" else args.offsets)
    motion_mode = str(args.motion_mode)
    if motion_mode == "convoy" and not (
        sweep_mode == "target-angle"
        and len(sweep_values) == 1
        and abs(float(sweep_values[0])) < 1e-6
    ):
        raise ValueError("convoy mode requires --sweep-mode target-angle --angles-deg 0")
    frames_per_offset = max(1, int(args.frames_per_offset))
    warmup_frames = max(1, int(args.warmup_frames))
    match_gate_m = max(0.1, float(args.match_gate_m))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    frames_dir = output_dir / "frames"
    csv_file = output_dir / "results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print("Loading baseline model...")
    model, model_size = load_baseline_model(device)
    model_w, model_h = int(model_size[0]), int(model_size[1])
    print(f"  Input size: {model_w}x{model_h}")

    print(f"Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(30.0)
    world = client.load_world(str(args.town))
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    map_obj = world.get_map()
    bp_lib = world.get_blueprint_library()
    spawn_points = map_obj.get_spawn_points()
    print(f"Loaded world: {map_obj.name}")

    if not spawn_points:
        raise RuntimeError("CARLA map has no vehicle spawn points")
    ego_sp_idx = min(max(0, int(args.spawn_index)), len(spawn_points) - 1)
    ego_sp = spawn_points[ego_sp_idx]
    ego_base_loc = ego_sp.location
    ego_base_heading = ego_sp.rotation.yaw
    ego_wp = map_obj.get_waypoint(ego_base_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    ahead = ego_wp.next(EGO_DISTANCE_M) if ego_wp else None
    tgt_wp = ahead[0] if ahead else ego_wp
    if tgt_wp is None:
        raise RuntimeError("Could not find a driving waypoint for the target")

    target = None
    all_results = []
    try:
        target_tf = carla.Transform(
            carla.Location(
                tgt_wp.transform.location.x,
                tgt_wp.transform.location.y,
                tgt_wp.transform.location.z + 0.3,
            ),
            tgt_wp.transform.rotation,
        )
        target = world.spawn_actor(bp_lib.find("vehicle.mini.cooper"), target_tf)
        target.set_simulate_physics(False)
        world.tick()
        target_loc = target.get_location()
        separation = math.hypot(target_loc.x - ego_base_loc.x, target_loc.y - ego_base_loc.y)
        print(
            f"Ego base at {ego_base_loc} heading {ego_base_heading:.0f}deg "
            f"(spawn_point[{ego_sp_idx}])"
        )
        print(
            f"Target {EGO_DISTANCE_M:.0f} m ahead at {target_loc}; "
            f"ego-target separation={separation:.1f} m"
        )
        if not (EGO_DISTANCE_M - 2.0 <= separation <= EGO_DISTANCE_M + 2.0):
            raise RuntimeError(f"target placement failed: expected ~{EGO_DISTANCE_M} m, got {separation:.1f} m")

        target_center_z = float(target_loc.z)
        target_rotation = target.get_transform().rotation
        for position_idx, sweep_value in enumerate(sweep_values):
            unit = "deg" if sweep_mode == "target-angle" else "m"
            print(
                f"\n=== Position {position_idx + 1}/{len(sweep_values)}: "
                f"{sweep_value:+.1f} {unit} ({sweep_mode}) ==="
            )
            yaw_rad = math.radians(ego_base_heading)
            if sweep_mode == "target-angle":
                angle_rad = math.radians(float(sweep_value))
                forward_x, forward_y = math.cos(yaw_rad), math.sin(yaw_rad)
                right_x, right_y = -math.sin(yaw_rad), math.cos(yaw_rad)
                target_sweep_loc = carla.Location(
                    x=(
                        ego_base_loc.x
                        + EGO_DISTANCE_M * math.cos(angle_rad) * forward_x
                        + EGO_DISTANCE_M * math.sin(angle_rad) * right_x
                    ),
                    y=(
                        ego_base_loc.y
                        + EGO_DISTANCE_M * math.cos(angle_rad) * forward_y
                        + EGO_DISTANCE_M * math.sin(angle_rad) * right_y
                    ),
                    z=target_center_z,
                )
                if args.target_orientation == "radial":
                    target_sweep_rotation = carla.Rotation(
                        pitch=target_rotation.pitch,
                        yaw=ego_base_heading + float(sweep_value),
                        roll=target_rotation.roll,
                    )
                else:
                    target_sweep_rotation = target_rotation
                target.set_transform(carla.Transform(target_sweep_loc, target_sweep_rotation))
                world.tick()
                ego_loc = carla.Location(
                    x=ego_base_loc.x,
                    y=ego_base_loc.y,
                    z=ego_base_loc.z,
                )
                lateral_offset_m = ""
                target_angle_deg = float(sweep_value)
            else:
                ego_loc = carla.Location(
                    x=ego_base_loc.x - math.sin(yaw_rad) * sweep_value,
                    y=ego_base_loc.y + math.cos(yaw_rad) * sweep_value,
                    z=ego_base_loc.z,
                )
                lateral_offset_m = float(sweep_value)
                target_angle_deg = ""
            ego_rot = carla.Rotation(yaw=ego_base_heading)
            ego = world.try_spawn_actor(
                bp_lib.find("vehicle.mini.cooper"), carla.Transform(ego_loc, ego_rot)
            )
            if ego is None:
                print(f"  SKIP — ego spawn failed (off-road/occupied)")
                continue
            if motion_mode == "convoy":
                convoy_speed_mps = max(0.1, float(args.convoy_speed_mps))
                # CARLA applies enable_constant_velocity in the actor's local frame.
                # Both actors have the same road-aligned yaw, so +local-x preserves
                # their world-space 15 m gap without Traffic Manager drift.
                convoy_velocity = carla.Vector3D(x=convoy_speed_mps, y=0.0, z=0.0)
                ego.set_simulate_physics(True)
                target.set_simulate_physics(True)
                ego.enable_constant_velocity(convoy_velocity)
                target.enable_constant_velocity(convoy_velocity)
                world.tick()
                print(f"  exact convoy velocity: {convoy_speed_mps:.1f} m/s")
            else:
                convoy_speed_mps = 0.0
                ego.set_simulate_physics(False)

            camera = None
            radar = None
            try:
                cam_bp = bp_lib.find("sensor.camera.rgb")
                cam_bp.set_attribute("image_size_x", "1536")
                cam_bp.set_attribute("image_size_y", "864")
                cam_bp.set_attribute("fov", str(DEFAULT_EGO_CAMERA_FOV))
                cam_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
                camera = world.spawn_actor(
                    cam_bp,
                    carla.Transform(
                        carla.Location(x=DEFAULT_EGO_CAMERA_X, z=DEFAULT_EGO_CAMERA_Z),
                        carla.Rotation(pitch=DEFAULT_EGO_CAMERA_PITCH),
                    ),
                    attach_to=ego,
                )

                radar_bp = bp_lib.find("sensor.other.radar")
                radar_bp.set_attribute("range", str(RADAR_RANGE))
                radar_bp.set_attribute("horizontal_fov", str(RADAR_HFOV))
                radar_bp.set_attribute("vertical_fov", str(RADAR_VFOV))
                radar_bp.set_attribute("points_per_second", str(RADAR_PPS))
                radar_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
                radar = world.spawn_actor(
                    radar_bp,
                    carla.Transform(
                        carla.Location(
                            x=DEFAULT_EGO_RADAR_X,
                            y=DEFAULT_EGO_RADAR_Y,
                            z=DEFAULT_EGO_RADAR_Z,
                        )
                    ),
                    attach_to=ego,
                )

                camera_queue: queue.Queue = queue.Queue(maxsize=8)
                radar_queue: queue.Queue = queue.Queue(maxsize=8)
                camera.listen(lambda item, q=camera_queue: put_latest(q, item))
                radar.listen(lambda item, q=radar_queue: put_latest(q, item))
                tracker = StationaryTrackAccumulator(
                    stationary_velocity_mps=STATIONARY_VEL_MPS,
                    parked_threshold_s=PARKED_THRESHOLD_S,
                    association_grid_m=ASSOC_GRID_M,
                    max_stale_s=MAX_STALE_S,
                )
                intrinsics_model = intrinsics_at(model_w, model_h, DEFAULT_EGO_CAMERA_FOV)

                # Warm the persistent stationary-radar tracker for the same parked-evidence semantics as training.
                warmed = 0
                for _ in range(warmup_frames):
                    frame_id = world.tick()
                    image = wait_for_frame(camera_queue, frame_id, args.sensor_timeout_s)
                    radar_meas = wait_for_frame(radar_queue, frame_id, args.sensor_timeout_s)
                    if image is None or radar_meas is None:
                        continue
                    detections = radar_raw_to_alt_az_depth_velocity(bytes(radar_meas.raw_data))
                    build_radar_sample(
                        detections=detections,
                        sensor_matrix=actor_world_matrix(radar),
                        camera_inverse_matrix=actor_world_inverse_matrix(camera),
                        camera_intrinsics=intrinsics_model,
                        width=model_w,
                        height=model_h,
                        frame_time_s=float(radar_meas.timestamp),
                        tracker=tracker,
                        max_range_m=RADAR_RANGE,
                        max_abs_velocity_mps=20.0,
                        parked_threshold_s=PARKED_THRESHOLD_S,
                        point_radius_px=2,
                    )
                    warmed += 1
                print(f"  synchronized warmup: {warmed}/{warmup_frames} frames")
                if warmed < max(5, int(0.9 * warmup_frames)):
                    raise RuntimeError("insufficient synchronized camera/radar warmup frames")

                offset_rows = []
                for sample_idx in range(frames_per_offset):
                    frame_id = world.tick()
                    image = wait_for_frame(camera_queue, frame_id, args.sensor_timeout_s)
                    radar_meas = wait_for_frame(radar_queue, frame_id, args.sensor_timeout_s)
                    if image is None or radar_meas is None:
                        print(f"  frame {frame_id}: missing synchronized camera/radar measurement")
                        continue

                    frame_bgr = camera_image_to_bgr(image)
                    frame_mean = float(np.mean(frame_bgr))
                    frame_std = float(np.std(frame_bgr))
                    if frame_mean < 5.0 or frame_std < 5.0:
                        raise RuntimeError(
                            f"invalid/black camera frame: mean={frame_mean:.1f}, std={frame_std:.1f}"
                        )

                    camera_inverse = actor_world_inverse_matrix(camera)
                    detections = radar_raw_to_alt_az_depth_velocity(bytes(radar_meas.raw_data))
                    radar_tensor, radar_points, radar_summary = build_radar_sample(
                        detections=detections,
                        sensor_matrix=actor_world_matrix(radar),
                        camera_inverse_matrix=camera_inverse,
                        camera_intrinsics=intrinsics_model,
                        width=model_w,
                        height=model_h,
                        frame_time_s=float(radar_meas.timestamp),
                        tracker=tracker,
                        max_range_m=RADAR_RANGE,
                        max_abs_velocity_mps=20.0,
                        parked_threshold_s=PARKED_THRESHOLD_S,
                        point_radius_px=2,
                    )
                    raw_support_count = radar_points_inside_actor(
                        np.asarray(radar_points["world_xyz"]), target
                    )

                    fused = prepare_fusion_input(frame_bgr, radar_tensor, model_size, device)
                    camera_matrix = actor_world_matrix(camera)
                    with torch.inference_mode():
                        features = model.encode(fused)
                        outputs = model.decode_outputs(features, output_size=(model_h, model_w))
                    objects = []
                    if "object" in outputs:
                        objects = decode_objects(
                            outputs["object"],
                            camera_matrix=camera_matrix,
                            topk=TOPK_OBJECTS,
                            score_threshold=OBJECT_SCORE_THRESHOLD,
                            nms_radius_px=OBJECT_NMS_RADIUS_PX,
                            object_class_names=getattr(model, "object_class_names", ["vehicle", "person"]),
                            predict_bbox2d=bool(getattr(model, "object_predict_bbox2d", False)),
                        )
                    vehicle_predictions = [
                        pred for pred in objects if str(pred.get("class_name", "")) == "vehicle"
                    ]
                    ego_now = ego.get_location()
                    target_loc = target.get_location()
                    ego_velocity = ego.get_velocity()
                    target_velocity = target.get_velocity()
                    ego_speed_mps = float(
                        math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2)
                    )
                    target_speed_mps = float(
                        math.sqrt(
                            target_velocity.x ** 2 + target_velocity.y ** 2 + target_velocity.z ** 2
                        )
                    )
                    target_pos = np.asarray([target_loc.x, target_loc.y], dtype=np.float64)
                    candidates = []
                    for pred in vehicle_predictions:
                        pred_pos = np.asarray([pred["world_x"], pred["world_y"]], dtype=np.float64)
                        candidates.append((float(np.linalg.norm(pred_pos - target_pos)), pred))
                    candidates.sort(key=lambda item: item[0])
                    nearest_error = candidates[0][0] if candidates else float("nan")
                    nearest_pred = candidates[0][1] if candidates else None
                    matched = bool(nearest_pred is not None and nearest_error <= match_gate_m)
                    matched_pred = nearest_pred if matched else None

                    target_u, target_v, target_depth, target_visible = project_world_point(
                        target_bbox_center_world(target),
                        camera_inverse,
                        intrinsics_model,
                        model_w,
                        model_h,
                    )
                    if not target_visible:
                        match_reason = "target_out_of_fov"
                    elif not vehicle_predictions:
                        match_reason = "no_vehicle_prediction"
                    elif not matched:
                        match_reason = "outside_match_gate"
                    else:
                        match_reason = "matched"

                    row = {
                        "sweep_mode": sweep_mode,
                        "sweep_value": float(sweep_value),
                        "offset_m": lateral_offset_m,
                        "target_angle_deg": target_angle_deg,
                        "target_orientation_mode": (
                            args.target_orientation if sweep_mode == "target-angle" else "fixed"
                        ),
                        "motion_mode": motion_mode,
                        "commanded_convoy_speed_mps": float(convoy_speed_mps),
                        "ego_speed_mps": ego_speed_mps,
                        "target_speed_mps": target_speed_mps,
                        "target_yaw_deg": float(target.get_transform().rotation.yaw),
                        "ego_yaw_deg": float(ego_base_heading),
                        "sample_idx": int(sample_idx),
                        "frame_id": int(frame_id),
                        "ego_x": float(ego_now.x),
                        "ego_y": float(ego_now.y),
                        "target_x": float(target_loc.x),
                        "target_y": float(target_loc.y),
                        "target_distance_m": float(
                            math.hypot(target_loc.x - ego_now.x, target_loc.y - ego_now.y)
                        ),
                        "target_pixel_x": float(target_u),
                        "target_pixel_x_from_center": float(target_u - model_w / 2.0),
                        "target_pixel_y": float(target_v),
                        "target_in_fov": int(target_visible),
                        "vehicle_prediction_count": int(len(vehicle_predictions)),
                        "matched": int(matched),
                        "match_reason": match_reason,
                        "nearest_error_m": float(nearest_error),
                        "pred_x": float(matched_pred["world_x"]) if matched_pred else "",
                        "pred_y": float(matched_pred["world_y"]) if matched_pred else "",
                        "error_m": float(nearest_error) if matched else "",
                        "pred_pixel_x": float(matched_pred["center_x_px"]) if matched_pred else "",
                        "pred_pixel_x_from_center": (
                            float(matched_pred["center_x_px"] - model_w / 2.0) if matched_pred else ""
                        ),
                        "radar_support_score": (
                            float(matched_pred.get("radar_support_score", 0.0)) if matched_pred else ""
                        ),
                        "raw_radar_support_count": int(raw_support_count),
                        "raw_radar_points": int(radar_summary["radar_points"]),
                        "score": float(matched_pred.get("score", 0.0)) if matched_pred else "",
                        "frame_mean": frame_mean,
                        "frame_std": frame_std,
                    }
                    all_results.append(row)
                    offset_rows.append(row)

                    if sample_idx == 0:
                        position_label = (
                            f"angle={sweep_value:+.0f}deg"
                            if sweep_mode == "target-angle"
                            else f"offset={sweep_value:+.0f}m"
                        )
                        label = (
                            f"{position_label} matched={matched} "
                            f"nearest={nearest_error:.2f}m"
                            if np.isfinite(nearest_error)
                            else f"{position_label} no vehicle prediction"
                        )
                        preview = annotate_frame(
                            frame_bgr,
                            target_u_model=target_u,
                            target_v_model=target_v,
                            prediction=matched_pred,
                            model_size=model_size,
                            label=label,
                        )
                        frame_prefix = "angle" if sweep_mode == "target-angle" else "offset"
                        frame_suffix = "deg" if sweep_mode == "target-angle" else "m"
                        cv2.imwrite(
                            str(frames_dir / f"{frame_prefix}_{sweep_value:+.0f}{frame_suffix}_frame.jpg"),
                            preview,
                        )

                matched_errors = [float(row["error_m"]) for row in offset_rows if row["matched"]]
                print(
                    f"  collected={len(offset_rows)} matched={len(matched_errors)} "
                    f"rate={len(matched_errors) / max(1, len(offset_rows)):.0%}"
                )
                if matched_errors:
                    print(
                        f"  error mean={np.mean(matched_errors):.2f}m "
                        f"std={np.std(matched_errors):.2f}m median={np.median(matched_errors):.2f}m"
                    )
            finally:
                destroy_actor(camera)
                destroy_actor(radar)
                if motion_mode == "convoy":
                    try:
                        target.disable_constant_velocity()
                        target.set_simulate_physics(False)
                    except RuntimeError:
                        pass
                destroy_actor(ego)

        if not all_results:
            raise RuntimeError("No observations were captured")
        with csv_file.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nWrote {len(all_results)} rows to {csv_file}")

        center_rows = [row for row in all_results if abs(float(row["sweep_value"])) < 1e-6]
        center_errors = [float(row["error_m"]) for row in center_rows if row["matched"]]
        if not center_errors:
            raise RuntimeError("SANITY FAIL: center offset produced no gated target matches")
        center_median = float(np.median(center_errors))
        center_rate = len(center_errors) / max(1, len(center_rows))
        print(f"Center sanity: median={center_median:.2f}m, match_rate={center_rate:.0%}")
        if center_median > float(args.center_sanity_max_m):
            raise RuntimeError(
                f"SANITY FAIL: center median {center_median:.2f}m exceeds "
                f"{float(args.center_sanity_max_m):.2f}m; do not interpret this run"
            )
        if motion_mode == "convoy":
            gaps = np.asarray([float(row["target_distance_m"]) for row in center_rows])
            ego_speeds = np.asarray([float(row["ego_speed_mps"]) for row in center_rows])
            target_speeds = np.asarray([float(row["target_speed_mps"]) for row in center_rows])
            speed_delta = np.abs(ego_speeds - target_speeds)
            print(
                f"Convoy sanity: gap={np.mean(gaps):.2f}±{np.std(gaps):.2f}m, "
                f"ego_speed={np.mean(ego_speeds):.2f}m/s, "
                f"target_speed={np.mean(target_speeds):.2f}m/s"
            )
            if (
                np.max(np.abs(gaps - EGO_DISTANCE_M)) > 1.0
                or np.mean(ego_speeds) < 0.8 * float(args.convoy_speed_mps)
                or np.mean(speed_delta) > 0.25
            ):
                raise RuntimeError("SANITY FAIL: convoy did not maintain its commanded speed/gap")
        print("SANITY PASS: frame, coordinate, and center-error checks passed")
    finally:
        destroy_actor(target)
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
