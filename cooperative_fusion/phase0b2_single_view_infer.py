#!/usr/bin/env python3
"""Phase 0b-2: single-view LIVE archK inference + world-position calibration vs CARLA GT.

Spawns a static ego + car + human (fixed poses), runs the archK fusion model in-process on
the live RGB+radar frame, decodes objects, and compares predicted world position (and the
per-detection bearing) to the CARLA ground-truth actor positions. Foundation for the
two-view triangulation fusion. Assumes CARLA on 127.0.0.1:2000.
"""
from __future__ import annotations
import math, queue, sys
from pathlib import Path
import numpy as np
import torch

import carla_split_inference_udp_demo as od_demo  # noqa: F401
import carla  # noqa: E402
import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fusion_runtime  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import decode_objects  # noqa: E402
from pole_lraspp_multimodal_fusion.radar_fusion import (  # noqa: E402
    radar_raw_to_alt_az_depth_velocity, build_radar_sample, StationaryTrackAccumulator,
)

ABS = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
import os
ARCHK = Path(os.environ.get("COOP_CKPT",
    str(ABS / "experiments/autonomous_arch_runs_20260625/archK_giou_adaptiveradius/checkpoints/archK_giou_adaptiveradius/best.pt")))
OUT = Path(__file__).resolve().parent / "phase0b"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_argv = sys.argv; sys.argv = ["x", "--sensor-platform", "ego_vehicle"]
try: ARGS = fusion_runtime.parse_args()
finally: sys.argv = _argv
ARGS.ego_spawn_index = 152; ARGS.ego_freeze = True


def load_model():
    c = torch.load(str(ARCHK), map_location=DEV, weights_only=False)
    sd = c["model"] if "model" in c else c.get("state_dict", c)
    m = build_multitask_fusion_lraspp(
        num_classes=3, radar_channels=int(c.get("radar_channels", 4)), pretrained=False,
        object_channels=int(c.get("object_channels", 14)),
        fuse_low_into_object_head=bool(c.get("fuse_low_into_object_head", True)),
        head_arch=str(c.get("object_head_arch", "shared")),
        use_coordconv=bool(c.get("object_use_coordconv", False)),
        head_depth=int(c.get("object_head_depth", 2)),
        predict_bbox2d=bool(c.get("object_predict_bbox2d", True)),
        device=DEV).to(DEV).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    m.object_class_names = list(c.get("object_class_names", ["vehicle", "person"]))
    m.object_predict_bbox2d = bool(c.get("object_predict_bbox2d", True))
    iw, ih = [int(v) for v in c.get("input_size", [768, 432])]
    print(f"archK loaded (missing {len(miss)}, unexpected {len(unexp)}); input={iw}x{ih}, bbox2d={m.object_predict_bbox2d}")
    return m, (iw, ih)


def K_for(w, h, fov=120.0):
    f = w / (2 * math.tan(math.radians(fov) / 2))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def infer(model, in_wh, camera, radar, img, radarm, tracker):
    iw, ih = in_wh
    import cv2
    rgb = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]
    rgb = cv2.resize(rgb, (iw, ih)).astype(np.float32) / 255.0
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    t = (t - RGB_MEAN) / RGB_STD
    K_in = K_for(iw, ih)
    det = radar_raw_to_alt_az_depth_velocity(bytes(radarm.raw_data))
    radar_tensor, _, _ = build_radar_sample(
        detections=det, sensor_matrix=np.array(radar.get_transform().get_matrix()),
        camera_inverse_matrix=np.array(camera.get_transform().get_inverse_matrix()),
        camera_intrinsics=K_in, width=iw, height=ih, frame_time_s=float(img.timestamp),
        tracker=tracker, max_range_m=120.0, max_abs_velocity_mps=20.0,
        parked_threshold_s=5.0, point_radius_px=4)
    fused = torch.cat([t, torch.from_numpy(np.ascontiguousarray(radar_tensor))], dim=0).unsqueeze(0).to(DEV)
    with torch.no_grad():
        out = model(fused)
    preds = decode_objects(out["object"], camera_matrix=np.array(camera.get_transform().get_matrix()),
                           topk=80, score_threshold=float(os.environ.get("COOP_THR", "0.20")), nms_radius_px=2,
                           object_class_names=model.object_class_names,
                           predict_bbox2d=model.object_predict_bbox2d)
    # bearing from heatmap-peak pixel (precise), in world frame
    cam_tf = camera.get_transform()
    cam_center = np.array([cam_tf.location.x, cam_tf.location.y, cam_tf.location.z])
    Rwc = np.array(cam_tf.get_matrix())[:3, :3]  # cam->world rotation (UE axes)
    for p in preds:
        u, v = float(p["center_x_px"]), float(p["center_y_px"])
        xs = (u - K_in[0, 2]) / K_in[0, 0]
        ys = (v - K_in[1, 2]) / K_in[1, 1]
        ue_cam_dir = np.array([1.0, xs, -ys])           # standard(u,v)->UE camera dir
        world_dir = Rwc @ ue_cam_dir
        p["bearing"] = (world_dir / np.linalg.norm(world_dir)).tolist()
        p["cam_center"] = cam_center.tolist()
    return preds


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(20.0)
    world = client.get_world(); s = world.get_settings(); orig = (s.synchronous_mode, s.fixed_delta_seconds)
    actors = []
    try:
        s.synchronous_mode = True; s.fixed_delta_seconds = 0.1; world.apply_settings(s)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        model, in_wh = load_model()
        ego = fusion_runtime._spawn_parked_ego_vehicle(world=world, args=ARGS); ego.set_autopilot(False); actors.append(ego)
        for _ in range(20): world.tick()
        etf = ego.get_transform(); fwd = etf.get_forward_vector(); rgt = etf.get_right_vector(); egl = etf.location

        def ground(fm, rm):
            x = egl.x + fwd.x * fm + rgt.x * rm; y = egl.y + fwd.y * fm + rgt.y * rm
            wp = world.get_map().get_waypoint(carla.Location(x, y, egl.z + 0.5), project_to_road=True)
            return carla.Location(x, y, wp.transform.location.z if wp else egl.z)
        bp = world.get_blueprint_library()
        car = world.try_spawn_actor(bp.filter("vehicle.nissan.patrol")[0], carla.Transform(ground(12, 0), etf.rotation))
        if car: car.set_simulate_physics(False); actors.append(car)
        hl = ground(8, 2.5); human = world.try_spawn_actor(bp.filter("walker.pedestrian.*")[0],
                                                            carla.Transform(carla.Location(hl.x, hl.y, hl.z + 1.2), etf.rotation))
        if human:
            human.set_simulate_physics(False)
            hh = float(human.bounding_box.extent.z)
            human.set_transform(carla.Transform(carla.Location(hl.x, hl.y, hl.z + hh + 0.05), etf.rotation)); actors.append(human)

        cbp = bp.find("sensor.camera.rgb")
        for k, vv in (("image_size_x", 1280), ("image_size_y", 720), ("fov", 120)): cbp.set_attribute(k, str(vv))
        rbp = bp.find("sensor.other.radar")
        for k, vv in (("range", 120), ("horizontal_fov", 120), ("vertical_fov", 30), ("points_per_second", 100000)): rbp.set_attribute(k, str(vv))
        cq, rq = queue.Queue(), queue.Queue()
        cam = world.spawn_actor(cbp, fusion_runtime._ego_camera_transform(ARGS), attach_to=ego); actors.append(cam)
        rad = world.spawn_actor(rbp, fusion_runtime._ego_radar_transform(ARGS), attach_to=ego); actors.append(rad)
        cam.listen(cq.put); rad.listen(rq.put)
        for _ in range(20): world.tick()
        img, radarm = cq.get(timeout=5), rq.get(timeout=5)
        preds = infer(model, in_wh, cam, rad, img, radarm, StationaryTrackAccumulator())

        # GT world positions
        gts = {"vehicle": car.get_location() if car else None, "person": human.get_location() if human else None}
        rep = [f"detections: {len(preds)} (thr 0.20)"]
        for cls in ("vehicle", "person"):
            g = gts[cls]
            cands = [p for p in preds if p.get("class_name") == cls]
            if g is None:
                rep.append(f"{cls}: no GT actor"); continue
            if not cands:
                rep.append(f"{cls}: GT=({g.x:.1f},{g.y:.1f}) -- NOT DETECTED by archK"); continue
            best = min(cands, key=lambda p: math.hypot(p["world_x"] - g.x, p["world_y"] - g.y))
            xy_err = math.hypot(best["world_x"] - g.x, best["world_y"] - g.y)
            # bearing sanity: angle between predicted bearing and true (GT - cam_center)
            cc = np.array(best["cam_center"]); br = np.array(best["bearing"])
            tdir = np.array([g.x, g.y, g.z]) - cc; tdir = tdir / np.linalg.norm(tdir)
            ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(br, tdir))))))
            rep.append(f"{cls}: GT=({g.x:.1f},{g.y:.1f}) pred=({best['world_x']:.1f},{best['world_y']:.1f}) "
                       f"xy_err={xy_err:.2f}m score={best['score']:.2f} bearing_err={ang:.1f}deg")
        print("\n=== PHASE 0b-2 SINGLE-VIEW LIVE INFERENCE ===")
        print("\n".join(rep))
        (OUT / "phase0b2_report.txt").write_text("\n".join(rep) + "\n")
    finally:
        for a in actors:
            try: a.destroy()
            except Exception: pass
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = orig; world.apply_settings(s)


if __name__ == "__main__":
    main()
