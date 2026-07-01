#!/usr/bin/env python3
"""Deployment snapshots: run the final fusion model live, place a car + pedestrian at 10/20/30 m,
and overlay the SEG mask + 2D boxes (bounding box of each seg region) so we can see, visually, how
the model + 2D-box localization looks at each distance. Vehicle = blue, person = magenta.

COOP_CKPT defaults to the deliverable (det_stage2c_centerw4). Assumes CARLA on 127.0.0.1:2000.
"""
from __future__ import annotations
import math, os, queue, sys
from pathlib import Path
import numpy as np
import torch, cv2

ABS = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
OUT = Path(__file__).resolve().parent / "deploy_snapshots"; OUT.mkdir(parents=True, exist_ok=True)
DEFAULT = ABS / "experiments/autonomous_arch_runs_20260625/det_stage2c_centerw4/checkpoints/det_stage2c_centerw4/best.pt"
CKPT = Path(os.environ.get("COOP_CKPT", str(DEFAULT)))

import carla_split_inference_udp_demo as od_demo  # noqa: F401
import carla  # noqa: E402
import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fr  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.radar_fusion import (  # noqa: E402
    radar_raw_to_alt_az_depth_velocity, build_radar_sample, StationaryTrackAccumulator)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
W, H, FOV = 1280, 720, 120.0

_argv = sys.argv; sys.argv = ["x", "--sensor-platform", "ego_vehicle"]
try: ARGS = fr.parse_args()
finally: sys.argv = _argv
ARGS.ego_spawn_index = 80; ARGS.ego_freeze = True


def load_model():
    c = torch.load(str(CKPT), map_location=DEV, weights_only=False)
    sd = c["model"] if "model" in c else c.get("state_dict", c)
    iw, ih = [int(v) for v in c.get("input_size", [768, 432])]
    m = build_multitask_fusion_lraspp(
        num_classes=3, radar_channels=int(c.get("radar_channels", 4)), pretrained=False,
        object_channels=int(c.get("object_channels", 14)),
        fuse_low_into_object_head=bool(c.get("fuse_low_into_object_head", True)),
        head_arch=str(c.get("object_head_arch", "shared")), use_coordconv=bool(c.get("object_use_coordconv", False)),
        head_depth=int(c.get("object_head_depth", 3)), predict_bbox2d=bool(c.get("object_predict_bbox2d", True)),
        device=DEV).to(DEV).eval()
    m.load_state_dict(sd, strict=False)
    print(f"model {CKPT.name}; input {iw}x{ih}")
    return m, (iw, ih)


def infer_seg(model, in_wh, cam, rad, img, radarm, tracker):
    iw, ih = in_wh
    rgb = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]
    rgb_r = cv2.resize(rgb, (iw, ih)).astype(np.float32) / 255.0
    t = torch.from_numpy(np.ascontiguousarray(rgb_r)).permute(2, 0, 1); t = (t - RGB_MEAN) / RGB_STD
    f = w = iw / (2 * math.tan(math.radians(FOV) / 2))
    K = np.array([[f, 0, iw / 2.0], [0, f, ih / 2.0], [0, 0, 1.0]])
    det = radar_raw_to_alt_az_depth_velocity(bytes(radarm.raw_data))
    rt, _, _ = build_radar_sample(detections=det, sensor_matrix=np.array(rad.get_transform().get_matrix()),
        camera_inverse_matrix=np.array(cam.get_transform().get_inverse_matrix()), camera_intrinsics=K,
        width=iw, height=ih, frame_time_s=float(img.timestamp), tracker=tracker,
        max_range_m=120.0, max_abs_velocity_mps=20.0, parked_threshold_s=5.0, point_radius_px=4)
    fused = torch.cat([t, torch.from_numpy(np.ascontiguousarray(rt))], 0).unsqueeze(0).to(DEV)
    with torch.no_grad():
        out = model(fused)
    seg = out["out"][0].argmax(0).cpu().numpy().astype(np.uint8)
    if seg.shape != (ih, iw):
        seg = cv2.resize(seg, (iw, ih), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)


def overlay(rgb, seg, dist_m):
    vis = rgb.copy()
    ov = rgb.copy(); ov[seg == 1] = (255, 90, 0); ov[seg == 2] = (200, 0, 255)  # BGR: veh blue, person magenta
    vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
    for cls, color, name in ((1, (255, 90, 0), "vehicle"), (2, (200, 0, 255), "person")):
        m = (seg == cls).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 80: continue
            x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, name, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, f"objects placed at ~{dist_m} m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(20.0)
    world = client.get_world(); s = world.get_settings(); orig = (s.synchronous_mode, s.fixed_delta_seconds)
    actors = []
    try:
        s.synchronous_mode = True; s.fixed_delta_seconds = 0.1; world.apply_settings(s)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = world.get_blueprint_library()
        for a in list(world.get_actors().filter("*vehicle*")) + list(world.get_actors().filter("*walker*")):
            try: a.destroy()
            except Exception: pass
        world.tick()
        model, in_wh = load_model()
        ego = fr._spawn_parked_ego_vehicle(world=world, args=ARGS); ego.set_autopilot(False); actors.append(ego)
        for _ in range(20): world.tick()
        etf = ego.get_transform(); fwd = etf.get_forward_vector(); rgt = etf.get_right_vector(); egl = etf.location

        def ground(fm, rm):
            x = egl.x + fwd.x * fm + rgt.x * rm; y = egl.y + fwd.y * fm + rgt.y * rm
            wp = world.get_map().get_waypoint(carla.Location(x, y, egl.z + 0.5), project_to_road=True)
            return carla.Location(x, y, wp.transform.location.z if wp else egl.z)

        cbp = bp.find("sensor.camera.rgb")
        for k, vv in (("image_size_x", W), ("image_size_y", H), ("fov", FOV)): cbp.set_attribute(k, str(vv))
        rbp = bp.find("sensor.other.radar")
        for k, vv in (("range", 120), ("horizontal_fov", 120), ("vertical_fov", 30), ("points_per_second", 100000)): rbp.set_attribute(k, str(vv))
        cq, rq = queue.Queue(), queue.Queue()
        cam = world.spawn_actor(cbp, fr._ego_camera_transform(ARGS), attach_to=ego); cam.listen(cq.put); actors.append(cam)
        rad = world.spawn_actor(rbp, fr._ego_radar_transform(ARGS), attach_to=ego); rad.listen(rq.put); actors.append(rad)

        for dist in (10, 20, 30):
            placed = []
            car = world.try_spawn_actor(bp.filter("vehicle.nissan.patrol")[0], carla.Transform(ground(dist, -1.8), etf.rotation))
            if car: car.set_simulate_physics(False); placed.append(car)
            pl = ground(dist, 2.2); ped = world.try_spawn_actor(bp.filter("walker.pedestrian.*")[0],
                                                                 carla.Transform(carla.Location(pl.x, pl.y, pl.z + 1.2), etf.rotation))
            if ped:
                ped.set_simulate_physics(False); hh = float(ped.bounding_box.extent.z)
                ped.set_transform(carla.Transform(carla.Location(pl.x, pl.y, pl.z + hh + 0.05), etf.rotation)); placed.append(ped)
            for _ in range(15): world.tick()
            img = cq.get(timeout=5); radarm = rq.get(timeout=5)
            seg = infer_seg(model, in_wh, cam, rad, img, radarm, StationaryTrackAccumulator())
            rgb = np.frombuffer(img.raw_data, np.uint8).reshape((H, W, 4))[:, :, :3].copy()  # BGR
            vis = overlay(rgb, seg, dist)
            outp = OUT / f"snapshot_{dist}m.png"
            cv2.imwrite(str(outp), vis)
            print(f"  {dist} m: vehicle_px={int((seg==1).sum())} person_px={int((seg==2).sum())} -> {outp.name}")
            for a in placed:
                try: a.destroy()
                except Exception: pass
            world.tick()
        print(f"\nsnapshots in {OUT}")
    finally:
        for a in actors:
            try: a.destroy()
            except Exception: pass
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = orig; world.apply_settings(s)


if __name__ == "__main__":
    main()
