#!/usr/bin/env python3
"""Phase 2b: full two-view cooperative fusion — position + DIMENSIONS + pedestrian + baseline sweep.

Extends phase2 with the complete world-frame 3D-box deliverable:
  - POSITION  : triangulate two camera bearings (B2 sweeps the baseline 4/8/14 m).
  - DIMENSIONS: read the model's per-view 3D size regression at the car center and fuse them by
                viewing geometry (front view -> width+height, side view -> length+height). B0 live.
  - PEDESTRIAN: radar-cluster association (seg-person is weak), triangulate two views. B1.
  - MULTI-FRAME averaging over N static frames to cut bearing pixel noise. B3.
Validated against CARLA GT (position + bounding-box extent). Oracle data-association (GT picks which
blob/cluster is the target); localization/dimension error vs GT stays a fair metric.

Run: COOP_CKPT=<gated best.pt> python cooperative_fusion/phase2b_full_fusion.py
"""
from __future__ import annotations
import math, os, queue, sys
from pathlib import Path
import numpy as np

ABS = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
OUT = Path(__file__).resolve().parent / "phase2b"; OUT.mkdir(parents=True, exist_ok=True)
GATED = ABS / "experiments/autonomous_arch_runs_20260625/det_rangegated40_archK/checkpoints/det_rangegated40_archK/best.pt"
sys.path.insert(0, str(ABS / "cooperative_fusion"))
from fusion import ViewDetection, fuse_mean, fuse_triangulate, fuse_dimensions  # noqa: E402
from phase2_two_view_fusion import K_for, pixel_to_world_bearing, world_to_pixel  # noqa: E402

import torch, cv2  # noqa: E402
import carla_split_inference_udp_demo as od_demo  # noqa: F401,E402
import carla  # noqa: E402
import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fr  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.radar_fusion import (  # noqa: E402
    radar_raw_to_alt_az_depth_velocity, build_radar_sample, StationaryTrackAccumulator)
from pole_lraspp_multimodal_fusion.object_targets import REG_DIMS, REG_YAW  # noqa: E402
from pole_lraspp_multimodal_fusion.common import VEHICLE_TAGS as VTAGS  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
W, H, FOV = 1280, 720, 120.0
K_FULL = K_for(W, H, FOV)
CKPT = Path(os.environ.get("COOP_CKPT", str(GATED)))

_argv = sys.argv; sys.argv = ["x", "--sensor-platform", "ego_vehicle"]
try: ARGS = fr.parse_args()
finally: sys.argv = _argv
ARGS.ego_spawn_index = 152; ARGS.ego_freeze = True


def load_model():
    c = torch.load(str(CKPT), map_location=DEV, weights_only=False)
    sd = c["model"] if "model" in c else c.get("state_dict", c)
    iw, ih = [int(v) for v in c.get("input_size", [768, 432])]
    m = build_multitask_fusion_lraspp(
        num_classes=3, radar_channels=int(c.get("radar_channels", 4)), pretrained=False,
        object_channels=int(c.get("object_channels", 14)),
        fuse_low_into_object_head=bool(c.get("fuse_low_into_object_head", True)),
        head_arch=str(c.get("object_head_arch", "shared")), use_coordconv=bool(c.get("object_use_coordconv", False)),
        head_depth=int(c.get("object_head_depth", 2)), predict_bbox2d=bool(c.get("object_predict_bbox2d", True)),
        device=DEV).to(DEV).eval()
    m.load_state_dict(sd, strict=False)
    print(f"model {CKPT.name}; input {iw}x{ih}")
    return m, (iw, ih)


def radar_world_points(radarm, rad):
    det = radar_raw_to_alt_az_depth_velocity(bytes(radarm.raw_data))
    if det.shape[0] == 0: return np.zeros((0, 3))
    alt, az, dep = det[:, 0], det[:, 1], det[:, 2]
    xl = dep * np.cos(alt) * np.cos(az); yl = dep * np.cos(alt) * np.sin(az); zl = dep * np.sin(alt)
    local = np.stack([xl, yl, zl, np.ones_like(xl)], 1)
    return (np.array(rad.get_transform().get_matrix()) @ local.T).T[:, :3]


def infer_object(model, in_wh, cam, rad, img, radarm, tracker):
    iw, ih = in_wh
    rgb = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]
    rgb = cv2.resize(rgb, (iw, ih)).astype(np.float32) / 255.0
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1); t = (t - RGB_MEAN) / RGB_STD
    K_in = K_for(iw, ih)
    det = radar_raw_to_alt_az_depth_velocity(bytes(radarm.raw_data))
    rt, _, _ = build_radar_sample(detections=det, sensor_matrix=np.array(rad.get_transform().get_matrix()),
        camera_inverse_matrix=np.array(cam.get_transform().get_inverse_matrix()), camera_intrinsics=K_in,
        width=iw, height=ih, frame_time_s=float(img.timestamp), tracker=tracker,
        max_range_m=120.0, max_abs_velocity_mps=20.0, parked_threshold_s=5.0, point_radius_px=4)
    fused = torch.cat([t, torch.from_numpy(np.ascontiguousarray(rt))], 0).unsqueeze(0).to(DEV)
    with torch.no_grad():
        out = model(fused)
    return out["object"][0].detach().cpu().numpy(), (iw, ih)


def read_dims_yaw(obj_out, cu, cv, in_wh):
    """Sample the regression head's 3D size + yaw at the (full-res) center pixel."""
    iw, ih = in_wh
    nseg = obj_out.shape[0] - 12  # 12 reg channels (bbox2d)
    regs = obj_out[nseg:]
    ox = int(np.clip(round(cu * iw / W), 0, iw - 1)); oy = int(np.clip(round(cv * ih / H), 0, ih - 1))
    dims = np.maximum(regs[REG_DIMS, oy, ox], 0.01)              # (L, W, H) meters, object frame
    ys, yc = regs[REG_YAW, oy, ox]
    yaw = math.atan2(float(ys), float(yc))
    return np.array(dims, float), yaw


def gt_vehicle_centroid(sem_raw, cam, gt_loc):
    """Oracle-associate the placed car: vehicle component containing its projected pixel.
    Returns (centroid_u, centroid_v, bbox_center_u, bbox_center_v) or None."""
    tags = np.frombuffer(sem_raw, np.uint8).reshape((H, W, 4))[:, :, 2]
    veh = np.isin(tags, list(VTAGS)).astype(np.uint8)
    w2c = np.array(cam.get_transform().get_inverse_matrix())
    gu, gv, inf, inv = world_to_pixel(np.array([gt_loc.x, gt_loc.y, gt_loc.z]), w2c, K_FULL, W, H)
    if not (inf and inv): return None
    n, lab, stats, cents = cv2.connectedComponentsWithStats(veh, 8)
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 150]
    gui, gvi = int(np.clip(gu, 0, W - 1)), int(np.clip(gv, 0, H - 1))
    lab_id = int(lab[gvi, gui]) if lab[gvi, gui] in cand else None
    if lab_id is None:
        c2 = [(i, math.hypot(cents[i][0] - gu, cents[i][1] - gv)) for i in cand]
        if c2 and min(c2, key=lambda t: t[1])[1] < 120: lab_id = min(c2, key=lambda t: t[1])[0]
    if lab_id is None: return None
    bx = stats[lab_id, cv2.CC_STAT_LEFT] + stats[lab_id, cv2.CC_STAT_WIDTH] / 2.0
    by = stats[lab_id, cv2.CC_STAT_TOP] + stats[lab_id, cv2.CC_STAT_HEIGHT] / 2.0
    bw = float(stats[lab_id, cv2.CC_STAT_WIDTH]); bh = float(stats[lab_id, cv2.CC_STAT_HEIGHT])
    return float(cents[lab_id][0]), float(cents[lab_id][1]), float(bx), float(by), bw, bh


def segbox_extent(bw_px, bh_px, range_m):
    """Metric extent of the visible face from the seg 2D box at the given range:
    horizontal extent (-> width OR length depending on view angle) and height."""
    horiz = bw_px * range_m / K_FULL[0, 0]
    height = bh_px * range_m / K_FULL[1, 1]
    return float(horiz), float(height)


def radar_cluster_pos(radarm, rad, gt_loc, assoc_r=2.5):
    """Oracle-associate a pedestrian: radar world points within assoc_r of the GT location."""
    rpw = radar_world_points(radarm, rad)
    if rpw.shape[0] == 0: return None, 0
    g = np.array([gt_loc.x, gt_loc.y, gt_loc.z])
    near = rpw[np.linalg.norm(rpw[:, :2] - g[:2], axis=1) <= assoc_r]
    if near.shape[0] == 0: return None, 0
    return np.median(near, axis=0), int(near.shape[0])


def make_sensors(world, bp, ego):
    cbp = bp.find("sensor.camera.rgb"); sbp = bp.find("sensor.camera.semantic_segmentation")
    for b in (cbp, sbp):
        for k, vv in (("image_size_x", W), ("image_size_y", H), ("fov", FOV)): b.set_attribute(k, str(vv))
    rbp = bp.find("sensor.other.radar")
    for k, vv in (("range", 120), ("horizontal_fov", 120), ("vertical_fov", 30), ("points_per_second", 100000)): rbp.set_attribute(k, str(vv))
    cq, sq, rq = queue.Queue(), queue.Queue(), queue.Queue()
    cam = world.spawn_actor(cbp, fr._ego_camera_transform(ARGS), attach_to=ego)
    sem = world.spawn_actor(sbp, fr._ego_camera_transform(ARGS), attach_to=ego)
    rad = world.spawn_actor(rbp, fr._ego_radar_transform(ARGS), attach_to=ego)
    cam.listen(cq.put); sem.listen(sq.put); rad.listen(rq.put)
    return dict(cam=cam, sem=sem, rad=rad, cq=cq, sq=sq, rq=rq, actors=[cam, sem, rad])


def run_baseline(world, bp, model, in_wh, baseline_m, n_frames=5):
    """Spawn ego A + car + pedestrian + ego B at the given baseline; return fused metrics vs GT."""
    actors = []
    try:
        egoA = fr._spawn_parked_ego_vehicle(world=world, args=ARGS); egoA.set_autopilot(False); actors.append(egoA)
        for _ in range(20): world.tick()
        atf = egoA.get_transform(); fwd = atf.get_forward_vector(); rgt = atf.get_right_vector(); al = atf.location
        def ground(fm, rm):
            x = al.x + fwd.x * fm + rgt.x * rm; y = al.y + fwd.y * fm + rgt.y * rm
            wp = world.get_map().get_waypoint(carla.Location(x, y, al.z + 0.5), project_to_road=True)
            return carla.Location(x, y, wp.transform.location.z if wp else al.z)
        car = world.try_spawn_actor(bp.filter("vehicle.nissan.patrol")[0], carla.Transform(ground(13, 0), atf.rotation))
        if car: car.set_simulate_physics(False); actors.append(car)
        # pedestrian ~8 m ahead + 3 m right
        pl = ground(8, 3.0); ped = world.try_spawn_actor(bp.filter("walker.pedestrian.*")[0],
                                                          carla.Transform(carla.Location(pl.x, pl.y, pl.z + 1.2), atf.rotation))
        if ped:
            ped.set_simulate_physics(False); hh = float(ped.bounding_box.extent.z)
            ped.set_transform(carla.Transform(carla.Location(pl.x, pl.y, pl.z + hh + 0.05), atf.rotation)); actors.append(ped)
        # ego B offset to the right by baseline, yawed to look at the car. Robust spawn: try a few
        # forward nudges / heights so a close baseline doesn't collide with ego A.
        cl = car.get_location(); egoB = None
        for fwd_m in (2.0, 4.0, 0.0, 6.0):
            bloc = ground(fwd_m, baseline_m)
            yawB = math.degrees(math.atan2(cl.y - bloc.y, cl.x - bloc.x))
            for dz in (0.3, 0.6, 1.0):
                egoB = world.try_spawn_actor(bp.filter("vehicle.lincoln.mkz")[0],
                                             carla.Transform(carla.Location(bloc.x, bloc.y, al.z + dz), carla.Rotation(yaw=yawB)))
                if egoB is not None: break
            if egoB is not None: break
        if egoB is None:
            print(f"  baseline {baseline_m:.0f} m: could not place ego B (collision/off-road) -> skip")
            return {"baseline_m": baseline_m, "skip": "egoB spawn failed"}
        egoB.set_simulate_physics(False); actors.append(egoB)
        for _ in range(10): world.tick()
        sA = make_sensors(world, bp, egoA); sB = make_sensors(world, bp, egoB)
        actors += sA["actors"] + sB["actors"]
        for _ in range(20): world.tick()

        gt_car = car.get_location(); gt_ped = ped.get_location() if ped else None
        Gc = np.array([gt_car.x, gt_car.y]); Gp = np.array([gt_ped.x, gt_ped.y]) if gt_ped else None
        # GT car dims (length, width, height) = 2*extent
        ext = car.bounding_box.extent; gt_dims = np.array([2 * ext.x, 2 * ext.y, 2 * ext.z])

        views = {}
        for tag, s in (("A", sA), ("B", sB)):
            cam, sem, rad = s["cam"], s["sem"], s["rad"]
            tr = StationaryTrackAccumulator()
            car_bearings, car_dims, car_yaws, car_radar = [], [], [], []
            car_box, car_rng, ped_bearings, ped_radar = [], [], [], []
            cam_c = None
            for _ in range(n_frames):
                world.tick()
                img = s["cq"].get(timeout=5); semm = s["sq"].get(timeout=5); radarm = s["rq"].get(timeout=5)
                ctf = cam.get_transform(); cam_c = np.array([ctf.location.x, ctf.location.y, ctf.location.z])
                Rwc = np.array(ctf.get_matrix())[:3, :3]
                obj_out, owh = infer_object(model, in_wh, cam, rad, img, radarm, tr)
                gc = gt_vehicle_centroid(bytes(semm.raw_data), cam, gt_car)
                if gc:
                    cu, cv, bxu, bxv, bw, bh = gc
                    car_bearings.append(pixel_to_world_bearing(bxu, bxv, K_FULL, Rwc))
                    d, yw = read_dims_yaw(obj_out, cu, cv, owh)
                    car_dims.append(d); car_yaws.append(yw)
                    car_box.append((bw, bh))
                    car_rng.append(float(np.linalg.norm(np.array([gt_car.x, gt_car.y, gt_car.z]) - cam_c)))
                    rp, _ = radar_cluster_pos(radarm, rad, gt_car, assoc_r=3.5)
                    if rp is not None: car_radar.append(rp)
                if gt_ped is not None:
                    rp, npts = radar_cluster_pos(radarm, rad, gt_ped, assoc_r=2.0)
                    if rp is not None and npts >= 2:
                        ped_radar.append(rp)
                        ped_bearings.append((rp - cam_c) / max(1e-9, np.linalg.norm(rp - cam_c)))
            v = dict(cam_c=cam_c)
            if car_bearings:
                v["car_bearing"] = np.mean(car_bearings, 0)
                v["car_dims"] = np.mean(car_dims, 0); v["car_yaw"] = float(np.mean(car_yaws))
                v["car_radar"] = np.mean(car_radar, 0) if car_radar else None
                v["car_box"] = np.mean(car_box, 0); v["car_rng"] = float(np.mean(car_rng))
            if ped_bearings:
                v["ped_bearing"] = np.mean(ped_bearings, 0); v["ped_radar"] = np.mean(ped_radar, 0)
            views[tag] = v

        out = {"baseline_m": baseline_m, "gt_car": Gc.tolist(), "gt_dims": gt_dims.tolist()}
        A, B = views.get("A", {}), views.get("B", {})
        # --- car position ---
        if "car_bearing" in A and "car_bearing" in B:
            dA = ViewDetection(A["cam_c"], A["car_bearing"], A.get("car_radar") if A.get("car_radar") is not None else A["cam_c"] + 13 * A["car_bearing"], dims=A["car_dims"], yaw=A["car_yaw"])
            dB = ViewDetection(B["cam_c"], B["car_bearing"], B.get("car_radar") if B.get("car_radar") is not None else B["cam_c"] + 13 * B["car_bearing"], dims=B["car_dims"], yaw=B["car_yaw"])
            tri = fuse_triangulate([dA, dB])
            out["car_tri_err"] = float(np.linalg.norm(tri[:2] - Gc))
            out["car_svA_err"] = float(np.linalg.norm(dA.world_pos[:2] - Gc))
            out["car_svB_err"] = float(np.linalg.norm(dB.world_pos[:2] - Gc))
            # --- car dimensions: (i) regression-head fusion, (ii) seg-2D-box fusion ---
            fdims = fuse_dimensions([dA, dB])
            out["dims_fused_reg"] = fdims.tolist()
            out["dims_err_reg"] = float(np.mean(np.abs(fdims - gt_dims)))
            out["dims_err_regA"] = float(np.mean(np.abs(A["car_dims"] - gt_dims)))
            out["dims_err_regB"] = float(np.mean(np.abs(B["car_dims"] - gt_dims)))
            # seg-2D-box: each view's horizontal box extent -> width OR length by view angle.
            yaw_c = math.radians(car.get_transform().rotation.yaw)
            fwd2 = np.array([math.cos(yaw_c), math.sin(yaw_c)]); lat2 = np.array([-math.sin(yaw_c), math.cos(yaw_c)])
            widths, lengths, heights = [], [], []
            for vv in (A, B):
                if "car_box" not in vv: continue
                bw, bh = vv["car_box"]; horiz, height = segbox_extent(bw, bh, vv["car_rng"])
                heights.append(height)
                r = np.asarray(vv["car_bearing"], float)[:2]; r = r / max(1e-9, np.linalg.norm(r))
                if abs(float(r @ fwd2)) >= abs(float(r @ lat2)):   # front/back view -> sees WIDTH
                    widths.append(horiz)
                else:                                              # side view -> sees LENGTH
                    lengths.append(horiz)
            segdims = np.array([np.mean(lengths) if lengths else np.nan,
                                np.mean(widths) if widths else np.nan,
                                np.mean(heights) if heights else np.nan])
            out["dims_segbox"] = [None if np.isnan(x) else float(x) for x in segdims]
            obs = ~np.isnan(segdims)   # only score axes actually observed by some view
            if obs.any():
                out["dims_err_segbox_obs"] = float(np.mean(np.abs(segdims[obs] - gt_dims[obs])))
                out["dims_segbox_axes"] = ["L" if obs[0] else "-", "W" if obs[1] else "-", "H" if obs[2] else "-"]
        # --- pedestrian position ---
        if Gp is not None and "ped_bearing" in A and "ped_bearing" in B:
            pA = ViewDetection(A["cam_c"], A["ped_bearing"], A["ped_radar"])
            pB = ViewDetection(B["cam_c"], B["ped_bearing"], B["ped_radar"])
            tri = fuse_triangulate([pA, pB])
            out["ped_tri_err"] = float(np.linalg.norm(tri[:2] - Gp))
            out["ped_svA_err"] = float(np.linalg.norm(pA.world_pos[:2] - Gp))
            out["ped_svB_err"] = float(np.linalg.norm(pB.world_pos[:2] - Gp))
        return out
    finally:
        for a in actors:
            try: a.destroy()
            except Exception: pass


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(20.0)
    world = client.get_world(); s = world.get_settings(); orig = (s.synchronous_mode, s.fixed_delta_seconds)
    try:
        s.synchronous_mode = True; s.fixed_delta_seconds = 0.1; world.apply_settings(s)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = world.get_blueprint_library()
        for a in list(world.get_actors().filter("*vehicle*")) + list(world.get_actors().filter("*walker*")):
            try: a.destroy()
            except Exception: pass
        world.tick()
        model, in_wh = load_model()
        results = []
        for base in (4.0, 8.0, 14.0):
            r = run_baseline(world, bp, model, in_wh, base, n_frames=5)
            results.append(r)
            print(f"\n=== baseline {base:.0f} m ===")
            if "car_tri_err" in r:
                print(f"  CAR  pos: single-view A {r['car_svA_err']:.2f} / B {r['car_svB_err']:.2f}  -> TRIANGULATE {r['car_tri_err']:.2f} m")
                print(f"  CAR  dims(reg-head): per-view A {r['dims_err_regA']:.2f} / B {r['dims_err_regB']:.2f}  -> FUSED {r['dims_err_reg']:.2f} m")
                if "dims_err_segbox_obs" in r:
                    print(f"  CAR  dims(seg-2D-box): {r['dims_err_segbox_obs']:.2f} m on axes {r['dims_segbox_axes']}  "
                          f"(LWH {[None if x is None else round(x,2) for x in r['dims_segbox']]} vs GT {['%.2f'%x for x in r['gt_dims']]})")
            if "ped_tri_err" in r:
                print(f"  PED  pos: single-view A {r['ped_svA_err']:.2f} / B {r['ped_svB_err']:.2f}  -> TRIANGULATE {r['ped_tri_err']:.2f} m")
        import json
        (OUT / "phase2b_results.json").write_text(json.dumps(results, indent=2))
        print(f"\nsaved {OUT}/phase2b_results.json")
    finally:
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = orig; world.apply_settings(s)


if __name__ == "__main__":
    main()
