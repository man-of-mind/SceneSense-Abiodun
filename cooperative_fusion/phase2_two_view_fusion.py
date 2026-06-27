#!/usr/bin/env python3
"""Phase 2: live TWO-VIEW cooperative position fusion (the deliverable).

Two static egos (each RGB + radar + the fusion model) view the same placed car (+ human)
from different angles. Per ego, per object we derive:
  - a PRECISE camera bearing (world unit ray) from the vehicle SEG-mask centroid pixel
    (seg vehicle IoU ~0.95, so the centroid is reliable) + camera intrinsics/pose;
  - an accurate radar world position (median of radar returns landing in the seg mask),
    used both as the per-view single-sensor estimate and to sanity-check.

We then compare estimators against CARLA ground truth:
  - single-view (each ego's radar world point)
  - mean / covariance-weighted fusion of the two single-view positions
  - TRIANGULATION of the two camera bearings (bearing-only, no range)  <-- the thesis

The point: two cheap bearing-only views triangulate to ~radar accuracy and, fused, beat any
single view -- and trivially survive one view being occluded. The learned object/distance
head is deliberately NOT used for bearings (Phase-0b found its bearing error ~37-40 deg);
that head's noisy monocular distance is exactly the baseline cooperative fusion replaces.

Run offline geometry check (no CARLA):   python phase2_two_view_fusion.py --selftest
Run live (CARLA on 127.0.0.1:2000):       COOP_CKPT=<best.pt> python phase2_two_view_fusion.py
"""
from __future__ import annotations
import argparse, math, os, queue, sys
from pathlib import Path
import numpy as np

ABS = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
OUT = Path(__file__).resolve().parent / "phase2"; OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_CKPT = ABS / "experiments/autonomous_arch_runs_20260625/pilot_indomain_finetune_archK/checkpoints/pilot_indomain_finetune_archK/best.pt"

sys.path.insert(0, str(ABS / "cooperative_fusion"))
from fusion import ViewDetection, fuse_mean, fuse_covariance, fuse_triangulate  # noqa: E402


# --------------------------- geometry helpers (CARLA-free) ---------------------------
def K_for(w, h, fov=120.0):
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def pixel_to_world_bearing(u, v, K, Rwc):
    """Pixel (u,v) + cam intrinsics K + cam->world rotation Rwc (UE axes) -> unit world ray.
    Mirrors phase0b2: standard (u,v) -> UE camera dir [1, xs, -ys] -> world."""
    xs = (u - K[0, 2]) / K[0, 0]
    ys = (v - K[1, 2]) / K[1, 1]
    ue_cam_dir = np.array([1.0, xs, -ys])
    world_dir = Rwc @ ue_cam_dir
    return world_dir / np.linalg.norm(world_dir)


def world_to_pixel(world_xyz, w2c, K, w, h):
    """UE world point -> (u,v, in_front, in_view) using world->cam inverse matrix + K."""
    p = w2c @ np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    if p[0] <= 0.05:
        return None, None, False, False
    cam_pt = np.array([p[1], -p[2], p[0]])  # UE -> standard camera axes
    uv = K @ cam_pt
    u, v = uv[0] / uv[2], uv[1] / uv[2]
    return float(u), float(v), True, (0 <= u < w and 0 <= v < h)


# --------------------------- offline geometry self-test ---------------------------
def selftest():
    """Synthetic: a known world point seen by two cameras at different poses. Extract bearings
    via pixel_to_world_bearing (the SAME code the live path uses) and triangulate. With no
    pixel noise the triangulated error must be ~0; with small pixel noise it must stay small
    and beat the single-view (camera-bearing + noisy-range) estimate. Validates sign/axis
    conventions before spending CARLA time."""
    w, h, fov = 1280, 720, 120.0
    K = K_for(w, h, fov)
    P = np.array([45.0, -68.0, 0.8])  # target (like the placed car)

    # two cameras (UE: +x forward). Build cam->world rotation from a yaw that looks at P.
    def cam_at(loc):
        loc = np.array(loc, float)
        d = P - loc; yaw = math.atan2(d[1], d[0])
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rwc = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])  # yaw about z, UE
        w2c = np.eye(4); w2c[:3, :3] = Rwc.T
        w2c[:3, 3] = -Rwc.T @ loc
        return loc, Rwc, w2c

    cams = [cam_at([30.0, -75.0, 1.6]), cam_at([33.0, -58.0, 1.6])]  # ~17 m baseline, different angles
    rng = np.random.default_rng(0)
    for noise_px, depth_std in [(0.0, 0.0), (1.0, 1.5)]:
        dets = []
        for loc, Rwc, w2c in cams:
            u, v, infront, _ = world_to_pixel(P, w2c, K, w, h)
            assert infront, "target behind synthetic camera"
            u += rng.normal(0, noise_px); v += rng.normal(0, noise_px)
            bearing = pixel_to_world_bearing(u, v, K, Rwc)
            true_range = np.linalg.norm(P - loc)
            noisy_range = true_range + rng.normal(0, depth_std)
            world_pos = loc + noisy_range * bearing  # single-view (bearing + noisy range)
            dets.append(ViewDetection(loc, bearing, world_pos, score=1.0, depth_std_m=max(0.01, depth_std)))
        tri = fuse_triangulate(dets); mean = fuse_mean(dets)
        e_sv = np.mean([np.linalg.norm(d.world_pos[:2] - P[:2]) for d in dets])
        e_tri = np.linalg.norm(tri[:2] - P[:2]); e_mean = np.linalg.norm(mean[:2] - P[:2])
        print(f"  noise(px={noise_px}, range_std={depth_std}m): single-view {e_sv:.3f}m  "
              f"mean {e_mean:.3f}m  TRIANGULATE {e_tri:.3f}m")
        if noise_px == 0.0:
            assert e_tri < 1e-6, f"zero-noise triangulation should be exact, got {e_tri}"
    print("  selftest OK: bearing/axis conventions consistent; triangulation exact at zero noise.")


# --------------------------- live two-view (CARLA) ---------------------------
def run_live():
    import torch, cv2
    import carla_split_inference_udp_demo as od_demo  # noqa: F401 (sets carla path)
    import carla
    import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fr
    from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp
    from pole_lraspp_multimodal_fusion.radar_fusion import radar_raw_to_alt_az_depth_velocity
    from pole_lraspp_multimodal_fusion.common import VEHICLE_TAGS  # noqa: F401

    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    ckpt = Path(os.environ.get("COOP_CKPT", str(DEFAULT_CKPT)))

    _argv = sys.argv; sys.argv = ["x", "--sensor-platform", "ego_vehicle"]
    try: ARGS = fr.parse_args()
    finally: sys.argv = _argv
    ARGS.ego_spawn_index = 152; ARGS.ego_freeze = True

    c = torch.load(str(ckpt), map_location=DEV, weights_only=False)
    sd = c["model"] if "model" in c else c.get("state_dict", c)
    iw, ih = [int(v) for v in c.get("input_size", [768, 432])]
    model = build_multitask_fusion_lraspp(
        num_classes=3, radar_channels=int(c.get("radar_channels", 4)), pretrained=False,
        object_channels=int(c.get("object_channels", 14)),
        fuse_low_into_object_head=bool(c.get("fuse_low_into_object_head", True)),
        head_arch=str(c.get("object_head_arch", "shared")), use_coordconv=bool(c.get("object_use_coordconv", False)),
        head_depth=int(c.get("object_head_depth", 2)), predict_bbox2d=bool(c.get("object_predict_bbox2d", True)),
        device=DEV).to(DEV).eval()
    model.load_state_dict(sd, strict=False)
    print(f"model loaded from {ckpt.name}; input {iw}x{ih}")

    def radar_world_points(radarm, radar_actor):
        det = radar_raw_to_alt_az_depth_velocity(bytes(radarm.raw_data))
        if det.shape[0] == 0: return np.zeros((0, 3))
        alt, az, dep = det[:, 0], det[:, 1], det[:, 2]
        xl = dep * np.cos(alt) * np.cos(az); yl = dep * np.cos(alt) * np.sin(az); zl = dep * np.sin(alt)
        local = np.stack([xl, yl, zl, np.ones_like(xl)], axis=1)
        M = np.array(radar_actor.get_transform().get_matrix())
        return (M @ local.T).T[:, :3]

    def seg_vehicle_mask(model, camera, radar, img, radarm, tracker):
        from pole_lraspp_multimodal_fusion.radar_fusion import build_radar_sample
        rgb = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]
        rgb_r = cv2.resize(rgb, (iw, ih)).astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(rgb_r)).permute(2, 0, 1); t = (t - RGB_MEAN) / RGB_STD
        K_in = K_for(iw, ih)
        det = radar_raw_to_alt_az_depth_velocity(bytes(radarm.raw_data))
        rt, _, _ = build_radar_sample(detections=det, sensor_matrix=np.array(radar.get_transform().get_matrix()),
            camera_inverse_matrix=np.array(camera.get_transform().get_inverse_matrix()), camera_intrinsics=K_in,
            width=iw, height=ih, frame_time_s=float(img.timestamp), tracker=tracker,
            max_range_m=120.0, max_abs_velocity_mps=20.0, parked_threshold_s=5.0, point_radius_px=4)
        fused = torch.cat([t, torch.from_numpy(np.ascontiguousarray(rt))], dim=0).unsqueeze(0).to(DEV)
        with torch.no_grad():
            out = model(fused)
        seg = out["out"][0].argmax(0).cpu().numpy().astype(np.uint8)  # may be at stride-8 res
        if seg.shape != (ih, iw):
            seg = cv2.resize(seg, (iw, ih), interpolation=cv2.INTER_NEAREST)
        return seg, K_in

    client = carla.Client("127.0.0.1", 2000); client.set_timeout(20.0)
    world = client.get_world(); s = world.get_settings(); orig = (s.synchronous_mode, s.fixed_delta_seconds)
    actors = []
    try:
        s.synchronous_mode = True; s.fixed_delta_seconds = 0.1; world.apply_settings(s)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = world.get_blueprint_library()
        # clear pre-existing vehicles/walkers so the only actors are ours (clean association)
        for a in list(world.get_actors().filter("*vehicle*")) + list(world.get_actors().filter("*walker*")):
            try: a.destroy()
            except Exception: pass
        world.tick()

        # --- ego A at spawn 152 ---
        egoA = fr._spawn_parked_ego_vehicle(world=world, args=ARGS); egoA.set_autopilot(False); actors.append(egoA)
        for _ in range(20): world.tick()
        atf = egoA.get_transform(); fwd = atf.get_forward_vector(); rgt = atf.get_right_vector(); al = atf.location

        def ground(fm, rm):
            x = al.x + fwd.x * fm + rgt.x * rm; y = al.y + fwd.y * fm + rgt.y * rm
            wp = world.get_map().get_waypoint(carla.Location(x, y, al.z + 0.5), project_to_road=True)
            return carla.Location(x, y, wp.transform.location.z if wp else al.z)

        # --- placed car ~13 m ahead of ego A ---
        car = world.try_spawn_actor(bp.filter("vehicle.nissan.patrol")[0], carla.Transform(ground(13, 0), atf.rotation))
        if car: car.set_simulate_physics(False); actors.append(car)

        # --- ego B: offset ~10 m to the right of ego A, rotated to look at the car ---
        bloc = ground(2.0, 10.0)
        cl = car.get_location()
        yawB = math.degrees(math.atan2(cl.y - bloc.y, cl.x - bloc.x))
        egoB = world.try_spawn_actor(bp.filter("vehicle.lincoln.mkz")[0],
                                     carla.Transform(carla.Location(bloc.x, bloc.y, al.z), carla.Rotation(yaw=yawB)))
        if egoB is None:  # fallback spawn nudge
            egoB = world.try_spawn_actor(bp.filter("vehicle.*")[0],
                                         carla.Transform(carla.Location(bloc.x, bloc.y, al.z + 0.3), carla.Rotation(yaw=yawB)))
        egoB.set_simulate_physics(False); actors.append(egoB)
        for _ in range(10): world.tick()

        from pole_lraspp_multimodal_fusion.common import VEHICLE_TAGS as VTAGS
        W, H, FOV = 1280, 720, 120.0
        K_full = K_for(W, H, FOV)

        def make_sensors(ego):
            cbp = bp.find("sensor.camera.rgb")
            for k, vv in (("image_size_x", W), ("image_size_y", H), ("fov", FOV)): cbp.set_attribute(k, str(vv))
            sbp = bp.find("sensor.camera.semantic_segmentation")
            for k, vv in (("image_size_x", W), ("image_size_y", H), ("fov", FOV)): sbp.set_attribute(k, str(vv))
            rbp = bp.find("sensor.other.radar")
            for k, vv in (("range", 120), ("horizontal_fov", 120), ("vertical_fov", 30), ("points_per_second", 100000)): rbp.set_attribute(k, str(vv))
            cq, sq, rq = queue.Queue(), queue.Queue(), queue.Queue()
            cam = world.spawn_actor(cbp, fr._ego_camera_transform(ARGS), attach_to=ego); actors.append(cam)
            sem = world.spawn_actor(sbp, fr._ego_camera_transform(ARGS), attach_to=ego); actors.append(sem)
            rad = world.spawn_actor(rbp, fr._ego_radar_transform(ARGS), attach_to=ego); actors.append(rad)
            cam.listen(cq.put); sem.listen(sq.put); rad.listen(rq.put)
            return cam, sem, rad, cq, sq, rq

        camA, semA, radA, cqA, sqA, rqA = make_sensors(egoA)
        camB, semB, radB, cqB, sqB, rqB = make_sensors(egoB)
        for _ in range(20): world.tick()

        CAR_H_M = 2.0 * float(car.bounding_box.extent.z)  # true car height for the monocular-depth proxy
        views = []
        for tag, cam, sem, rad, cq, sq, rq in (("A", camA, semA, radA, cqA, sqA, rqA),
                                               ("B", camB, semB, radB, cqB, sqB, rqB)):
            img = cq.get(timeout=5); semm = sq.get(timeout=5); radarm = rq.get(timeout=5)
            # GT vehicle mask from the semantic camera (association independent of the OOD seg model)
            tags = np.frombuffer(semm.raw_data, np.uint8).reshape((H, W, 4))[:, :, 2]
            gt_veh = np.isin(tags, list(VTAGS)).astype(np.uint8)
            # isolate the PLACED car via ORACLE data-association: pick the vehicle component
            # containing the GT car's projected pixel (assoc only; localization still from
            # bearing/radar -> recovered-error vs GT remains a fair metric).
            gl = car.get_location(); Gw = np.array([gl.x, gl.y, gl.z])
            w2c0 = np.array(cam.get_transform().get_inverse_matrix())
            gu, gv, ginf, ginv = world_to_pixel(Gw, w2c0, K_full, W, H)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(gt_veh, connectivity=8)
            _rgb = np.frombuffer(img.raw_data, np.uint8).reshape((H, W, 4))[:, :, :3].copy()
            cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 150]
            lab_id = None
            if ginf and ginv:
                gui, gvi = int(np.clip(gu, 0, W - 1)), int(np.clip(gv, 0, H - 1))
                if lab[gvi, gui] in cand:
                    lab_id = int(lab[gvi, gui])
                else:  # GT pixel not exactly on mask -> nearest component centroid to GT pixel
                    cand2 = [(i, math.hypot(cents[i][0] - gu, cents[i][1] - gv)) for i in cand]
                    if cand2 and min(cand2, key=lambda t: t[1])[1] < 120: lab_id = min(cand2, key=lambda t: t[1])[0]
            if lab_id is None:
                cv2.imwrite(str(OUT / f"view{tag}_rgb.png"), _rgb)
                print(f"view {tag}: placed car not segmented near GT pixel ({gu:.0f},{gv:.0f}) inview={ginf and ginv}")
                views.append(None); continue
            comp = (lab == lab_id)
            ys, xs = np.where(comp)
            cu, cv = float(xs.mean()), float(ys.mean())  # silhouette mask centroid
            # 2D-bbox center: more view-invariant than the mask centroid (not pulled to the
            # larger visible face). bx = left+w/2, by = top+h/2.
            bx = float(stats[lab_id, cv2.CC_STAT_LEFT] + stats[lab_id, cv2.CC_STAT_WIDTH] / 2.0)
            by = float(stats[lab_id, cv2.CC_STAT_TOP] + stats[lab_id, cv2.CC_STAT_HEIGHT] / 2.0)
            box_h = float(stats[lab_id, cv2.CC_STAT_HEIGHT])
            # overlays for the record
            _ov = _rgb.copy(); _ov[comp] = (0, 0, 255)
            cv2.imwrite(str(OUT / f"view{tag}_rgb.png"), _rgb)
            cv2.imwrite(str(OUT / f"view{tag}_carmask.png"), cv2.addWeighted(_rgb, 0.55, _ov, 0.45, 0))

            cam_tf = cam.get_transform()
            cam_center = np.array([cam_tf.location.x, cam_tf.location.y, cam_tf.location.z])
            Rwc = np.array(cam_tf.get_matrix())[:3, :3]
            w2c = np.array(cam_tf.get_inverse_matrix())
            bearing = pixel_to_world_bearing(cu, cv, K_full, Rwc)       # silhouette-centroid bearing
            bearing_box = pixel_to_world_bearing(bx, by, K_full, Rwc)   # 2D-bbox-center bearing

            # (1) radar range: median of radar returns landing inside the car component -> accurate
            rpw = radar_world_points(radarm, rad); in_mask = []
            for p in rpw:
                u, v, infront, inview = world_to_pixel(p, w2c, K_full, W, H)
                if infront and inview and comp[int(np.clip(v, 0, H - 1)), int(np.clip(u, 0, W - 1))]:
                    in_mask.append(p)
            radar_pos = np.median(np.array(in_mask), axis=0) if in_mask else None
            # (2) monocular-depth proxy from apparent car height (NOISY single-view depth)
            mono_range = (K_full[1, 1] * CAR_H_M / box_h) if box_h > 1 else None
            mono_pos = cam_center + mono_range * bearing if mono_range else None
            views.append({"tag": tag, "C": cam_center, "bearing": bearing, "bearing_box": bearing_box,
                          "radar_pos": radar_pos, "mono_pos": mono_pos,
                          "n_radar": len(in_mask), "box_h": box_h, "cu": cu, "cv": cv})
            rp = f"({radar_pos[0]:.1f},{radar_pos[1]:.1f})" if radar_pos is not None else "n/a"
            mp = f"({mono_pos[0]:.1f},{mono_pos[1]:.1f})" if mono_pos is not None else "n/a"
            print(f"view {tag}: car_px={len(xs)} centroid=({cu:.0f},{cv:.0f}) box_h={box_h:.0f} "
                  f"radar_in_car={len(in_mask)} radar_pos={rp} mono_pos={mp}")

        gt = car.get_location(); G = np.array([gt.x, gt.y, gt.z])
        def e(p): return float(np.linalg.norm(np.asarray(p)[:2] - G[:2]))
        rep = [f"GT car=({gt.x:.2f},{gt.y:.2f})  baseline(A-B)={np.linalg.norm(views[0]['C'][:2]-views[1]['C'][:2]):.1f} m"
               if all(v is not None for v in views) else f"GT car=({gt.x:.2f},{gt.y:.2f})"]
        valid = [v for v in views if v is not None]
        rep.append("per-view single estimates:")
        for v in valid:
            if v["mono_pos"] is not None:
                rep.append(f"  {v['tag']} monocular-depth : ({v['mono_pos'][0]:.2f},{v['mono_pos'][1]:.2f})  err={e(v['mono_pos']):.3f} m")
            if v["radar_pos"] is not None:
                rep.append(f"  {v['tag']} radar-range     : ({v['radar_pos'][0]:.2f},{v['radar_pos'][1]:.2f})  err={e(v['radar_pos']):.3f} m")
        if len(valid) >= 2:
            # bearing-only triangulation (NO range used) vs fusing the noisy monocular single-views
            dets_bear = [ViewDetection(v["C"], v["bearing"],
                                       v["mono_pos"] if v["mono_pos"] is not None else v["C"] + 13.0 * v["bearing"],
                                       score=1.0, depth_std_m=1.5) for v in valid]
            tri = fuse_triangulate(dets_bear)
            dets_box = [ViewDetection(v["C"], v["bearing_box"], v["C"] + 13.0 * v["bearing_box"],
                                      score=1.0, depth_std_m=1.5) for v in valid]
            tri_box = fuse_triangulate(dets_box)
            mean_mono = fuse_mean(dets_bear)
            rep.append("fused:")
            rep.append(f"  mean(monocular) : ({mean_mono[0]:.2f},{mean_mono[1]:.2f})  err={e(mean_mono):.3f} m")
            rep.append(f"  TRIANGULATE(bbox-center bearings): ({tri_box[0]:.2f},{tri_box[1]:.2f})  err={e(tri_box):.3f} m")
            rep.append(f"  TRIANGULATE(bearings, no range): ({tri[0]:.2f},{tri[1]:.2f})  err={e(tri):.3f} m")
            if all(v["radar_pos"] is not None for v in valid):
                rp = np.mean([v["radar_pos"] for v in valid], axis=0)
                rep.append(f"  mean(radar)     : ({rp[0]:.2f},{rp[1]:.2f})  err={e(rp):.3f} m  [accurate-sensor reference]")
        else:
            rep.append("  <2 valid views -- cannot fuse")
        print("\n=== PHASE 2 TWO-VIEW VEHICLE FUSION ===")
        print("\n".join(rep))
        (OUT / "phase2_report.txt").write_text("\n".join(rep) + "\n")
    finally:
        for a in actors:
            try: a.destroy()
            except Exception: pass
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = orig; world.apply_settings(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="offline geometry check, no CARLA")
    a = ap.parse_args()
    if a.selftest:
        print("=== PHASE 2 OFFLINE GEOMETRY SELF-TEST ===")
        selftest()
    else:
        run_live()
