#!/usr/bin/env python3
"""Phase 0b-1: controlled static scene health check.

Spawns a static ego (RGB + semantic + radar, geometry matching the archK model) and
places one car + one pedestrian at known poses in front of it. Saves frames for visual
inspection and runs programmatic sanity checks (ego/car/human placement, sensors live,
objects in view). No model yet -- this only confirms CARLA is behaving before calibration.

Assumes CARLA is already running on 127.0.0.1:2000.
"""
from __future__ import annotations

import math
import queue
import sys
from pathlib import Path

import numpy as np

import carla_split_inference_udp_demo as od_demo  # noqa: F401  (sets up carla import path)
import carla  # noqa: E402
import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fusion_runtime  # noqa: E402
from pole_lraspp_multimodal_fusion.common import carla_semantic_tags_to_training_mask  # noqa: E402

OUT = Path(__file__).resolve().parent / "phase0b"
OUT.mkdir(parents=True, exist_ok=True)

# Build a complete, correctly-typed args namespace from the runtime's own parser
# (so ego-spawn + sensor geometry exactly match the archK model), then override.
_argv = sys.argv
sys.argv = ["phase0b", "--sensor-platform", "ego_vehicle"]
try:
    ARGS = fusion_runtime.parse_args()
finally:
    sys.argv = _argv
ARGS.ego_spawn_index = 152
ARGS.ego_freeze = True  # static ego
CAM_W, CAM_H, CAM_FOV = 1280, 720, 120.0


def intrinsics(w, h, fov):
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def project_to_image(world_loc, camera, K):
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    p = w2c @ np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
    if p[0] <= 0.05:  # behind camera (UE camera: +x forward)
        return None, False
    cam_pt = np.array([p[1], -p[2], p[0]])  # UE -> standard camera axes
    uv = K @ cam_pt
    u, v = uv[0] / uv[2], uv[1] / uv[2]
    return (float(u), float(v)), (0 <= u < CAM_W and 0 <= v < CAM_H)


def main():
    report = []
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    settings = world.get_settings()
    orig = (settings.synchronous_mode, settings.fixed_delta_seconds)
    actors = []
    try:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)  # bright, standard lighting

        ego = fusion_runtime._spawn_parked_ego_vehicle(world=world, args=ARGS)
        ego.set_autopilot(False)
        actors.append(ego)
        for _ in range(20):
            world.tick()
        ego_tf = ego.get_transform()
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        egl = ego_tf.location

        def ground_loc(fwd_m, right_m):
            x = egl.x + fwd.x * fwd_m + right.x * right_m
            y = egl.y + fwd.y * fwd_m + right.y * right_m
            loc = carla.Location(x=x, y=y, z=egl.z + 0.5)
            wp = world.get_map().get_waypoint(loc, project_to_road=True)
            z = wp.transform.location.z if wp else egl.z
            return carla.Location(x=x, y=y, z=z)

        bp_lib = world.get_blueprint_library()

        def pick(patterns):
            for p in patterns:
                bps = bp_lib.filter(p)
                if bps:
                    return bps[0]
            raise RuntimeError(f"No blueprint for any of {patterns}")

        # Car ~12 m ahead; freeze physics so it stays exactly placed (static scene).
        car_loc = ground_loc(12.0, 0.0)
        car_bp = pick(["vehicle.nissan.patrol", "vehicle.dodge.charger", "vehicle.mini.cooper", "vehicle.*"])
        car = world.try_spawn_actor(car_bp, carla.Transform(car_loc, ego_tf.rotation))
        if car:
            car.set_simulate_physics(False)
            actors.append(car)
        # Human ~8 m ahead + 2.5 m right. Walkers fall through driving lanes in sync mode,
        # so freeze physics and stand it on the ground (origin = ground + half-height).
        human_loc = ground_loc(8.0, 2.5)
        ped_bp = pick(["walker.pedestrian.*"])
        human = world.try_spawn_actor(ped_bp, carla.Transform(carla.Location(human_loc.x, human_loc.y, human_loc.z + 1.2), ego_tf.rotation))
        if human:
            human.set_simulate_physics(False)
            half_h = float(human.bounding_box.extent.z)
            human.set_transform(carla.Transform(
                carla.Location(human_loc.x, human_loc.y, human_loc.z + half_h + 0.05),
                ego_tf.rotation))
            actors.append(human)

        # Sensors
        cam_q, sem_q, rad_q = queue.Queue(), queue.Queue(), queue.Queue()
        cbp = bp_lib.find("sensor.camera.rgb")
        cbp.set_attribute("image_size_x", str(CAM_W)); cbp.set_attribute("image_size_y", str(CAM_H)); cbp.set_attribute("fov", str(CAM_FOV))
        sbp = bp_lib.find("sensor.camera.semantic_segmentation")
        sbp.set_attribute("image_size_x", str(CAM_W)); sbp.set_attribute("image_size_y", str(CAM_H)); sbp.set_attribute("fov", str(CAM_FOV))
        rbp = bp_lib.find("sensor.other.radar")
        rbp.set_attribute("range", "120"); rbp.set_attribute("horizontal_fov", "120"); rbp.set_attribute("vertical_fov", "30"); rbp.set_attribute("points_per_second", "100000")
        cam = world.spawn_actor(cbp, fusion_runtime._ego_camera_transform(ARGS), attach_to=ego)
        sem = world.spawn_actor(sbp, fusion_runtime._ego_camera_transform(ARGS), attach_to=ego)
        rad = world.spawn_actor(rbp, fusion_runtime._ego_radar_transform(ARGS), attach_to=ego)
        actors += [cam, sem, rad]
        cam.listen(cam_q.put); sem.listen(sem_q.put); rad.listen(rad_q.put)

        for _ in range(20):
            world.tick()
        img = cam_q.get(timeout=5.0)
        seg = sem_q.get(timeout=5.0)
        radm = rad_q.get(timeout=5.0)
        img.save_to_disk(str(OUT / "scene_rgb.png"))
        seg.save_to_disk(str(OUT / "scene_semantic.png"), carla.ColorConverter.CityScapesPalette)

        # ---- sanity checks ----
        K = intrinsics(CAM_W, CAM_H, CAM_FOV)
        rgb = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))[:, :, :3]
        report.append(f"ego: z={egl.z:.2f} pitch={ego_tf.rotation.pitch:.1f} roll={ego_tf.rotation.roll:.1f} yaw={ego_tf.rotation.yaw:.1f}")
        on_road = world.get_map().get_waypoint(egl, project_to_road=False)
        report.append(f"ego on driving lane: {on_road is not None}")
        report.append(f"camera frame: mean={rgb.mean():.1f} std={rgb.std():.1f} (black if ~0)")
        npts = len([d for d in radm])
        report.append(f"radar returns this frame: {npts}")
        for name, act in [("car", car), ("human", human)]:
            if act is None:
                report.append(f"{name}: FAILED TO SPAWN"); continue
            loc = act.get_location()
            dist = egl.distance(loc)
            uv, inview = project_to_image(loc, cam, K)
            report.append(f"{name}: world=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f}) dist_from_ego={dist:.2f}m "
                          f"pixel={('%.0f,%.0f'%uv) if uv else 'behind'} in_view={inview}")
        # Semantic-class health via the ACTUAL training pipeline (raw tag IDs -> 3-class
        # mask {0 bg, 1 vehicle, 2 person}), not the CityScapes palette.
        try:
            tags = np.frombuffer(seg.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))[:, :, 2]
            from pole_lraspp_multimodal_fusion.common import PERSON_TAGS, VEHICLE_TAGS
            uvals, ucnts = np.unique(tags, return_counts=True)
            order = np.argsort(-ucnts)
            report.append("raw semantic tag histogram (tag:count [->class]):")
            for i in order[:14]:
                t = int(uvals[i])
                cls = "PERSON" if t in PERSON_TAGS else ("VEHICLE" if t in VEHICLE_TAGS else "bg")
                report.append(f"    tag {t:3d}: {int(ucnts[i]):7d}  -> {cls}")
            # Tags in the human's projected region -> reveals the true pedestrian tag id.
            hu, hv = 787, 434
            sub = tags[max(0, hv - 10):hv + 90, max(0, hu - 40):hu + 40]
            su, sc = np.unique(sub, return_counts=True)
            report.append("tags in human region (tag:count):")
            for i in np.argsort(-sc)[:8]:
                report.append(f"    tag {int(su[i]):3d}: {int(sc[i]):6d}")
            tmask = np.asarray(carla_semantic_tags_to_training_mask(tags)).copy()
            # CARLA 0.10 doesn't render walker semantics -> synthesize the person mask from
            # the walker's projected 3D box (ellipse proxy).
            from pole_lraspp_multimodal_fusion.common import rasterize_person_regions
            person_boxes = []
            if human is not None:
                verts = human.bounding_box.get_world_vertices(human.get_transform())
                uvs = [project_to_image(v, cam, K)[0] for v in verts]
                uvs = [p for p in uvs if p is not None]
                if uvs:
                    xs = [p[0] for p in uvs]; ys = [p[1] for p in uvs]
                    person_boxes.append((min(xs), min(ys), max(xs), max(ys)))
            rasterize_person_regions(tmask, person_boxes, shape="ellipse")
            person_px = int((tmask == 2).sum())
            veh_px = int((tmask == 1).sum())
            report.append(f"training-mask person px: {person_px}  {'OK' if person_px > 150 else 'WARN: human not in seg!'}")
            report.append(f"training-mask vehicle px: {veh_px}")
            # Save a viewable 3-class mask (bg=black, vehicle=blue, person=red).
            from PIL import Image as _Img
            viz = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            viz[tmask == 1] = (0, 0, 255)
            viz[tmask == 2] = (255, 0, 0)
            _Img.fromarray(viz).save(OUT / "scene_trainmask.png")
        except Exception as exc:
            report.append(f"training-mask check failed: {exc}")
        with open(OUT / "scene_health_report.txt", "w") as f:
            f.write("\n".join(report) + "\n")
        print("\n=== PHASE 0b-1 SCENE HEALTH ===")
        print("\n".join(report))
        print(f"frames: {OUT}/scene_rgb.png , scene_semantic.png")
    finally:
        for a in actors:
            try: a.destroy()
            except Exception: pass
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = orig; world.apply_settings(s)


if __name__ == "__main__":
    main()
