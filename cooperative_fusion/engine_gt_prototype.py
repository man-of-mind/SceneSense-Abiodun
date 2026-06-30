#!/usr/bin/env python3
"""Engine-GT prototype: build pedestrian/vehicle masks from CARLA ENGINE LOGIC (actor 3D boxes)
instead of the (pedestrian-broken) semantic camera, and compare fidelity.

Produces one figure with, for a controlled car+pedestrian scene:
  RGB | broken semantic-camera mask | projected-3D-box mask | depth-refined silhouette
plus a numeric check of whether semantic LIDAR tags pedestrians.

"CARLA's logic" = the engine knows each actor's exact 3D bbox+pose. Turning that into pixels:
  (b) project the 8 box corners -> convex hull -> fill   [coarse, this IS engine logic directly]
  (c) + keep hull pixels whose DEPTH ~ the actor's distance [my refinement -> tight silhouette + occlusion]
Assumes CARLA on 127.0.0.1:2000.
"""
from __future__ import annotations
import math, queue, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import carla_split_inference_udp_demo as od_demo  # noqa: F401
import carla  # noqa: E402
import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fr  # noqa: E402
from pole_lraspp_multimodal_fusion.common import PERSON_TAGS, VEHICLE_TAGS  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"; OUT.mkdir(parents=True, exist_ok=True)
W, H, FOV = 1280, 720, 120.0
_argv = sys.argv; sys.argv = ["x", "--sensor-platform", "ego_vehicle"]
try: ARGS = fr.parse_args()
finally: sys.argv = _argv
ARGS.ego_spawn_index = 152; ARGS.ego_freeze = True


def K_for(w, h, fov):
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def project(world_xyz, w2c, K):
    p = w2c @ np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    if p[0] <= 0.05: return None, None, False
    cam = np.array([p[1], -p[2], p[0]])
    uv = K @ cam
    return uv[0] / uv[2], uv[1] / uv[2], True


def decode_depth_m(depth_img):
    a = np.frombuffer(depth_img.raw_data, np.uint8).reshape((H, W, 4)).astype(np.float32)
    B, G, R = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    norm = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return 1000.0 * norm  # meters


def hull_from_vertices(verts, cam, K):
    """Projected convex hull (2D) of 8 world-space box vertices + camera-distance range.
    Works for dynamic actors AND static environment objects (just pass their world vertices)."""
    w2c = np.array(cam.get_transform().get_inverse_matrix())
    cam_loc = cam.get_transform().location; cc = np.array([cam_loc.x, cam_loc.y, cam_loc.z])
    pts = []; dists = []
    for v in verts:
        u, vv, infront = project([v.x, v.y, v.z], w2c, K)
        if infront and -300 < u < W + 300 and -300 < vv < H + 300:
            pts.append([u, vv])
        dists.append(math.sqrt((v.x - cc[0]) ** 2 + (v.y - cc[1]) ** 2 + (v.z - cc[2]) ** 2))
    if len(pts) < 3: return None, None, 1e9
    return np.array(pts, np.float32), (min(dists), max(dists)), min(dists)


def main():
    import cv2
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
        ego = fr._spawn_parked_ego_vehicle(world=world, args=ARGS); ego.set_autopilot(False); actors.append(ego)
        for _ in range(20): world.tick()
        etf = ego.get_transform(); fwd = etf.get_forward_vector(); rgt = etf.get_right_vector(); egl = etf.location
        def ground(fm, rm):
            x = egl.x + fwd.x * fm + rgt.x * rm; y = egl.y + fwd.y * fm + rgt.y * rm
            wp = world.get_map().get_waypoint(carla.Location(x, y, egl.z + 0.5), project_to_road=True)
            return carla.Location(x, y, wp.transform.location.z if wp else egl.z)
        pl = ground(7, -2.0); ped = world.try_spawn_actor(bp.filter("walker.pedestrian.*")[0],
                                                           carla.Transform(carla.Location(pl.x, pl.y, pl.z + 1.2), etf.rotation))
        if ped:
            ped.set_simulate_physics(False); hh = float(ped.bounding_box.extent.z)
            ped.set_transform(carla.Transform(carla.Location(pl.x, pl.y, pl.z + hh + 0.05), etf.rotation)); actors.append(ped)

        # sensors: rgb, semantic, depth (shared cam pose), + semantic lidar
        def cam_bp(kind):
            b = bp.find(kind)
            for k, vv in (("image_size_x", W), ("image_size_y", H), ("fov", FOV)): b.set_attribute(k, str(vv))
            return b
        ctf = fr._ego_camera_transform(ARGS)
        qr, qs, qd = queue.Queue(), queue.Queue(), queue.Queue()
        rgbc = world.spawn_actor(cam_bp("sensor.camera.rgb"), ctf, attach_to=ego); rgbc.listen(qr.put); actors.append(rgbc)
        semc = world.spawn_actor(cam_bp("sensor.camera.semantic_segmentation"), ctf, attach_to=ego); semc.listen(qs.put); actors.append(semc)
        depc = world.spawn_actor(cam_bp("sensor.camera.depth"), ctf, attach_to=ego); depc.listen(qd.put); actors.append(depc)
        # semantic lidar
        lbp = bp.find("sensor.lidar.ray_cast_semantic")
        for k, vv in (("range", "60"), ("rotation_frequency", "10"), ("channels", "64"), ("points_per_second", "600000")): lbp.set_attribute(k, str(vv))
        ql = queue.Queue(); lid = world.spawn_actor(lbp, carla.Transform(carla.Location(z=2.4)), attach_to=ego); lid.listen(ql.put); actors.append(lid)
        for _ in range(20): world.tick()
        img = qr.get(timeout=5); semm = qs.get(timeout=5); depm = qd.get(timeout=5); lidm = ql.get(timeout=5)

        K = K_for(W, H, FOV)
        rgb = np.frombuffer(img.raw_data, np.uint8).reshape((H, W, 4))[:, :, :3][:, :, ::-1]
        tags = np.frombuffer(semm.raw_data, np.uint8).reshape((H, W, 4))[:, :, 2]
        depth = decode_depth_m(depm)

        # (a) broken semantic-camera mask
        sem_mask = np.zeros((H, W), np.uint8)
        sem_mask[np.isin(tags, list(VEHICLE_TAGS))] = 1; sem_mask[np.isin(tags, list(PERSON_TAGS))] = 2
        # Build the target list from ENGINE LOGIC:
        #   vehicles = static environment objects (Car/Truck/Bus) + any dynamic vehicle actors
        #   pedestrian = the dynamic walker actor
        # Each target -> its 8 world-space box vertices + class.
        targets = []  # (vertices, class)
        for lab in (carla.CityObjectLabel.Car, carla.CityObjectLabel.Truck, carla.CityObjectLabel.Bus):
            for obj in world.get_environment_objects(lab):
                targets.append((obj.bounding_box.get_world_vertices(carla.Transform()), 1))
        if ped is not None:
            targets.append((ped.bounding_box.get_world_vertices(ped.get_transform()), 2))

        # (b) projected-box mask + (c) depth-refined silhouette
        box_mask = np.zeros((H, W), np.uint8); ref_mask = np.zeros((H, W), np.uint8)
        for verts, cls in targets:
            hull, drange, mindist = hull_from_vertices(verts, rgbc, K)
            if hull is None or mindist > 45.0: continue   # operating-range gate
            region = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(region, cv2.convexHull(hull).astype(np.int32), 1)
            box_mask[region == 1] = cls
            near, far = drange
            keep = (region == 1) & (depth >= near - 0.4) & (depth <= far + 0.4)
            ref_mask[keep] = cls
        # semantic-lidar pedestrian check
        ldata = np.frombuffer(lidm.raw_data, dtype=np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                              ('cos', np.float32), ('idx', np.uint32), ('tag', np.uint32)]))
        ltags, lcnt = np.unique(ldata['tag'], return_counts=True)
        ped_lidar = int(sum(c for t, c in zip(ltags, lcnt) if int(t) in PERSON_TAGS))

        # report
        def cnt(m, c): return int((m == c).sum())
        print("Pedestrian pixels by GT method:")
        print(f"  semantic camera (broken): {cnt(sem_mask,2)}")
        print(f"  projected 3D box        : {cnt(box_mask,2)}")
        print(f"  depth-refined silhouette: {cnt(ref_mask,2)}")
        print(f"  semantic-LIDAR pedestrian points: {ped_lidar}  (tags present: {[int(t) for t in ltags]})")
        print(f"Vehicle pixels: semantic-cam {cnt(sem_mask,1)} | box {cnt(box_mask,1)} | depth-refined {cnt(ref_mask,1)}")

        def colorize(m):
            o = np.zeros((H, W, 3), np.uint8); o[m == 1] = (60, 120, 255); o[m == 2] = (255, 40, 200); return o
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        for a in ax.ravel(): a.axis("off")
        ax[0, 0].imshow(rgb); ax[0, 0].set_title("RGB", fontsize=13)
        ax[0, 1].imshow(rgb); ax[0, 1].imshow(colorize(sem_mask), alpha=0.55)
        ax[0, 1].set_title(f"Semantic CAMERA (broken): person px={cnt(sem_mask,2)}", fontsize=13)
        ax[1, 0].imshow(rgb); ax[1, 0].imshow(colorize(box_mask), alpha=0.55)
        ax[1, 0].set_title(f"ENGINE GT: projected 3D box (person px={cnt(box_mask,2)})", fontsize=13)
        ax[1, 1].imshow(rgb); ax[1, 1].imshow(colorize(ref_mask), alpha=0.55)
        ax[1, 1].set_title(f"ENGINE GT: box + depth-refined silhouette (person px={cnt(ref_mask,2)})", fontsize=13)
        fig.suptitle("Engine-logic GT (actor 3D box, optionally depth-refined) vs the broken semantic camera",
                     fontsize=14, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / "engine_gt_prototype.png", dpi=110, bbox_inches="tight")
        print(f"\nsaved {OUT/'engine_gt_prototype.png'}")
    finally:
        for a in actors:
            try: a.destroy()
            except Exception: pass
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = orig; world.apply_settings(s)


if __name__ == "__main__":
    main()
