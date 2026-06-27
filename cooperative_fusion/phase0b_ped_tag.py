#!/usr/bin/env python3
"""Keystone diagnostic: what semantic tag does CARLA 0.10 assign to a pedestrian?

Spawns a static ego + semantic camera and a PROPERLY-spawned walker (physics on, with an
AI controller, on the nav mesh) in view. Reports (a) the walker actor's declared
semantic_tags, and (b) the tag the semantic camera actually renders at the walker pixels.
Assumes CARLA running on 127.0.0.1:2000.
"""
from __future__ import annotations

import math
import queue
import sys
from pathlib import Path

import numpy as np

import carla_split_inference_udp_demo as od_demo  # noqa: F401
import carla  # noqa: E402
import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fusion_runtime  # noqa: E402

OUT = Path(__file__).resolve().parent / "phase0b"
OUT.mkdir(parents=True, exist_ok=True)
_argv = sys.argv
sys.argv = ["x", "--sensor-platform", "ego_vehicle"]
try:
    ARGS = fusion_runtime.parse_args()
finally:
    sys.argv = _argv
ARGS.ego_spawn_index = 152
ARGS.ego_freeze = True
W, H, FOV = 1280, 720, 120.0


def K():
    f = W / (2 * math.tan(math.radians(FOV) / 2))
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def project(loc, cam):
    w2c = np.array(cam.get_transform().get_inverse_matrix())
    p = w2c @ np.array([loc.x, loc.y, loc.z, 1.0])
    if p[0] <= 0.05:
        return None
    uv = K() @ np.array([p[1], -p[2], p[0]])
    return int(uv[0] / uv[2]), int(uv[1] / uv[2])


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(20.0)
    world = client.get_world()
    s = world.get_settings(); orig = (s.synchronous_mode, s.fixed_delta_seconds)
    actors = []
    try:
        s.synchronous_mode = True; s.fixed_delta_seconds = 0.1; world.apply_settings(s)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        ego = fusion_runtime._spawn_parked_ego_vehicle(world=world, args=ARGS); ego.set_autopilot(False); actors.append(ego)
        for _ in range(20): world.tick()
        etf = ego.get_transform(); fwd = etf.get_forward_vector(); egl = etf.location
        bp = world.get_blueprint_library()
        sem_bp = bp.find("sensor.camera.semantic_segmentation")
        for k, v in (("image_size_x", W), ("image_size_y", H), ("fov", FOV)):
            sem_bp.set_attribute(k, str(v))
        semq = queue.Queue()
        sem = world.spawn_actor(sem_bp, fusion_runtime._ego_camera_transform(ARGS), attach_to=ego); actors.append(sem)
        sem.listen(semq.put)
        # Also test the instance-segmentation camera (often labels walkers where the
        # plain semantic camera does not; R channel = semantic tag).
        inst_bp = bp.find("sensor.camera.instance_segmentation")
        for k, v in (("image_size_x", W), ("image_size_y", H), ("fov", FOV)):
            inst_bp.set_attribute(k, str(v))
        instq = queue.Queue()
        inst = world.spawn_actor(inst_bp, fusion_runtime._ego_camera_transform(ARGS), attach_to=ego); actors.append(inst)
        inst.listen(instq.put)

        # Proper walker spawn: on nav mesh, physics on, with AI controller.
        wbp = bp.filter("walker.pedestrian.*")[0]
        # place ~8 m ahead, snap to a navigable point near there
        target = carla.Location(egl.x + fwd.x * 8.0, egl.y + fwd.y * 8.0, egl.z + 1.0)
        nav = world.get_random_location_from_navigation()
        # prefer the nav point closest to our target so it is roughly in view
        walker = None
        for cand in [target, nav]:
            walker = world.try_spawn_actor(wbp, carla.Transform(cand, etf.rotation))
            if walker: break
        report = []
        if walker is None:
            report.append("WALKER FAILED TO SPAWN")
        else:
            actors.append(walker)
            world.tick()
            report.append(f"walker.semantic_tags (actor-declared) = {list(walker.semantic_tags)}")
            try:
                cbp = bp.find("controller.ai.walker")
                ctrl = world.spawn_actor(cbp, carla.Transform(), attach_to=walker); actors.append(ctrl)
                world.tick(); ctrl.start(); ctrl.set_max_speed(0.0)  # stand in place
            except Exception as exc:
                report.append(f"(controller spawn note: {exc})")
            for _ in range(15): world.tick()
            seg = semq.get(timeout=5.0)
            tags = np.frombuffer(seg.raw_data, dtype=np.uint8).reshape((H, W, 4))[:, :, 2]
            wl = walker.get_location()
            uv = project(wl, sem)
            report.append(f"walker world=({wl.x:.1f},{wl.y:.1f},{wl.z:.1f}) pixel={uv}")
            if uv and 0 <= uv[0] < W and 0 <= uv[1] < H:
                box = tags[max(0, uv[1] - 40):uv[1] + 60, max(0, uv[0] - 25):uv[0] + 25]
                u, c = np.unique(box, return_counts=True)
                report.append("tags rendered in walker region (tag:count):")
                for i in np.argsort(-c)[:6]:
                    report.append(f"    tag {int(u[i]):3d}: {int(c[i]):6d}")
            # global histogram too
            u, c = np.unique(tags, return_counts=True)
            report.append("global tag histogram top: " + ", ".join(f"{int(u[i])}:{int(c[i])}" for i in np.argsort(-c)[:10]))
            # instance-seg camera: R channel = semantic tag; check the same walker region
            try:
                ins = instq.get(timeout=5.0)
                itags = np.frombuffer(ins.raw_data, dtype=np.uint8).reshape((H, W, 4))[:, :, 2]
                if uv and 0 <= uv[0] < W and 0 <= uv[1] < H:
                    ibox = itags[max(0, uv[1] - 40):uv[1] + 60, max(0, uv[0] - 25):uv[0] + 25]
                    iu, ic = np.unique(ibox, return_counts=True)
                    report.append("INSTANCE-cam tags in walker region (tag:count):")
                    for i in np.argsort(-ic)[:6]:
                        report.append(f"    tag {int(iu[i]):3d}: {int(ic[i]):6d}")
                iu, ic = np.unique(itags, return_counts=True)
                report.append("INSTANCE-cam global top: " + ", ".join(f"{int(iu[i])}:{int(ic[i])}" for i in np.argsort(-ic)[:10]))
            except Exception as exc:
                report.append(f"instance-seg check failed: {exc}")
        print("\n=== PEDESTRIAN TAG DIAGNOSTIC ===")
        print("\n".join(report))
        (OUT / "ped_tag_report.txt").write_text("\n".join(report) + "\n")
    finally:
        for a in actors:
            try: a.destroy()
            except Exception: pass
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = orig; world.apply_settings(s)


if __name__ == "__main__":
    main()
