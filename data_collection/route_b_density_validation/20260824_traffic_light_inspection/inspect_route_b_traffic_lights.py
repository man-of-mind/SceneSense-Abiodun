#!/usr/bin/env python3
"""Read-only Town10HD_Opt traffic-light inspection against the accepted Route B.

Records original red/yellow/green durations, group identity, world location, and
distance to the Route B planned path. Nothing is modified.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import carla

ROOT = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/"
            "PythonAPI/neu_collab/abiodun")
ROUTE = ROOT / "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/route_b_lights.json")

# The two locations where the accepted 50/50 run recorded sustained ego blocks.
LONG_WAIT_XY = {"leg11_x109_y70": (109.39, 69.66), "leg16_xm73_y28": (-72.86, 27.98)}
NEAR_ROUTE_M = 30.0


def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def path_distance(px, py, path):
    best = float("inf")
    best_i = -1
    for i in range(len(path) - 1):
        d = seg_dist(px, py, path[i]["x"], path[i]["y"], path[i + 1]["x"], path[i + 1]["y"])
        if d < best:
            best, best_i = d, i
    return best, best_i


def main() -> int:
    route = json.loads(ROUTE.read_text())
    path = route["planned_path"]
    legs = route["intermediate_waypoints"]

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    world = client.load_world("Town10HD_Opt", True)
    print(f"map={world.get_map().name}", flush=True)
    world.tick() if world.get_settings().synchronous_mode else world.wait_for_tick()

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"traffic lights found: {len(lights)}", flush=True)

    # Stable group identity: lowest actor id in the group.
    group_of = {}
    for tl in lights:
        try:
            gids = sorted([int(tl.id)] + [int(o.id) for o in tl.get_group_traffic_lights()])
        except RuntimeError:
            gids = [int(tl.id)]
        group_of[int(tl.id)] = {"group_key": gids[0], "group_members": gids,
                                "group_size": len(gids)}

    records = []
    for tl in lights:
        loc = tl.get_transform().location
        d_path, seg_i = path_distance(loc.x, loc.y, path)
        d_legs = sorted(
            ((math.hypot(loc.x - w["x"], loc.y - w["y"]), i) for i, w in enumerate(legs))
        )
        waits = {k: round(math.hypot(loc.x - xy[0], loc.y - xy[1]), 2)
                 for k, xy in LONG_WAIT_XY.items()}
        try:
            opendrive_id = str(tl.get_opendrive_id())
        except RuntimeError:
            opendrive_id = ""
        try:
            pole = int(tl.get_pole_index())
        except RuntimeError:
            pole = -1
        red = round(float(tl.get_red_time()), 3)
        yellow = round(float(tl.get_yellow_time()), 3)
        green = round(float(tl.get_green_time()), 3)
        records.append({
            "actor_id": int(tl.id),
            "type_id": str(tl.type_id),
            "opendrive_id": opendrive_id,
            "pole_index": pole,
            **group_of[int(tl.id)],
            "x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2),
            "yaw_deg": round(tl.get_transform().rotation.yaw, 2),
            "red_time_s": red, "yellow_time_s": yellow, "green_time_s": green,
            "cycle_s": round(red + yellow + green, 3),
            "state_at_read": str(tl.get_state()),
            "distance_to_route_b_path_m": round(d_path, 2),
            "nearest_path_segment_index": seg_i,
            "nearest_ordered_waypoint_index": d_legs[0][1],
            "nearest_ordered_waypoint_dist_m": round(d_legs[0][0], 2),
            "distance_to_long_wait_m": waits,
            "on_route_b": d_path <= NEAR_ROUTE_M,
        })

    records.sort(key=lambda r: r["distance_to_route_b_path_m"])
    OUT.write_text(json.dumps(
        {"map": str(world.get_map().name), "route_id": route["name"],
         "near_route_threshold_m": NEAR_ROUTE_M,
         "long_wait_locations": LONG_WAIT_XY,
         "total_traffic_lights": len(records), "lights": records}, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)

    on = [r for r in records if r["on_route_b"]]
    print(f"\n=== {len(on)} lights within {NEAR_ROUTE_M} m of the Route B path ===")
    hdr = (f"{'id':>4} {'grp':>4} {'sz':>2} {'pole':>4} {'x':>8} {'y':>8} "
           f"{'red':>6} {'yel':>5} {'grn':>6} {'cyc':>6} {'d_path':>7} "
           f"{'wp':>3} {'d_L11':>7} {'d_L16':>7}")
    print(hdr)
    for r in on:
        print(f"{r['actor_id']:>4} {r['group_key']:>4} {r['group_size']:>2} "
              f"{r['pole_index']:>4} {r['x']:>8.2f} {r['y']:>8.2f} "
              f"{r['red_time_s']:>6.2f} {r['yellow_time_s']:>5.2f} "
              f"{r['green_time_s']:>6.2f} {r['cycle_s']:>6.2f} "
              f"{r['distance_to_route_b_path_m']:>7.2f} "
              f"{r['nearest_ordered_waypoint_index']:>3} "
              f"{r['distance_to_long_wait_m']['leg11_x109_y70']:>7.2f} "
              f"{r['distance_to_long_wait_m']['leg16_xm73_y28']:>7.2f}")

    import statistics as st
    for name, key in (("red", "red_time_s"), ("yellow", "yellow_time_s"),
                      ("green", "green_time_s"), ("cycle", "cycle_s")):
        vals = [r[key] for r in on]
        allv = [r[key] for r in records]
        print(f"route-b {name:6s}: min={min(vals):.2f} median={st.median(vals):.2f} "
              f"max={max(vals):.2f}   |  town-wide: min={min(allv):.2f} "
              f"median={st.median(allv):.2f} max={max(allv):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
