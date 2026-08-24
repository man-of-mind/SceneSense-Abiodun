#!/usr/bin/env python3
"""Read-only: which signal approach each observed long ego block sat on."""
from __future__ import annotations
import math, sys
import carla

PTS = {"leg11_block (109.39, 69.66)": (109.39, 69.66),
       "leg16_block (-72.86, 27.98)": (-72.86, 27.98)}

client = carla.Client("127.0.0.1", 2000); client.set_timeout(60.0)
world = client.get_world(); amap = world.get_map()
lights = sorted(world.get_actors().filter("traffic.traffic_light*"), key=lambda a: int(a.id))
group = {}
for tl in lights:
    try:
        g = sorted([int(tl.id)] + [int(o.id) for o in tl.get_group_traffic_lights()])
    except RuntimeError:
        g = [int(tl.id)]
    group[int(tl.id)] = (g[0], len(g))

for label, (px, py) in PTS.items():
    wp = amap.get_waypoint(carla.Location(x=px, y=py, z=0.0), project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    print(f"\n=== {label} ===")
    print(f"  projected onto road_id={wp.road_id} lane_id={wp.lane_id} s={wp.s:.1f} "
          f"junction={wp.is_junction} lane_width={wp.lane_width:.2f} "
          f"proj_err={math.hypot(px-wp.transform.location.x, py-wp.transform.location.y):.2f} m")
    # Walk forward along the lane and report the first signal encountered.
    cur, walked, hit = wp, 0.0, None
    while walked < 120.0:
        nxt = cur.next(2.0)
        if not nxt:
            break
        cur, walked = nxt[0], walked + 2.0
        for tl in lights:
            try:
                stops = tl.get_stop_waypoints()
            except RuntimeError:
                continue
            for s in stops:
                if (s.road_id == cur.road_id and s.lane_id == cur.lane_id
                        and abs(s.s - cur.s) < 3.0):
                    hit = (int(tl.id), walked, s.road_id, s.lane_id)
                    break
                if hit: break
            if hit: break
        if hit: break
    if hit:
        aid, dist, rid, lid = hit
        gk, gs = group[aid]
        print(f"  --> first signal ahead in-lane: light {aid} (group {gk}, {gs} poles) "
              f"at {dist:.0f} m ahead on road {rid} lane {lid}")
    else:
        print("  --> no traffic-light stop line found within 120 m ahead in-lane")

    near = []
    for tl in lights:
        loc = tl.get_transform().location
        try:
            stops = [(w.transform.location.x, w.transform.location.y,
                      w.road_id, w.lane_id) for w in tl.get_stop_waypoints()]
        except RuntimeError:
            stops = []
        for sx, sy, rid, lid in stops:
            d = math.hypot(px - sx, py - sy)
            if d <= 60.0:
                near.append((d, int(tl.id), rid, lid, round(sx, 1), round(sy, 1)))
    near.sort()
    print(f"  stop lines within 60 m ({len(near)}):")
    for d, aid, rid, lid, sx, sy in near[:8]:
        gk, gs = group[aid]
        print(f"    {d:6.2f} m  light {aid:>3} (group {gk}, {gs} poles) "
              f"road {rid} lane {lid} at ({sx}, {sy})")
