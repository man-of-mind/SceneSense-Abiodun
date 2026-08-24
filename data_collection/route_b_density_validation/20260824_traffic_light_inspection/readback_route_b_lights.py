#!/usr/bin/env python3
"""Read-only observed traffic-light dwell-time readback + stop-line geometry.

Ticks an empty Town10HD_Opt synchronously and records how long each light
actually holds each state, so the *effective* red an approach sees (group
cycling) is measured rather than assumed. Nothing is modified.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import carla

OUT = Path(sys.argv[1])
SIM_SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
DT = 0.05
LONG_WAIT_XY = {"leg11_x109_y70": (109.39, 69.66), "leg16_xm73_y28": (-72.86, 27.98)}


def main() -> int:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    original = world.get_settings()
    lights = sorted(world.get_actors().filter("traffic.traffic_light*"),
                    key=lambda a: int(a.id))

    # Stop-line geometry: closest stop waypoint of each light to each long-wait xy.
    geom = {}
    for tl in lights:
        try:
            stops = [(w.transform.location.x, w.transform.location.y)
                     for w in tl.get_stop_waypoints()]
        except RuntimeError:
            stops = []
        entry = {"stop_waypoint_count": len(stops)}
        for key, (px, py) in LONG_WAIT_XY.items():
            entry[key] = (round(min((math.hypot(px - x, py - y) for x, y in stops),
                                    default=float("inf")), 2) if stops else None)
        geom[int(tl.id)] = entry

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)
    try:
        dwell = defaultdict(lambda: defaultdict(list))   # id -> state -> [durations]
        cur = {}
        since = {}
        t = 0.0
        for _ in range(int(SIM_SECONDS / DT)):
            world.tick()
            t += DT
            for tl in lights:
                aid = int(tl.id)
                st = str(tl.get_state())
                if aid not in cur:
                    cur[aid], since[aid] = st, t
                elif st != cur[aid]:
                    dwell[aid][cur[aid]].append(round(t - since[aid], 2))
                    cur[aid], since[aid] = st, t
    finally:
        world.apply_settings(original)

    rows = []
    for tl in lights:
        aid = int(tl.id)
        try:
            group = sorted([aid] + [int(o.id) for o in tl.get_group_traffic_lights()])
        except RuntimeError:
            group = [aid]
        d = dwell[aid]
        # Drop the first observation of each state: it is a truncated partial dwell.
        obs = {}
        for state in ("Red", "Yellow", "Green"):
            vals = d.get(state, [])[1:]
            obs[state] = {
                "n": len(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "mean": round(sum(vals) / len(vals), 2) if vals else None,
            }
        rows.append({
            "actor_id": aid, "group_key": group[0], "group_size": len(group),
            "configured_red_s": round(float(tl.get_red_time()), 2),
            "configured_yellow_s": round(float(tl.get_yellow_time()), 2),
            "configured_green_s": round(float(tl.get_green_time()), 2),
            "observed": obs,
            "stop_line_geometry_m": geom[aid],
        })

    OUT.write_text(json.dumps(
        {"sim_seconds_observed": SIM_SECONDS, "fixed_delta_seconds": DT,
         "long_wait_locations": LONG_WAIT_XY, "lights": rows}, indent=2) + "\n")

    print(f"observed {SIM_SECONDS:.0f} sim-s on an empty world; wrote {OUT}\n")
    print(f"{'id':>4} {'grp':>4} {'sz':>2} | {'cfg R/Y/G':>12} | "
          f"{'obs Red mean':>12} {'max':>6} | {'obs Grn mean':>12} | "
          f"{'stop->L11':>9} {'stop->L16':>9}")
    for r in rows:
        o = r["observed"]
        g = r["stop_line_geometry_m"]
        cfg = f"{r['configured_red_s']:.0f}/{r['configured_yellow_s']:.0f}/{r['configured_green_s']:.0f}"
        print(f"{r['actor_id']:>4} {r['group_key']:>4} {r['group_size']:>2} | {cfg:>12} | "
              f"{str(o['Red']['mean']):>12} {str(o['Red']['max']):>6} | "
              f"{str(o['Green']['mean']):>12} | "
              f"{str(g['leg11_x109_y70']):>9} {str(g['leg16_xm73_y28']):>9}")

    reds = [r["observed"]["Red"]["mean"] for r in rows if r["observed"]["Red"]["mean"]]
    grns = [r["observed"]["Green"]["mean"] for r in rows if r["observed"]["Green"]["mean"]]
    import statistics as st
    if reds:
        print(f"\nobserved red  mean: min={min(reds):.2f} median={st.median(reds):.2f} max={max(reds):.2f}")
    if grns:
        print(f"observed green mean: min={min(grns):.2f} median={st.median(grns):.2f} max={max(grns):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
