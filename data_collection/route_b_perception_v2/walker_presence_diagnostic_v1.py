#!/usr/bin/env python3
"""Does the walker population really shrink, or does world.get_actors() undercount?

Spawns the same managed 30/30 population the Route B runner uses and, every second,
compares three independent counts of the SAME owned walker bodies:
  A) world.get_actors().filter('walker.pedestrian.*')   <- what the collection gate reads
  B) world.get_actors(owned_ids) filtered to alive      <- what reconcile() reads
  C) per-id world.get_actor(id).is_alive                <- direct per-actor query
"""
import sys, types, random, json
from pathlib import Path

REPO_ROOT = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
ADVISOR = REPO_ROOT / "rl_agent" / "advisor_helper_scripts" / "codes"
for p in (str(REPO_ROOT), str(ADVISOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import carla
import generate_traffic_v1 as traffic

SEED = 101
client = carla.Client("127.0.0.1", 2000); client.set_timeout(120.0)
world = client.load_world("Town10HD_Opt", True)
orig = world.get_settings()
s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05
s.no_rendering_mode = False
world.apply_settings(s)
random.seed(SEED)
tm = client.get_trafficmanager(8010)
tm.set_synchronous_mode(True); tm.set_random_device_seed(SEED)
tm.set_global_distance_to_leading_vehicle(3.0)
world.set_pedestrians_seed(SEED)
world.set_pedestrians_cross_factor(traffic.PERCENTAGE_PEDESTRIANS_CROSSING)

vbps = traffic.get_actor_blueprints(world, "vehicle.*", "All")
vbps = [b for b in vbps if b.has_attribute("base_type") and b.get_attribute("base_type").as_str() == "car"]
vbps = sorted(vbps, key=lambda b: b.id)
wbps = traffic.get_actor_blueprints(world, "walker.pedestrian.*", "All")
spawn_points = world.get_map().get_spawn_points(); random.shuffle(spawn_points)

args = types.SimpleNamespace(number_of_vehicles=30, number_of_walkers=30, car_lights_on=False,
                             hero=False, asynch=False, replenish_interval=5.0,
                             population_log_interval=1e9)
pop = traffic.TrafficPopulationManager(client, world, tm, args, vbps, wbps, spawn_points, True)
pop.spawn_initial_population()
owned = [int(r["id"]) for r in pop.walkers if r.get("id") is not None]
print(f"spawned walkers={len(pop.walkers)} vehicles={len(pop.vehicle_ids)}", flush=True)

rows = []
try:
    for tick in range(2400):          # 120 s simulated
        world.tick()
        if tick % 20 != 0:
            continue
        a = sum(1 for _ in world.get_actors().filter("walker.pedestrian.*"))
        av = sum(1 for x in world.get_actors().filter("vehicle.*"))
        blist = world.get_actors(owned)
        b = sum(1 for x in blist if x is not None and x.is_alive
                and x.type_id.startswith("walker.pedestrian."))
        c = 0
        missing_from_A = []
        present_A = {int(x.id) for x in world.get_actors().filter("walker.pedestrian.*")}
        for i in owned:
            try:
                act = world.get_actor(i)
            except RuntimeError:
                act = None
            if act is not None and act.is_alive:
                c += 1
                if i not in present_A:
                    missing_from_A.append(i)
        rows.append({"sim_s": round(tick * 0.05, 2), "A_full_list": a, "B_by_owned_ids": b,
                     "C_per_actor": c, "vehicles_full_list": av,
                     "alive_but_absent_from_full_list": len(missing_from_A)})
        print(rows[-1], flush=True)
finally:
    try:
        pop.destroy()
    except Exception as exc:
        print("cleanup:", exc)
    world.tick()
    tm.set_synchronous_mode(False)
    world.apply_settings(orig)
Path("/tmp/claude-200171/-home-shr-aisvcs-workarea-carla-0-10-env-Carla-0-10-0-Linux-Shipping-PythonAPI-neu-collab-abiodun/467db1bd-7f26-4822-92bf-4d996b0502e4/scratchpad/walker_presence_diag.json").write_text(json.dumps(rows, indent=2))
print("DIAG_DONE")
