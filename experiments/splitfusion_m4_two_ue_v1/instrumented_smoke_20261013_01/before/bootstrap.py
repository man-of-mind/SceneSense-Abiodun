"""Production CARLA/traffic bootstrap. Importing this module starts nothing."""
import copy
import dataclasses
import json
import sys
import threading
import types
from pathlib import Path

from .binding import bind
from .identity import require
from .composition import BothStreamLifecycle
from .world import TickOwner, drive_pair, following_transform


def finish_world_resources(steps, output, *, original_error=None):
    """Attempt every owned teardown and never replace an active failure."""
    from .configuration import write_new
    rows=[]
    for name, action in steps:
        try:
            action()
            rows.append(dict(name=name,status='CLOSED'))
        except BaseException as error:
            rows.append(dict(name=name,status='FAILED',error=repr(error)))
    try:
        write_new(Path(output)/'WORLD_CLEANUP.json',dict(
            original_error=repr(original_error) if original_error is not None else None,steps=rows))
    except BaseException:
        if original_error is None: raise
        # Host-level handling still receives the original exception.
    if original_error is None:
        require(all(row['status']=='CLOSED' for row in rows),'world teardown failed: '+repr(rows))
    return rows


def run_world(*, config, sources, output, catalogues, make_collector, start_edge, stop_edge, health=lambda:None):
    """Reuse density.run for one world/population and each retained route driver.

    All lifecycle resources created here are explicitly tracked. The second
    ego is spawned before NPCs. A failed lane/spawn check aborts, never searches
    another endpoint. No actor is teleported or made non-colliding.
    """
    from data_collection import run_route_b_density_loop as D
    from data_collection.run_route_b_perception_collection_v2 import ClientProxy
    from agents.navigation.basic_agent import BasicAgent
    from .configuration import write_new
    import carla

    output = Path(output)
    args = D.build_parser().parse_args([
        '--density', 'traffic_50_50', '--vehicles', '50', '--pedestrians', '50', '--loops', '1',
        '--seed', '31', '--host', config['carla_host'], '--port', str(config['carla_port']), '--tm-port', '8010',
        '--route-config', str(config['repository'] + '/' + config['route_json']),
        '--lane-offset-m', '-0.5', '--target-speed-kph', '25.0', '--walker-brake-distance-m', '10.0',
        '--fixed-delta-seconds', '0.05', '--maximum-loop-sim-s', '900', '--replenish-interval-s', '2.0',
        '--real-time-tick-period-s', '0.05', '--no-spectator', '--no-hybrid-physics',
        '--allow-scenario-interventions', '--maximum-overtakes', '0',
        '--out-csv', str(output / 'route_metrics.csv'), '--summary-json', str(output / 'route_metrics_summary.json')])
    if config.get('smoke'):
        args.maximum_loop_sim_s = 20.  # explicit bounded smoke; never a full-route verdict
    roster, collectors, extras, outcomes = sources.sources, {}, [], {}
    owned_egos = set()
    lock = threading.RLock()
    cleanup = {}
    retired_actors = set()

    class Client(ClientProxy):
        def __init__(self, *values, **kw):
            super().__init__(carla.Client, 31, *values, **kw)
        def load_world(self, *values, **kw):
            world = self._client.load_world(*values, **kw)
            for catalogue in catalogues.values():
                catalogue.after_load(world)
            return world

    def spawn(world, route, run_args):
        leader = D.spawn_ego(world, route, run_args)
        owned_egos.add(int(leader[0].id))
        try:
            transform, evidence = following_transform(world.get_map(), D.transform_of(route['start']))
            follower_route = copy.deepcopy(route)
            follower_route['start'] = dict(route['start'])
            # The following pose is only the spawn. Both drivers retain the
            # original full route and its original completion criteria.
            follower_route['start'] = dict(location=dict(x=transform.location.x, y=transform.location.y,
                z=transform.location.z), rotation=dict(yaw=transform.rotation.yaw,
                pitch=transform.rotation.pitch, roll=transform.rotation.roll))
            follower_args = copy.copy(run_args)
            follower_args.ego_role_name = roster[1].ego_id
            second = D.spawn_ego(world, follower_route, follower_args)
            extras.append(second)
            owned_egos.add(int(second[0].id))
            require(leader[0].bounding_box.extent.x + second[0].bounding_box.extent.x < evidence['distance_m'],
                    'ego bounding boxes overlap initial following gap')
            write_new(output / 'SPAWN.json', dict(evidence, actor_ids=sorted(owned_egos),
                                                follower_transform=follower_route['start']))
            return leader
        except BaseException:
            leader[1].stop(); leader[1].destroy(); leader[0].destroy()
            raise

    def clear_blocker(actor, *values, **kw):
        require(int(actor.id) not in owned_egos, 'retained NPC intervention targeted an ego; abort without moving it')
        with lock:
            return D.clear_blocker(actor, *values, **kw)
    drive_one = bind(D.drive_one_loop_with_traffic, clear_blocker=clear_blocker)

    def drive(world, leader, leader_agent, route, collisions, run_args, loop_index, maintain, janitor):
        follower, _, follower_collisions = extras[0]
        second_agent = BasicAgent(follower, target_speed=25., opt_dict=dict(
            sampling_resolution=float(route['route_sampling_resolution_m']), base_tlight_threshold=5.,
            base_vehicle_threshold=float(run_args.vehicle_threshold_m),
            use_bbs_detection=bool(run_args.use_bbs_detection), max_brake=1., offset=-.5))
        second_agent.follow_speed_limits(False)
        second_agent.set_target_speed(25.)
        for source, ego in zip(roster, (leader, follower)):
            collectors[source.stream_id] = make_collector(source, world, ego)
        start_edge()  # GT sockets exist; edge READY + all-mode prewarm before first offered frame
        lifecycle = BothStreamLifecycle(sources)
        owner = TickOwner(world, collectors, maintain=maintain,health=health)

        class JanitorProxy:
            def __getattr__(self, name):
                return getattr(janitor, name)
            def __setattr__(self, name, value):
                setattr(janitor, name, value)
            def sweep(self, now):
                with lock:
                    if now <= getattr(self, '_m4_last_sweep', -1):
                        return []
                    self._m4_last_sweep = now
                    return janitor.sweep(now)

        def driver(ego, agent, mailbox):
            def run(proxy):
                return drive_one(proxy, ego, agent, route, mailbox, run_args, loop_index,
                                 lambda _: None, JanitorProxy())
            return run
        drivers = {s.stream_id: driver(v, a, c) for s, v, a, c in zip(roster,
            (leader, follower), (leader_agent, second_agent), (collisions, follower_collisions))}
        def drain(stream):
            result = collectors[stream].finish()
            require(bool(result), 'collector cleanup failed: ' + stream)
            cleanup[stream] = dict(result)
        def retired(stream):
            # Cleanup of a completed, drained ego prevents it permanently
            # blocking the common finish point. Never advance/teleport an ego
            # to obtain completion, and never destroy the still-running ego.
            index=next(i for i,s in enumerate(roster) if s.stream_id==stream)
            vehicle=(leader,follower)[index]
            sensor=(None,extras[0][1])[index]
            if sensor is not None:
                sensor.stop(); sensor.destroy()
            vehicle.destroy()
            retired_actors.add(int(vehicle.id))
            cleanup[stream]['retired_after_own_completion_and_drain']=int(vehicle.id)
        try:
            drive_pair(owner, drivers, lifecycle,
                       {name: lambda n=name: drain(n) for name in collectors}, results=outcomes,
                       completion_required=not config.get('smoke',False),retired=retired)
        finally:
            write_new(output / 'TWO_ROUTE_RESULTS.json', dict(results=outcomes, completed=lifecycle.completed,
                       drained=lifecycle.drained, owner_ticks=owner.tick_count, owner_error=owner.error,
                       cleanup=cleanup))
        # Density.run retains the authoritative population ledger and original
        # collision qualification. Its row is conservative across BOTH egos.
        combined = dict(outcomes[roster[0].stream_id])
        combined['completed'] = lifecycle.done
        for name in ('collision_events', 'collision_incidents'):
            combined[name] = [r for v in outcomes.values() for r in v[name]]
        combined['collision_count'] = len(combined['collision_events'])
        combined['collision_incident_count'] = len(combined['collision_incidents'])
        combined['watchdog_aborted']=any(v['watchdog_aborted'] for v in outcomes.values())
        for key in ('return_position_error_m','return_heading_error_deg','ego_overtakes','intervention_count'):
            combined[key]=max(v[key] for v in outcomes.values())
        combined['walker_brake_ticks']=sum(v['walker_brake_ticks'] for v in outcomes.values())
        return combined

    run = bind(D.run, spawn_ego=spawn, drive_one_loop_with_traffic=drive,
               carla=types.SimpleNamespace(**dict(vars(carla), Client=Client)))
    try:
        return run(args)
    finally:
        original_error=sys.exc_info()[1]
        def finish_collector(name, collector):
            result=collector.finish()
            cleanup[name]=dict(result)
            require(bool(result),'collector cleanup failed: '+name)
        steps=[('collector_'+name,lambda n=name,c=collector:finish_collector(n,c))
               for name,collector in collectors.items() if name not in cleanup]
        steps.append(('edge',stop_edge))
        for vehicle,sensor,_ in reversed(extras):
            if int(vehicle.id) not in retired_actors:
                steps.extend([('sensor_stop_'+str(sensor.id),sensor.stop),
                              ('sensor_destroy_'+str(sensor.id),sensor.destroy),
                              ('vehicle_destroy_'+str(vehicle.id),vehicle.destroy)])
        finish_world_resources(steps,output,original_error=original_error)
