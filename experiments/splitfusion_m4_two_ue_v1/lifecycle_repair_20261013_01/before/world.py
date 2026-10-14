"""One world tick owner; two retained route-driver state machines.

Drivers rendezvous only at their existing world.tick boundary. Sensor offers
remain independent: neither waits for the other source's inference or GT.
An ended driver leaves the rendezvous, not the other driver's pending work.
"""
import concurrent.futures
import math
import threading
import time

from .identity import require


class TickOwner:
    def __init__(self, world, collectors, *, maintain=lambda _: None, timeout_s=30., health=lambda:None):
        self.world, self.collectors, self.maintain = world, collectors, maintain
        self.condition = threading.Condition()
        self.active = set(collectors)
        self.waiting = set()
        self.generation = self.tick_count = 0
        self.frame = None
        self.error = None
        self.timeout_s = timeout_s
        self.health = health

    def proxy(self, stream):
        owner = self
        class WorldProxy:
            def __getattr__(self, name):
                return getattr(owner.world, name)
            def tick(self, *args, **kwargs):
                require(not args and not kwargs, 'retained route tick signature drift')
                with owner.condition:
                    require(stream in owner.active and stream not in owner.waiting, 'duplicate/retired ticker')
                    generation = owner.generation
                    owner.waiting.add(stream)
                    owner.condition.notify_all()
                    ready = owner.condition.wait_for(
                        lambda: owner.generation != generation or owner.error is not None, timeout=owner.timeout_s)
                    require(ready and owner.error is None, 'world owner failed: ' + str(owner.error))
                    return owner.frame
        return WorldProxy()

    def retire(self, stream):
        with self.condition:
            require(stream not in self.waiting, 'retired during tick')
            self.active.remove(stream)
            self.condition.notify_all()

    def fail(self, error):
        with self.condition:
            self.error = repr(error)
            self.condition.notify_all()

    def run(self):
        try:
            while True:
                with self.condition:
                    ready = self.condition.wait_for(
                        lambda: not self.active or self.error is not None or self.waiting == self.active,
                        timeout=self.timeout_s)
                    require(ready and self.error is None, 'route driver failed/stalled: ' + str(self.error))
                    if not self.active:
                        return
                    active = sorted(self.active)
                self.health()
                frame = int(self.world.tick())  # the ONLY real world.tick call
                snapshot = self.world.get_snapshot()
                require(int(snapshot.frame) == frame, 'foreign/unobserved world tick')
                self.tick_count += 1
                # Match SamplingWorld: barriers before any on_world_tick call.
                for stream in active:
                    self.collectors[stream].wait_for_sensor_frame(frame, snapshot=snapshot, timeout_s=2.)
                for stream in active:
                    self.collectors[stream].on_world_tick(frame, self.tick_count)
                self.maintain(self.tick_count * .05)
                with self.condition:
                    self.frame = frame
                    self.waiting.clear()
                    self.generation += 1
                    self.condition.notify_all()
        except BaseException as error:
            self.fail(error)
            raise


def drive_pair(owner, drivers, lifecycle, drains, *, results=None, completion_required=True,
               retired=lambda stream:None):
    """Every driver returns its own retained full-route result, including errors."""
    require(set(drivers) == set(owner.collectors) == set(drains), 'two-source coverage mismatch')
    results = {} if results is None else results
    failures = {}
    def drive(stream, callback):
        try:
            result = callback(owner.proxy(stream))
            results[stream] = result
            if result['completed']:
                lifecycle.complete(lifecycle.sources.resolve(stream))
            elif completion_required:
                raise ValueError('incomplete route: ' + str(result.get('abort_reason')))
            # No more offers from this source. Let the other driver keep
            # ticking while this source's retained final-frame drain runs.
            owner.retire(stream)
            drains[stream]()
            lifecycle.drained[stream]=True
            if result['completed']:
                retired(stream)
        except BaseException as error:
            failures[stream] = repr(error)
            owner.fail(error)
        finally:
            with owner.condition:
                owner.waiting.discard(stream)
                if stream in owner.active:
                    owner.active.remove(stream)
                owner.condition.notify_all()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(drive, name, fn) for name, fn in drivers.items()]
        try:
            owner.run()
        finally:
            for future in futures:
                future.result()
            lifecycle.drain({name:(lambda:None) if lifecycle.drained[name] else callback
                             for name,callback in drains.items()})
    require(not failures and all(lifecycle.drained.values()) and
            (lifecycle.done or not completion_required), 'both route completion/drain required: ' + repr(failures))
    return results


def following_transform(world_map, leader_transform, *, separation_m=15.):
    """Use the actual lane graph, refuse junctions/branches/wrong-way spawns."""
    waypoint = world_map.get_waypoint(leader_transform.location, project_to_road=False)
    require(waypoint is not None and not waypoint.is_junction, 'leader not on an unambiguous driving lane')
    candidates = waypoint.previous(separation_m)
    require(len(candidates) == 1, 'following spawn branches at lane boundary')
    follower = candidates[0]
    require(not follower.is_junction and str(follower.lane_type).endswith('Driving'), 'unsafe following lane')
    require((waypoint.road_id, waypoint.section_id, waypoint.lane_id) ==
            (follower.road_id, follower.section_id, follower.lane_id), 'following spawn leaves leader lane')
    a, b = leader_transform.location, follower.transform.location
    distance = math.hypot(float(a.x - b.x), float(a.y - b.y))
    heading = abs((float(leader_transform.rotation.yaw - follower.transform.rotation.yaw) + 180) % 360 - 180)
    require(13. <= distance <= 17. and heading <= 15., 'following spawn separation/heading not validated')
    return follower.transform, dict(distance_m=distance, heading_delta_deg=heading,
        road_id=waypoint.road_id, section_id=waypoint.section_id, lane_id=waypoint.lane_id)
