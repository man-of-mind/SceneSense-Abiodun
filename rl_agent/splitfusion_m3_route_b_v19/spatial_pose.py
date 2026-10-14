"""Causal pose publication at the retained V13/V14 route tick-owner seam.

No extra tick, route change, projection, future waypoint or actor feature.
Publish the observed initial pose before the first route tick; then after each
completed tick. The independent controller reads an atomic current snapshot.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
import time

from . import contract as C

SCHEMA = C.SCHEDULE['pose_schema']
SOURCE = C.SCHEDULE['pose_source']


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('spatial pose symlink refused')
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.spatial_pose_')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


class Publisher:
    def __init__(self, attempt):
        self.directory = Path(attempt)/'m3_route'
        self.directory.mkdir(exist_ok=True)
        self.snapshot = Path(attempt)/C.POSE_SNAPSHOT_RELATIVE
        self.snapshot.parent.mkdir(exist_ok=True)
        self.summary = self.directory/'SPATIAL_POSE_SUMMARY.json'
        self.trace_path = self.directory/'SPATIAL_POSE_TRACE.jsonl'
        if any(p.exists() or p.is_symlink() for p in (self.snapshot, self.summary, self.trace_path)):
            raise FileExistsError('spatial pose evidence is create-only')
        self.trace = self.trace_path.open('x')
        self.last = None
        self.samples = 0
        self.route_sha = C.sha(C.repo_root()/C.ROUTE_JSON)

    def publish(self, frame, x, y):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('nonfinite route ego pose')
        if self.last is not None and int(frame) <= self.last['source_frame_id']:
            raise ValueError('route pose frame failed strict ordering')
        value = dict(schema=SCHEMA, source=SOURCE, status='ACTIVE', clock_domain='CLOCK_MONOTONIC',
                     source_frame_id=int(frame), ego_x=float(x), ego_y=float(y),
                     source_monotonic_ns=time.monotonic_ns(),
                     source_monotonic_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                     source_wall_ns=time.time_ns(), route_json_sha256=self.route_sha)
        self.trace.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        self.trace.flush()
        atomic_json(self.snapshot, value)
        self.last = value
        self.samples += 1
        return value

    def close(self, status, error=None):
        if self.trace.closed:
            return
        # Preserve the source pose/time, not a restamped observation during drain.
        value = dict(self.last or {}, schema=SCHEMA, source=SOURCE, status=status, error=error,
                     closed_monotonic_ns=time.monotonic_ns(), samples=self.samples)
        self.trace.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        self.trace.flush()
        self.trace.close()
        atomic_json(self.snapshot, value)
        atomic_json(self.summary, value)


class PoseWorld:
    def __init__(self, world, vehicle, publisher):
        self.world, self.vehicle, self.publisher = world, vehicle, publisher

    def tick(self, *args, **kwargs):
        frame = int(self.world.tick(*args, **kwargs))
        location = self.vehicle.get_transform().location
        self.publisher.publish(frame, location.x, location.y)
        return frame

    def __getattr__(self, name):
        return getattr(self.world, name)


@contextmanager
def install_spatial_pose(attempt):
    import data_collection.run_route_b_density_loop as density
    original = density.drive_one_loop_with_traffic
    if getattr(original, '__m3_spatial_pose__', False):
        raise RuntimeError('spatial pose seam already installed')

    def drive(world, vehicle, agent, route, collisions, args, loop_index, maintain, janitor):
        publisher = Publisher(attempt)
        try:
            snapshot = world.get_snapshot()
            location = vehicle.get_transform().location
            publisher.publish(int(snapshot.frame), location.x, location.y)
            result = original(PoseWorld(world, vehicle, publisher), vehicle, agent, route,
                              collisions, args, loop_index, maintain, janitor)
            publisher.close('ROUTE_COMPLETED' if result.get('completed') else 'ROUTE_INCOMPLETE')
            return result
        except BaseException as exc:
            publisher.close('ABORTED', f'{type(exc).__name__}: {exc}'[:500])
            raise

    drive.__m3_spatial_pose__ = True
    density.drive_one_loop_with_traffic = drive
    try:
        yield
    finally:
        density.drive_one_loop_with_traffic = original
