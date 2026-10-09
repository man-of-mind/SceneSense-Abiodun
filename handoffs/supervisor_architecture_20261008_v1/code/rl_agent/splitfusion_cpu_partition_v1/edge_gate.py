"""Hold edge READY until all radio/edge threads have verified placement."""
import json
import os
import threading
import time
from pathlib import Path


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def snapshot(tid):
    directory = Path('/proc/self/task') / str(tid)
    return dict(pid=os.getpid(), tid=tid, raw_ns=raw(),
        affinity=sorted(os.sched_getaffinity(tid)),
        policy=os.sched_getscheduler(tid), priority=os.sched_getparam(tid).sched_priority,
        starttime=directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19])


def wait(path, timeout=60):
    end = time.monotonic() + timeout
    while not Path(path).exists():
        if time.monotonic() > end:
            raise TimeoutError(str(path))
        time.sleep(.02)


def install(path, prewarm):
    directory = Path(path)
    pending = []
    publish = prewarm.publish_ready_after_warmup
    run, start = threading.Thread.run, threading.Thread.start

    def deferred(warm, ready, *args, **kwargs):
        return publish(warm, lambda: pending.append(ready), *args, **kwargs)

    def target(thread):
        function = getattr(thread, '_target', None)
        return (thread.name == 'run4-edge-process'
                and getattr(function, '__name__', '') == 'process_loop'
                and getattr(function, '__module__', '').endswith('phase6_edge_runtime_v2'))

    def gated_run(thread):
        if not target(thread):
            return run(thread)
        tid = threading.get_native_id()
        before = snapshot(tid)
        write(directory / 'worker_ready.json', before)
        try:
            wait(directory / 'release.json')
            release = json.loads((directory / 'release.json').read_text())
            after = snapshot(tid)
            assert after['affinity'] == release['expected_affinity']
            assert (after['policy'], after['priority']) == (before['policy'], before['priority'])
            write(directory / 'worker_verified.json', after)
            return run(thread)
        finally:
            # The host controller restores EVERY affected TID, not just this worker.
            write(directory / 'worker_exit.json', snapshot(tid))

    def gated_start(thread):
        chosen = target(thread)
        result = start(thread)
        if chosen:
            wait(directory / 'worker_verified.json', 65)
            assert len(pending) == 1
            pending.pop()()
            write(directory / 'ready_released.json', {'raw_ns': raw()})
        return result

    prewarm.publish_ready_after_warmup = deferred
    threading.Thread.run, threading.Thread.start = gated_run, gated_start
