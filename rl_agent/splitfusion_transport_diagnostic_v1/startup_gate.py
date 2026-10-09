"""A setup-only listener identity barrier and pre-teardown collector drain."""
import json
import os
from pathlib import Path
import threading
import time


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def wait(path, seconds):
    end = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > end:
            raise TimeoutError(str(path))
        time.sleep(.02)


def task(tid):
    directory = Path('/proc/self/task') / str(tid)
    return dict(tid=tid, starttime=directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19],
                affinity=sorted(os.sched_getaffinity(tid)), policy=os.sched_getscheduler(tid),
                priority=os.sched_getparam(tid).sched_priority)


def wrap_runtime_class(cls, output):
    original = cls.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        assert self.thread.is_alive() and self.thread.native_id is not None
        assert self.sent == 0 and self._run4_ready.is_set()
        assert not self._feedback_rows
        record = dict(pid=os.getpid(), main=task(threading.get_native_id()),
                      listener=task(self.thread.native_id), raw_ns=raw(),
                      receiver=list(self.receiver.getsockname()), remote=list(self.remote),
                      sent=self.sent, boundary='AFTER_PREWARM_BEFORE_SENSOR_CREATION')
        write(output / 'LISTENER_READY.json', record)
        wait(output / 'LISTENER_RELEASE.json', 60)
        assert self.thread.is_alive()
        write(output / 'LISTENER_RELEASED.json', dict(raw_ns=raw()))

    cls.__init__ = init
    return cls


def teardown_gate(stop, output):
    def wrapped(*args, **kwargs):
        write(output / 'TRANSPORT_STOP_REQUESTED.json', dict(raw_ns=raw()))
        try:
            wait(output / 'COLLECTORS_STOPPED.json', 25)
        except TimeoutError as exc:
            write(output / 'COLLECTOR_DRAIN_TIMEOUT.json', dict(error=str(exc), raw_ns=raw()))
        finally:
            result = stop(*args, **kwargs)
        return result
    return wrapped
