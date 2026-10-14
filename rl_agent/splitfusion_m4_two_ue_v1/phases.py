"""Flushed M4 startup/teardown phase markers. Importing this module starts nothing.

Each marker is one unbuffered O_APPEND write, so it survives a later native
abort of this process. faulthandler records the Python frames active when a
fatal signal arrives; that locates the abort, it does not explain a native
(C/C++) cause.
"""
import atexit
import contextlib
import faulthandler
import json
import os
from pathlib import Path
import threading
import time
import traceback

_ACTIVE = None


class _Null:
    def step(self, *_, **__): pass
    def event(self, *_, **__): pass
    def fail_open_step(self, *_, **__): pass
    @contextlib.contextmanager
    def phase(self, *_, **__):
        yield


class PhaseLog:
    """PHASES.jsonl = every ENTER/DONE/ERROR; FIRST_FAILURE.json = first
    non-teardown error only; teardown errors never replace it."""

    def __init__(self, output):
        self.output = Path(output)
        self._lock = threading.Lock()
        self._seq = 0
        self._open = None
        self.first_failure = None
        self._fd = os.open(self.output / 'PHASES.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)
        # Kept open for the process lifetime: an abort can occur after run() returns.
        self._fault = (self.output / 'FAULTHANDLER.log').open('x')
        faulthandler.enable(file=self._fault, all_threads=True)
        self.event('faulthandler.enabled', file='FAULTHANDLER.log', all_threads=True)
        # Runs after non-daemon threads join, before interpreter/module teardown.
        atexit.register(lambda: self.event('process.atexit', threads=live_threads()))

    def _write(self, **row):
        with self._lock:
            self._seq += 1
            row = dict(seq=self._seq, wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                       pid=os.getpid(), thread=threading.current_thread().name, **row)
            os.write(self._fd, (json.dumps(row, sort_keys=True, default=repr) + '\n').encode())

    def _failure(self, name, stage, error):
        self._write(event='ERROR', phase=name, stage=stage, error=repr(error),
                    traceback=traceback.format_exception(type(error), error, error.__traceback__))
        if stage == 'teardown':
            return
        with self._lock:
            if self.first_failure is not None:
                return
            self.first_failure = dict(phase=name, stage=stage, error=repr(error), wall_ns=time.time_ns(),
                thread=threading.current_thread().name,
                traceback=traceback.format_exception(type(error), error, error.__traceback__))
            with (self.output / 'FIRST_FAILURE.json').open('x') as stream:
                json.dump(self.first_failure, stream, indent=2, sort_keys=True, default=repr)
                stream.write('\n'); stream.flush()

    def event(self, name, **fields):
        self._write(event='EVENT', phase=name, **fields)

    def step(self, name, *, stage='startup'):
        """Linear main-thread sequence: completes the previous step, enters the next."""
        if self._open is not None:
            self._write(event='DONE', phase=self._open[0], stage=self._open[1])
        self._open = None if name is None else (name, stage)
        if name is not None:
            self._write(event='ENTER', phase=name, stage=stage)

    def fail_open_step(self, error):
        if self._open is not None:
            name, stage = self._open; self._open = None
            self._failure(name, stage, error)
        else:
            self._failure('unattributed', 'startup', error)

    @contextlib.contextmanager
    def phase(self, name, *, stage='startup'):
        self._write(event='ENTER', phase=name, stage=stage)
        try:
            yield
        except BaseException as error:
            self._failure(name, stage, error)
            raise
        self._write(event='DONE', phase=name, stage=stage)


def live_threads():
    return [dict(name=t.name, daemon=t.daemon) for t in threading.enumerate()]


def install(output):
    global _ACTIVE
    _ACTIVE = PhaseLog(output)
    return _ACTIVE


def current():
    return _ACTIVE if _ACTIVE is not None else _Null()
