"""Bound the startup repair to the unchanged, synchronously owned V4 runtime.

This is an integration primitive, not a live entry point. A future live
registration must bind both the original body and this bootstrap separately.
"""
from contextlib import contextmanager

from rl_agent.splitfusion_live_gt_optimization_v4 import runtime as R
from . import contract as C
from .client import Client


@contextmanager
def install_cpu_gt_worker(*, attempt_dir, max_frames, expected_sources,
                          expected_bootstrap_sources, timeout_s=30.):
    if expected_bootstrap_sources != C.sources():
        raise ValueError("CPU worker bootstrap binding differs before runtime installation")

    def factory(**kwargs):
        return Client(expected_bootstrap_sources=expected_bootstrap_sources, **kwargs)

    # V4 starts the factory inside _spawn_sensors, before calling the real
    # sensor startup. Client only returns after verified READY. Its constructor
    # owns abort/join on failure because R does not yet hold that client.
    with R.install_cpu_gt_worker(attempt_dir=attempt_dir, max_frames=max_frames,
            expected_sources=expected_sources, timeout_s=timeout_s,
            _client_factory=factory) as state:
        state["bootstrap_sources"] = dict(expected_bootstrap_sources)
        state["effective_caps_required_before_sensor_start"] = True
        yield state
