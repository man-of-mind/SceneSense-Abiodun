"""Same native handoff session with an independently bound worker bootstrap."""
from contextlib import contextmanager
from rl_agent.splitfusion_live_gt_handoff_v1 import runtime as H
from rl_agent.splitfusion_live_gt_placement_trial_v1 import contract as T


@contextmanager
def install_cpu_gt_worker(*, attempt_dir, max_frames, expected_sources,
                          expected_bootstrap_sources, timeout_s=30., _client_factory=None):
    if expected_bootstrap_sources != T.sources() or _client_factory is None:
        raise ValueError("registered trial bootstrap and explicit observed factory required")
    def bound(**kwargs):
        return _client_factory(expected_bootstrap_sources=expected_bootstrap_sources, **kwargs)
    with H._installation()(attempt_dir=attempt_dir, max_frames=max_frames,
            expected_sources=expected_sources, timeout_s=timeout_s, _client_factory=bound) as state:
        state["bootstrap_sources"] = dict(expected_bootstrap_sources)
        yield state
