"""Prospectively bound overlay separating CUDA lifecycle from CPU settings.

The sealed implementation, source validation, lock, prewarm and restoration
are reused unchanged. No setting is assigned here, and no CUDA work is run.
"""
from contextlib import contextmanager
from types import FunctionType

from rl_agent.splitfusion_live_gt_optimization_v1 import runtime as R


def configuration_settings():
    result = R.settings()
    result.pop("cuda_initialized", None)
    return result


@contextmanager
def install_cpu_gt_optimizations(*, expected_sources):
    # Validate the prospective binding before even loading the CPU modules.
    R.verify_sources(expected_sources)
    import torch
    import cv2  # noqa: F401 - stable optional settings keys before the baseline
    before = torch.cuda.is_initialized()
    if type(before) is not bool:
        raise TypeError("CUDA lifecycle observation must be boolean")
    generator = R.install_cpu_gt_optimizations.__wrapped__
    private = FunctionType(generator.__code__,
        dict(generator.__globals__, settings=configuration_settings),
        generator.__name__, generator.__defaults__, generator.__closure__)
    private.__kwdefaults__ = generator.__kwdefaults__
    state = None
    try:
        with contextmanager(private)(expected_sources=expected_sources) as state:
            state["cuda_lifecycle_before"] = before
            yield state
    finally:
        if state is not None:
            after = torch.cuda.is_initialized()
            if type(after) is not bool:
                raise TypeError("CUDA lifecycle observation must be boolean")
            state["cuda_lifecycle_after"] = after
            state["cuda_initialized_by_retained_path"] = not before and after
