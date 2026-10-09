"""Reuse qualified v10 lifecycle around unchanged v8 CONTROL computation."""
from types import FunctionType
from rl_agent.splitfusion_live_gt_main_thread_placement_v1.engine import Engine as MainEngine
from .placement import Placement
from . import contract as C


class Engine(MainEngine):
    def __init__(self, *, placement_mode, placement_original_context, **kwargs):
        if type(placement_mode) is not str or placement_mode not in C.MODES:
            raise ValueError("registered trial mode required")
        def placement_factory(*, requested_cpu, **options):
            return Placement(mode=placement_mode, **options)
        function = MainEngine.__init__
        private = FunctionType(function.__code__, dict(function.__globals__, Placement=placement_factory),
                               function.__name__, function.__defaults__, function.__closure__)
        private(self, placement_cpu=C.TREATMENT_CPU if placement_mode == "p4" else None,
                placement_original_context=placement_original_context, **kwargs)
