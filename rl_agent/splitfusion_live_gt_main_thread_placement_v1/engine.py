"""v8 CONTROL lifecycle; no native-pool materialization or helper pinning."""
from rl_agent.splitfusion_live_gt_optimization_v8.engine import ControlEngine
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from rl_agent.splitfusion_live_gt_placement_v1.engine import Engine as AllThreadEngine
from .placement import Placement


class Engine(ControlEngine):
    def __init__(self, *, placement_cpu, placement_original_context, **kwargs):
        self.placement = None
        self.placement_scope_record = None
        super().__init__(**kwargs)
        try:
            W.require(self.archiver is not None and self.archiver.process.poll() is None
                      and self.count == 0, "archive READY required before main-thread placement")
            self.placement = Placement(requested_cpu=placement_cpu,
                original_context=placement_original_context, archive_pid=self.archiver.process.pid,
                max_frames=self.max_frames)
            self.placement.start()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def close(self):
        # Reuse only the sealed v9 close implementation; its super() targets
        # ControlEngine through an explicit private __class__ closure below.
        from types import FunctionType
        function = AllThreadEngine.close
        closure = tuple(W.cell(Engine) if name == "__class__" else value
            for name, value in zip(function.__code__.co_freevars, function.__closure__))
        private = FunctionType(function.__code__, function.__globals__, function.__name__,
                               function.__defaults__, closure)
        return private(self)
