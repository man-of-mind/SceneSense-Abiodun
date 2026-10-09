"""Private namespace specialization of the exact frozen GT write body."""
from types import FunctionType, SimpleNamespace

from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from .depth_cache import DepthReuse


def clone(function, namespace, *, closure=None):
    result = FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__ if closure is None else closure)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def source_producer(base, depth):
    class RequestDepthProducer(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # These are per-producer private namespaces, never original
            # visibility/static modules. Keep all geometry/validation methods.
            self.visibility = SimpleNamespace(**{**vars(self.visibility),
                "decode_depth_bgra": depth.decode_bgra})
            old = self.static
            private = dict(old)
            for name, function in old.items():
                if isinstance(function, FunctionType) and function.__globals__ is old:
                    private[name] = clone(function, private)
            private["decode_depth_bgra"] = depth.decode_bgra
            self.static = private
    return RequestDepthProducer


def prepare_function(original, depth):
    # V2 has already cloned the adapter and memory-backed V1 target builder.
    # Add only the decoder in its fresh reconstruction namespace. Preserve all
    # other functions, private ndarray operations and repeated validators.
    adapter = original.__globals__["targets"]
    build = adapter.build_complete_v010
    modules = build.__globals__["_modules"]

    def request_modules(*args, **kwargs):
        builder, visibility, recon = modules(*args, **kwargs)
        return builder, visibility, SimpleNamespace(**{**vars(recon),
            "decode_depth_bgra": depth.decode_bgra})

    private_build = clone(build, dict(build.__globals__, _modules=request_modules))
    private_adapter = SimpleNamespace(**{**vars(adapter), "build_complete_v010": private_build})
    return clone(original, dict(original.__globals__, targets=private_adapter))


class Engine(W.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S, evaluation as E
            from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
            self.depth = DepthReuse(LG.decode_depth, S._original()[3].decode_depth_bgra)
            # The original process timer and error handler invoke self.write.
            # Retain its exact code and closed-over ownership/catalogue/state.
            original = self.write
            cells = dict(zip(original.__code__.co_freevars, original.__closure__))
            old_tuple = cells["original"].cell_contents
            avo = old_tuple[2].__wrapped__
            private_avo = clone(avo, dict(avo.__globals__, decode_depth=self.depth.decode_image))
            cells["original"] = W.cell((old_tuple[0], old_tuple[1],
                self.log.wrap("avo_scoring_and_sidecar", private_avo), old_tuple[3]))
            prepare = self.log.wrap("prepare_complete_target",
                prepare_function(E.prepare_target.__wrapped__, self.depth))
            namespace = dict(original.__globals__,
                S=SimpleNamespace(**{**vars(S),
                    "LiveCompleteTargetProducer": source_producer(S.LiveCompleteTargetProducer, self.depth)}),
                evaluation=SimpleNamespace(**{**vars(E), "prepare_target": prepare}))
            private_write = clone(original, namespace,
                closure=tuple(cells[name] for name in original.__code__.co_freevars))

            def scoped_write(directory, **values):
                with self.depth.request_scope(identity=values["identity"], depth_image=values["depth_image"]):
                    return private_write(directory, **values)
            self.write = scoped_write
        except BaseException:
            self.close()
            raise

    def process(self, **kwargs):
        result = super().process(**kwargs)
        stats = dict(self.depth.last_stats or {}, retained_after_request=self.depth.retained,
                     active_after_request=self.depth.active)
        if result["status"] == "OK" and (stats.get("decodes") != 1
                or stats.get("reuses") != 3 or stats.get("calls") != 4
                or stats.get("fallbacks") != 0 or not stats.get("completed")
                or stats["retained_after_request"] or stats["active_after_request"]):
            self.failed = True
            raise W.WorkerError("successful GT request did not exercise exact bounded depth reuse")
        return dict(result, depth_reuse=stats)
