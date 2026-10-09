"""CPU-only actual profiler/numeric evaluator class composition probe."""
from contextlib import nullcontext
from pathlib import Path


class ProbeRecorder:
    def __init__(self):
        self.gauges = []
    def register_gauge(self, name, callback):
        self.gauges.append(name)
    def span(self, *args, **kwargs):
        return nullcontext()
    def mark(self, *args, **kwargs):
        pass
    def note(self, *args, **kwargs):
        pass
    def bind(self, **kwargs):
        pass
    def unbind(self):
        pass
    def context(self):
        return {}


def check(directory):
    """No predictions/GT scores are faked; exercise real empty start/drain."""
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from .edge_live import install_numeric_edge, NumericEvaluatorV1
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    sidecars = directory / "sidecars"
    sidecars.mkdir()
    rec = ProbeRecorder()
    with install_numeric_edge(socket_path=directory / "numeric_gt.sock", scorer_client=None,
            archive_client=None, archive_root_index=0, sidecar_directory=sidecars,
            max_frames=1) as state:
        EP.install_edge_seams(rec)
        if not issubclass(E.Run4EvaluatorV2, NumericEvaluatorV1):
            raise ValueError("profiler replaced numerical evaluator rather than composing it")
        spec = W.load_run4_quality_spec(Path(__file__).resolve().parents[2])
        evaluator = E.Run4EvaluatorV2(spec=spec, read_ground_truth=None,
            send=lambda payload: (_ for _ in ()).throw(AssertionError("unexpected empty feedback")),
            match_distance_m=2., gt_timeout_s=.25)
        evaluator.start()
        report = evaluator.close()
        if report["worker_alive"] or report["archive_worker_alive"] or report["failure"] is not None:
            raise ValueError("actual composed evaluator failed empty startup/drain")
        if sorted(rec.gauges) != ["edge.evaluator_outstanding", "edge.evaluator_queue"]:
            raise ValueError("actual profiler evaluator gauge integration differs")
    return dict(actual_profiled_numeric_class=True, evaluator_close=report,
                numeric_scope=state, empty_probe_only=True, cuda_work=False)
