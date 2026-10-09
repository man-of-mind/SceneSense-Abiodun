"""Retained live entries, with registered worker and strict clean-exit adapter."""
from rl_agent.splitfusion_live_parity_runtime_v8 import entries as V8
from rl_agent.splitfusion_live_gt_optimization_v7.lifecycle import dispatch_with_retained_shutdown
from . import contract as C
from .bindings import namespace, specialize

_bound = namespace(V8, C=C, H=C.WORKER,
                   dispatch_with_retained_shutdown=dispatch_with_retained_shutdown)
specialize(V8.child, _bound, [("from rl_agent.splitfusion_live_gt_handoff_v1.runtime import install_cpu_gt_worker",
                            "from .runtime import install_cpu_gt_worker")])
specialize(V8.edge, _bound, [("return run_edge_only(args, max_frames=C.BUDGET)",
    "return dispatch_with_retained_shutdown(run_edge_only, args, max_frames=C.BUDGET, handoff_state=state)")])
globals().update({name: value for name, value in _bound.items() if not name.startswith("__")})
