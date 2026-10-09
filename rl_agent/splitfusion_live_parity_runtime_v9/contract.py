"""Independent worker-source identity with the unchanged scoring wire."""
from pathlib import Path
from types import SimpleNamespace
from rl_agent.splitfusion_live_parity_runtime_v8 import contract as V8
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H
from rl_agent.splitfusion_live_gt_placement_trial_v1 import contract as T

PACKAGE = "rl_agent/splitfusion_live_parity_runtime_v9"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_parity_v9"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_parity_v9"
RUNNER = "rl_agent.splitfusion_live_parity_runtime_v9.runner"
PARTITION = "rl_agent.splitfusion_live_parity_runtime_v9.partition_apply"
ID = "RUN5_MAIN_GT_TID_MATCHED_100_V9"
TOKEN = "RUN5_MAIN_GT_TID_MATCHED_100_EXECUTE_V9"
BUDGET = 100
WORKER_TIMEOUT_S = V8.WORKER_TIMEOUT_S
WORKER_BINDING_SHA256 = V8.WORKER_BINDING_SHA256
WORKER_BINDING = V8.WORKER_BINDING
INPUT_MAX_FRAMES, INPUT_MAX_BYTES = V8.INPUT_MAX_FRAMES, V8.INPUT_MAX_BYTES
INPUT_SPEC = dict(V8.INPUT_SPEC)
IMAGE_ID = V8.IMAGE_ID
sha, canonical, metrics = V8.sha, V8.canonical, V8.metrics
optimization_inventory, worker_inventory = V8.optimization_inventory, V8.worker_inventory
_regular = V8._regular
HANDOFF_BINDING = V8.HANDOFF_BINDING
BOOTSTRAP_PACKAGES = (*V8.BOOTSTRAP_PACKAGES,
    "rl_agent/splitfusion_live_gt_optimization_v8", "rl_agent/splitfusion_live_gt_placement_v1",
    "rl_agent/splitfusion_live_gt_main_thread_placement_v1", "rl_agent/splitfusion_live_gt_placement_trial_v1")
WORKER = SimpleNamespace(**dict(vars(H), sources=T.sources))


def bootstrap_inventory(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for name in T.sources():
        path = root / name
        _regular(path, root)
        result[name] = sha(path)
    return dict(sorted(result.items()))


def handoff_binding(root):
    return V8.handoff_binding(root)


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = V8.source_inventory(root)
    paths = list((root / PACKAGE).rglob("*.py"))
    for package in (*BOOTSTRAP_PACKAGES, "rl_agent/splitfusion_live_gt_optimization_v7"):
        paths += list((root / package).rglob("*.py"))
    paths += [root / (module.replace(".", "/") + ".py") for module in (CHILD, EDGE)]
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        _regular(path, root)
        result[str(path.relative_to(root))] = sha(path)
    return dict(sorted(result.items()))


def lineage(root, actor_boundary):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root),
        optimization_sources=optimization_inventory(root), worker_sources=worker_inventory(root),
        bootstrap_sources=bootstrap_inventory(root), handoff_binding=handoff_binding(root), metrics=metrics(),
        controller="RewardHoldControllerV2", engine="Run4DecisionEngineV2",
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", fallback="mode11_q9800_a71"))


def trial(mode):
    if mode not in ("control", "p4"):
        raise ValueError("registered trial mode must be control or p4")
    return dict(mode=mode, placement_cpu=4 if mode == "p4" else None,
                affinity_scope="main_gt_tid_only" if mode == "p4" else "none_control",
                gc_mode="control", native_pool_preparation=False)
