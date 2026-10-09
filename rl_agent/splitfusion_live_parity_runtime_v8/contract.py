"""Additive source identity for live scoring-before-archive, not new numerics."""
from pathlib import Path
from rl_agent.splitfusion_live_parity_runtime_v7 import contract as V7

PACKAGE = "rl_agent/splitfusion_live_parity_runtime_v8"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_parity_v8"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_parity_v8"
RUNNER = "rl_agent.splitfusion_live_parity_runtime_v8.runner"
PARTITION = "rl_agent.splitfusion_live_parity_runtime_v8.partition_apply"
ID = "RUN5_GT_SCORING_HANDOFF_100_ENGINEERING_V8"
TOKEN = "RUN5_GT_SCORING_HANDOFF_100_ONCE_EXECUTE_V8"
BUDGET = 100
HANDOFF_BINDING = "ee133ad260fce945d041664a30af9aaef8b1109706a681228bba587db7f34b49"
BOOTSTRAP_PACKAGES = ("rl_agent/splitfusion_live_gt_optimization_v5",
    "rl_agent/splitfusion_live_gt_optimization_v6", "rl_agent/splitfusion_live_gt_handoff_v1")
WORKER_PACKAGES = V7.WORKER_PACKAGES
WORKER_TIMEOUT_S = V7.WORKER_TIMEOUT_S
WORKER_BINDING_SHA256 = V7.WORKER_BINDING_SHA256
WORKER_BINDING = V7.PACKAGE + "/worker_binding.json"
REPLAY_PACKAGES = V7.REPLAY_PACKAGES
REPAIR_PACKAGE = V7.REPAIR_PACKAGE
OPTIMIZATION_PACKAGE = V7.OPTIMIZATION_PACKAGE
OPTIMIZATION_SOURCES = V7.OPTIMIZATION_SOURCES
INPUT_MAX_FRAMES, INPUT_MAX_BYTES = V7.INPUT_MAX_FRAMES, V7.INPUT_MAX_BYTES
INPUT_SPEC = dict(V7.INPUT_SPEC)
IMAGE_ID = V7.IMAGE_ID
sha, canonical, metrics = V7.sha, V7.canonical, V7.metrics
optimization_inventory = V7.optimization_inventory
worker_inventory = V7.worker_inventory


def _regular(path, root):
    if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root))):
        raise ValueError("unsafe/incomplete source: " + str(path))


def bootstrap_inventory(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for package in BOOTSTRAP_PACKAGES:
        folder = root / package
        if not (folder / "__init__.py").is_file() or folder.is_symlink():
            raise ValueError("missing handoff package")
        for path in folder.glob("*.py"):
            if not path.name.startswith("test_"):
                _regular(path, root)
                result[str(path.relative_to(root))] = sha(path)
    result = dict(sorted(result.items()))
    if canonical(dict(schema="splitfusion.gt_scoring_before_archive.v1", sources=result)) != HANDOFF_BINDING:
        raise ValueError("handoff differs from sealed 192-replay implementation")
    return result


def handoff_binding(root):
    bootstrap_inventory(root)
    return HANDOFF_BINDING


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = V7.source_inventory(root)
    bootstrap_inventory(root)
    paths = {p for package in (PACKAGE, *BOOTSTRAP_PACKAGES)
             for p in (root/package).rglob("*")
             if p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
    if not (root/PACKAGE/"__init__.py").is_file():
        raise ValueError("new runtime package missing")
    paths.update(root/(m.replace(".", "/")+".py") for m in (CHILD, EDGE))
    for path in sorted(paths):
        _regular(path, root)
        result[str(path.relative_to(root))] = sha(path)
    return dict(sorted(result.items()))


def lineage(root, actor_boundary):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root),
        optimization_sources=optimization_inventory(root), worker_sources=worker_inventory(root),
        bootstrap_sources=bootstrap_inventory(root), handoff_binding=handoff_binding(root), metrics=metrics(),
        controller="RewardHoldControllerV2", engine="Run4DecisionEngineV2",
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", fallback="mode11_q9800_a71"))
