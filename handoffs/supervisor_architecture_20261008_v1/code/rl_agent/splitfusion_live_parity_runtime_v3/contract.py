"""New source identity; the sealed v1 scientific contract remains unchanged."""
from pathlib import Path
from rl_agent.splitfusion_live_parity_runtime_v2 import contract as V1

PACKAGE = "rl_agent/splitfusion_live_parity_runtime_v3"
REPAIR_PACKAGE = "rl_agent/splitfusion_live_parity_repair_v2"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_parity_v3"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_parity_v3"
RUNNER = "rl_agent.splitfusion_live_parity_runtime_v3.runner"
PARTITION = "rl_agent.splitfusion_live_parity_runtime_v3.partition_apply"
ID = "RUN5_COMPLETE_GT_PARITY_100_ENGINEERING_V3"
BUDGET = 100
TOKEN = "RUN5_COMPLETE_GT_PARITY_100_ONCE_EXECUTE_V3"
IMAGE_ID = "sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba"
sha = V1.sha
canonical = V1.canonical
metrics = V1.metrics


def source_inventory(root):
    """Include actual retained v1, all v2 code, and both new command shims."""
    root = Path(root).resolve(strict=True)
    result = V1.source_inventory(root)
    directory = root / PACKAGE
    if not directory.is_dir() or directory.is_symlink() or not (directory / "__init__.py").is_file():
        raise ValueError("required v2 package missing")
    repair_directory = root / REPAIR_PACKAGE
    if not (repair_directory / "__init__.py").is_file():
        raise ValueError("required v2 repair package missing")
    paths = {p for folder in (directory, repair_directory) for p in folder.rglob("*") if p.suffix in (".py", ".json", ".pt")
             and "__pycache__" not in p.parts}
    paths.update(root / name for name in (
        "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v3.py",
        "rl_agent/splitfusion_edge_optimization_v1/detached_tail_v3.py",
        "rl_agent/splitfusion_live_dispatch_v1/frame_context.py",
        "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_edge_runtime_v2.py"))
    paths.update(root / (module.replace(".", "/") + ".py") for module in (CHILD, EDGE))
    if any(not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root)
           or any(parent.is_symlink() for parent in p.parents if parent != root and parent.is_relative_to(root))
           for p in paths):
        raise ValueError("incomplete/nonregular/escaping v2 runtime inventory")
    result.update({str(p.relative_to(root)): sha(p) for p in sorted(paths)})
    return dict(sorted(result.items()))


def lineage(root, actor_boundary):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root),
        metrics=metrics(), controller="RewardHoldControllerV2", engine="Run4DecisionEngineV2",
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", fallback="mode11_q9800_a71"))
