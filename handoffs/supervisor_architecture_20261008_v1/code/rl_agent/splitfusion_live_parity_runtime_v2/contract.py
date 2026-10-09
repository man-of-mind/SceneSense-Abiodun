"""New source identity; the sealed v1 scientific contract remains unchanged."""
from pathlib import Path
from rl_agent.splitfusion_live_parity_runtime_v1 import contract as V1

PACKAGE = "rl_agent/splitfusion_live_parity_runtime_v2"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_parity_v2"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_parity_v2"
RUNNER = "rl_agent.splitfusion_live_parity_runtime_v2.runner"
PARTITION = "rl_agent.splitfusion_live_parity_runtime_v2.partition_apply"
ID = "RUN5_COMPLETE_GT_PARITY_100_ENGINEERING_V2"
BUDGET = 100
TOKEN = "RUN5_COMPLETE_GT_PARITY_100_ONCE_EXECUTE_V2"
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
    paths = {p for p in directory.rglob("*") if p.suffix in (".py", ".json", ".pt")
             and "__pycache__" not in p.parts}
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
