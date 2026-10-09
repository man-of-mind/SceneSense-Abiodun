"""Additive source identity; retained process/scientific settings are unchanged."""
from pathlib import Path
from rl_agent.splitfusion_live_parity_runtime_v3 import contract as V3

PACKAGE = "rl_agent/splitfusion_live_parity_runtime_v4"
REPAIR_PACKAGE = "rl_agent/splitfusion_live_parity_repair_v2"
OPTIMIZATION_PACKAGE = "rl_agent/splitfusion_live_gt_optimization_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_parity_v4"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_parity_v4"
RUNNER = "rl_agent.splitfusion_live_parity_runtime_v4.runner"
PARTITION = "rl_agent.splitfusion_live_parity_runtime_v4.partition_apply"
ID = "RUN5_COMPLETE_GT_CPU_OPTIMIZATION_100_ENGINEERING_V4"
BUDGET = 100
TOKEN = "RUN5_COMPLETE_GT_CPU_OPTIMIZATION_100_ONCE_EXECUTE_V4"
IMAGE_ID = V3.IMAGE_ID
sha = V3.sha
canonical = V3.canonical
metrics = V3.metrics
OPTIMIZATION_SOURCES = {
    "__init__.py": "d593dbf01a7a2def5f4c4d3dd03fc619783c85ff861342275d0e0ad4dfe503f1",
    "preservation.py": "a4a99062c055e37903a6c693c8a05a06d217c9ec65c02a919de932ab833d9557",
    "replay_combined.py": "f1e26f4fbed9ddac874970f57c75f2555a76b6c7c77a9ef1c9c8b06a48839d65",
    "replay_source.py": "093d6b5516d4ea5f780e9c3380d7d62423c8f8bc289b020241624936f6f8aa9b",
    "replay_target.py": "112f32a8f4daf5ded6101d54c81adc443e963db6e51807ca53e7279106077c7a",
    "runtime.py": "b680f5c66fe2af1873a284911803f3af3a15a21ecc43bbc249e5c19e5c413049",
    "source_geometry.py": "47d0aec30d433b11db899e9a83a56c05d1994af631d6d90156dd1ed81299ff4d",
    "target_builder.py": "a6b013d4fc18ba052fe5fd0a6073f5fe80fe4ce570fcb2b61e66edfd465953ce",
}


def optimization_inventory(root):
    folder = Path(root).resolve(strict=True) / OPTIMIZATION_PACKAGE
    paths = [p for p in folder.glob("*.py") if not p.name.startswith("test_")]
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("regular optimization sources required")
    result = {p.name:sha(p) for p in paths}
    if result != OPTIMIZATION_SOURCES:
        raise ValueError("candidate differs from exactly replayed optimization")
    return dict(sorted(result.items()))


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = V3.source_inventory(root)
    optimization_inventory(root)
    folders = [root/PACKAGE, root/OPTIMIZATION_PACKAGE]
    if any(not (folder/"__init__.py").is_file() or folder.is_symlink() for folder in folders):
        raise ValueError("required optimization/runtime4 package missing")
    paths = {p for folder in folders for p in folder.rglob("*")
             if p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
    paths.update(root/(module.replace(".","/")+".py") for module in (CHILD,EDGE))
    if any(not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root)
           or any(parent.is_symlink() for parent in p.parents if parent != root and parent.is_relative_to(root)) for p in paths):
        raise ValueError("unsafe/incomplete runtime4 inventory")
    result.update({str(p.relative_to(root)):sha(p) for p in sorted(paths)})
    return dict(sorted(result.items()))


def lineage(root, actor_boundary):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root),
        optimization_sources=optimization_inventory(root), metrics=metrics(),
        controller="RewardHoldControllerV2",engine="Run4DecisionEngineV2",
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE",fallback="mode11_q9800_a71"))
