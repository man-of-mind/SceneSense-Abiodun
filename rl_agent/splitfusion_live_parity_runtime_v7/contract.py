"""Additive source identity; retained process/scientific settings are unchanged."""
from pathlib import Path
import json
from rl_agent.splitfusion_live_parity_runtime_v6 import contract as V4

PACKAGE = "rl_agent/splitfusion_live_parity_runtime_v7"
REPAIR_PACKAGE = "rl_agent/splitfusion_live_parity_repair_v2"
OPTIMIZATION_PACKAGE = "rl_agent/splitfusion_live_gt_optimization_v2"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_parity_v7"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_parity_v7"
RUNNER = "rl_agent.splitfusion_live_parity_runtime_v7.runner"
PARTITION = "rl_agent.splitfusion_live_parity_runtime_v7.partition_apply"
ID = "RUN5_CPU_GT_WORKER_100_ENGINEERING_V7"
BUDGET = 100
TOKEN = "RUN5_CPU_GT_WORKER_100_ONCE_EXECUTE_V7"
WORKER_PACKAGES = ("rl_agent/splitfusion_live_gt_optimization_v3", "rl_agent/splitfusion_live_gt_optimization_v4")
WORKER_TIMEOUT_S = 30.0
WORKER_BINDING_SHA256 = "d272c1234494a9a544046a3d7fd34be73031f1ecee3d55f3bd6643a4cb4ca364"
INPUT_MAX_FRAMES = 50
INPUT_MAX_BYTES = 512 * 1024 * 1024
REPLAY_PACKAGES = ("rl_agent/splitfusion_paired_quality_replay_v1", "rl_agent/splitfusion_paired_quality_replay_v2")
INPUT_SPEC = {"max_frames": INPUT_MAX_FRAMES, "max_bytes": INPUT_MAX_BYTES,
    "persistence": "AFTER_PREPARATION_AND_GT_WORKERS_DRAIN", "no_extra_device_transfer": True,
    "timing_improvement_claim": False, "required_selected_payload_and_prediction_parity": True}
IMAGE_ID = V4.IMAGE_ID
sha = V4.sha
canonical = V4.canonical
metrics = V4.metrics
OPTIMIZATION_SOURCES = {
    "__init__.py": "91dddbd7941c87de28b28946853bf7bd46e7fc893b23bddd57c2496b92655e1c",
    "benchmark.py": "562c5d5dc6b517b0467659786040ee855e79435bc02f3eebae735b1374017b86",
    "candidate.py": "30ba50410eb4c61e30a32f6e5cfcaff93acc436ec3e439c48c3c404cffeafd5c",
    "profile_saved.py": "e1f268ec86a2e33399f9a1ea9aa0e45a94618a442c56d2600cbeca0e79f2cfbd",
    "runtime.py": "33da88ea6395382c902766ae5c49f27d33f3735f2b7fdd22b653bec035828615",
    "selected_replay.py": "f2f7c404621615314d2e0e86445e2b3723ef73461db035168b0ed6fc8f69b248",
    "verify_saved_scores.py": "43069c927a74492bf6f196549fed2016370ab9f43e8de7d243ce51268edfc9ea"
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
    result = V4.source_inventory(root)
    optimization_inventory(root)
    worker_inventory(root)
    folders = [root/PACKAGE, root/OPTIMIZATION_PACKAGE, *(root/p for p in REPLAY_PACKAGES), *(root/p for p in WORKER_PACKAGES)]
    if any(not (folder/"__init__.py").is_file() or folder.is_symlink() for folder in folders):
        raise ValueError("required optimization/runtime5 package missing")
    paths = {p for folder in folders for p in folder.rglob("*")
             if p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
    paths.update(root/(module.replace(".","/")+".py") for module in (CHILD,EDGE))
    if any(not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root)
           or any(parent.is_symlink() for parent in p.parents if parent != root and parent.is_relative_to(root)) for p in paths):
        raise ValueError("unsafe/incomplete runtime5 inventory")
    result.update({str(p.relative_to(root)):sha(p) for p in sorted(paths)})
    return dict(sorted(result.items()))


def lineage(root, actor_boundary):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root),
        optimization_sources=optimization_inventory(root), worker_sources=worker_inventory(root), metrics=metrics(),
        controller="RewardHoldControllerV2",engine="Run4DecisionEngineV2",
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE",fallback="mode11_q9800_a71"))


def worker_inventory(root):
    root = Path(root).resolve(strict=True)
    binding_path = root/PACKAGE/"worker_binding.json"
    if binding_path.is_symlink() or sha(binding_path) != WORKER_BINDING_SHA256:
        raise ValueError("CPU worker proof binding changed")
    binding = json.loads(binding_path.read_bytes())
    if binding.get("schema") != "splitfusion.cpu_gt_worker.proof_binding.v7":
        raise ValueError("foreign CPU worker proof binding")
    expected = binding["worker_sources"]
    packages = (*WORKER_PACKAGES, "rl_agent/splitfusion_live_gt_optimization_v1",
        "rl_agent/splitfusion_live_gt_optimization_v2", "rl_agent/splitfusion_live_parity_repair_v1",
        "rl_agent/splitfusion_live_parity_repair_v2", "rl_agent/splitfusion_run5_live_avo_v1")
    paths = [p for package in packages for p in (root/package).glob("*.py") if not p.name.startswith("test_")]
    if any(p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root) for p in paths):
        raise ValueError("unsafe CPU worker sources")
    actual = {str(p.relative_to(root)):sha(p) for p in paths}
    if actual != expected or binding.get("exact_replays") != 288 or binding.get("live_qualified") is not False:
        raise ValueError("CPU worker differs from sealed offline proof")
    return dict(sorted(actual.items()))
