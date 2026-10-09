"""Source-bound, quality-aware original Run5 runtime; no training authority."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

PACKAGE = "rl_agent/splitfusion_live_parity_runtime_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_parity"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_parity"
RUNNER = "rl_agent.splitfusion_live_parity_runtime_v1.runner"
PARTITION = "rl_agent.splitfusion_live_parity_runtime_v1.partition_apply"
ID = "RUN5_COMPLETE_GT_PARITY_100_ENGINEERING_V1"
BUDGET = 100
TOKEN = "RUN5_COMPLETE_GT_PARITY_100_ONCE_EXECUTE"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def metrics():
    from rl_agent.splitfusion_live_parity_repair_v1 import perception, target_transport
    return dict(schema="splitfusion.run5.complete_gt_runtime.v1",
        perception=json.loads(json.dumps(asdict(perception.CONTRACT))),
        perception_contract_sha256=perception.CONTRACT.sha256,
        conflict_scope=target_transport.LIVE_SCOPE,
        person_visibility="ORIGINAL_UNNORMALIZED_AVO_GE_0_65",
        person_confidence_min=0.25, vehicle_confidence_min=0.20,
        previous_quality_in_state=True, actor_state_dim=22,
        quality_evaluation_inside_deadline=True, inclusive_deadline_ms=170,
        model_pose="OWNED_RGB_RADAR_SNAPSHOT_EGO__EDGE_TARGET_RECONCILED",
        segmentation="ORIGINAL_V010_DEPTH_SUPPORTED_POSITIVE_AND_IGNORE_RULES",
        historical_manual_quarantine_reused=False,
        policy_performance_claim=False, live_qualification_claim=False,
        training_started=False)


def source_inventory(root):
    """Complete additive package inventory plus exact historical source pins."""
    from rl_agent.splitfusion_live_parity_repair_v1 import perception, evaluation, source_geometry
    from rl_agent.splitfusion_live_target_parity_v1 import adapter
    from rl_agent.splitfusion_run5_live_avo_v1 import prepare_binding
    root = Path(root).resolve(strict=True)
    paths = set()
    for name in (PACKAGE, "rl_agent/splitfusion_live_parity_repair_v1",
                 "rl_agent/splitfusion_live_target_parity_v1", "rl_agent/splitfusion_run5_live_avo_v1"):
        directory = root / name
        if not directory.is_dir() or directory.is_symlink() or not (directory/"__init__.py").is_file():
            raise ValueError(f"required package missing: {name}")
        paths.update(p for p in directory.rglob("*") if p.suffix in (".py", ".json", ".pt")
                     and "__pycache__" not in p.parts)
    paths.update((root / "rl_agent/splitfusion_run5_live_avo_v1/actor_export").iterdir())
    for module in (adapter, source_geometry):
        for relative, digest in module.SOURCES.values():
            path = root / relative
            if sha(path) != digest:
                raise ValueError(f"historical source drift: {relative}")
            paths.add(path)
    for item in evaluation.verify_sources().values():
        path = root / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError("evaluation source drift")
        paths.add(path)
    for relative, digest in perception.verify_reference_sources(root).items():
        paths.add(root / relative)
    paths.update(root / name for name in prepare_binding.FROZEN_DEPENDENCIES)
    for module in (CHILD, EDGE):
        paths.add(root / (module.replace(".", "/") + ".py"))
    if not paths or any(not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root)
                        for p in paths):
        raise ValueError("incomplete/nonregular/escaping runtime inventory")
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def lineage(root, actor_boundary):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root),
        metrics=metrics(), controller="RewardHoldControllerV2", engine="Run4DecisionEngineV2",
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", fallback="mode11_q9800_a71"))
