"""Process-local integration around retained Run5 child/partition entry points.

These functions do not select/authorize a live configuration. A prospective
runner must bind this package, the new edge trampoline and the unchanged actor
before calling them. No default to the old raw-GT evaluator is permitted.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

from . import gt_bridge, runtime


def preserve_prediction_archive(edge_scratch, destination):
    """Copy AFTER retained edge stop/drain, BEFORE scratch teardown."""
    if not edge_scratch:
        raise ValueError("edge scratch missing; prediction evidence not preserved")
    source = Path(edge_scratch) / "profiler" / "prediction_archive"
    if not source.is_dir() or source.is_symlink():
        raise ValueError("complete edge prediction archive missing")
    manifest = json.loads((source / "MANIFEST.json").read_bytes())
    if set(p.name for p in source.iterdir()) != set(manifest["sha256"]) | {"MANIFEST.json"}:
        raise ValueError("prediction archive inventory mismatch")
    for name, digest in manifest["sha256"].items():
        path = source / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("unsafe prediction evidence path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("edge prediction evidence changed")
    destination = Path(destination)
    shutil.copytree(source, destination)
    for path in source.iterdir():
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256((destination/path.name).read_bytes()).digest():
            raise ValueError("prediction evidence copy differs")
    return {"directory": str(destination), "frames": manifest["frames"],
        "manifest_sha256": hashlib.sha256((destination/"MANIFEST.json").read_bytes()).hexdigest(),
        "preserved_before_teardown": True}


@contextmanager
def ue_scope(*, attempt_dir: Path, max_frames: int):
    """Install before old Run5 main; preserve all new evidence on old stop path."""
    gt_bridge.bind_legacy_namespace()
    from rl_agent.splitfusion_single_gpu_profiler_v1 import seams_ue_v1 as SU
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    old_copy = SU.copy_edge_profile
    old_gt = (LG._INSTALLED, P.semantic_gt_3class, Q.QualitySceneSnapshotSource,
              Q.QualityPassiveSplitCollector, P.SceneSnapshotSource)

    def copy(edge_scratch, target):
        result = old_copy(edge_scratch, target)
        # On a setup failure with no edge archive, record explicit absence;
        # postflight rejects it. Never hide a corrupt existing archive.
        source = None if not edge_scratch else Path(edge_scratch)/"profiler"/"prediction_archive"
        if source is None or not source.exists():
            return {**result, "parity_predictions_preserved": False, "reason": "no completed archive"}
        value = preserve_prediction_archive(edge_scratch, Path(target)/"prediction_archive")
        return {**result, "parity_predictions_preserved": True, "parity_predictions": value}

    SU.copy_edge_profile = copy
    try:
        with runtime.install_ue_perception(), gt_bridge.install_complete_gt(
                attempt_dir=attempt_dir, max_frames=max_frames) as state:
            yield state
    finally:
        SU.copy_edge_profile = old_copy
        (LG._INSTALLED, P.semantic_gt_3class, Q.QualitySceneSnapshotSource,
         Q.QualityPassiveSplitCollector, P.SceneSnapshotSource) = old_gt


def run_ue_child(argv, *, max_frames, edge_module):
    """Invoke retained child with an explicitly registered repaired edge module.

    The edge trampoline must call run_edge below and remain identifiable to the
    retained sampler. Deliberately no guessed default or standalone live CLI.
    """
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_run5_live_avo_v1 import child_entry as CHILD
    attempt = BASE.attempt_dir_from_argv(argv)
    if attempt is None or not str(edge_module).startswith("rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_"):
        raise ValueError("owned attempt and explicit sampler-visible edge trampoline required")
    if edge_module == CHILD.EDGE_MODULE:
        raise ValueError("legacy raw-GT edge is not the parity-repair edge")
    original = CHILD.EDGE_MODULE
    CHILD.EDGE_MODULE = edge_module
    try:
        with ue_scope(attempt_dir=attempt, max_frames=max_frames):
            return CHILD.main(argv)
    finally:
        CHILD.EDGE_MODULE = original


def run_edge(argv, *, max_frames):
    """Called by the registered trampoline; retained partition/profiler/order."""
    import sys
    sys.path.insert(0, "/cpu_partition_code")
    import edge_gate
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
    edge_gate.install("/partition_output", PW)
    with runtime.install_edge_evaluation(
            archive_directory=EP.profile_path(argv).parent/"prediction_archive", max_frames=max_frames):
        return ORDERED.main(argv)
