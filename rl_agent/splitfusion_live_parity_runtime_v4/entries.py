"""Optimized CPU GT scope around the retained, sealed v3 runtime order."""
import json
from pathlib import Path
import sys

from . import contract as C
from rl_agent.splitfusion_live_parity_runtime_v1.entries import (
    write_x, evidence_scope, edge_close_evidence,
)


def _scope_report(record, prospective_sha256, optimization, ownership, result, error):
    drain = ownership["inspect_drain"]() if ownership is not None else {}
    manifest = ownership.get("manifest") if ownership is not None else None
    sources = record["optimization_sources"]
    valid = (optimization is not None and optimization.get("sources") == sources
        and optimization.get("retained_settings_unchanged") is True
        and optimization.get("settings_before") == optimization.get("settings_after")
        and optimization.get("source_integrity_error") is None)
    drained = drain.get("worker_drained") is True
    complete = isinstance(manifest, dict) and manifest.get("complete") is True
    return {"schema": "splitfusion.parity_runtime.optimized_gt_scope.v4",
        "status": "COMPLETE" if result == 0 and error is None and valid and drained and complete else "FAILED",
        "optimization_sources": dict(sources),
        "prospective_record_sha256": prospective_sha256,
        "optimization": optimization, "worker_drained": drained,
        "ownership_manifest_complete": complete, "returncode": result, "error": error}


def child(argv=None):
    from rl_agent.splitfusion_live_parity_repair_v1 import composition
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    from rl_agent.splitfusion_live_parity_repair_v2.source_ownership import install_source_ownership
    from rl_agent.splitfusion_live_parity_repair_v2.gt_timing import install_gt_timing
    from .optimization_scope import install_cpu_gt_optimizations
    from .startup import preserve_startup
    from .host_capture import verify_prospective
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = BASE.attempt_dir_from_argv(args)
    config = BASE.load_config(attempt)
    if config is None or config.get("qualification_id") != C.ID:
        raise ValueError("enabled, sealed parity profiler configuration required")
    if C.canonical({k: v for k, v in config.items() if k != "sha256"}) != config.get("sha256"):
        raise ValueError("profiler configuration digest mismatch")
    capture = Path(attempt).resolve().parent.parent
    record = verify_prospective(capture.parent)
    prospective_sha256 = C.sha(capture.parent / "PROSPECTIVE.json")
    if record["budget"] != C.BUDGET:
        raise ValueError("child archive bound differs from registration")
    sources = record.get("optimization_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("prospectively bound optimization source inventory required")
    optimization = ownership = result = error = None
    try:
        with install_cpu_gt_optimizations(expected_sources=dict(sources)) as optimization:
            with install_source_ownership(attempt_dir=attempt, max_frames=C.BUDGET) as ownership, \
                    install_gt_timing(attempt_dir=attempt, max_frames=C.BUDGET,
                        drain_check=lambda: ownership["inspect_drain"]()["worker_drained"]), \
                    pose.install_ue_pose(attempt_dir=attempt, max_frames=C.BUDGET), \
                    evidence_scope(attempt), preserve_startup(attempt):
                result = composition.run_ue_child(args, max_frames=C.BUDGET, edge_module=C.EDGE)
        if result == 0 and _scope_report(record, prospective_sha256, optimization, ownership, result, None)["status"] != "COMPLETE":
            raise RuntimeError("optimized GT scope did not close with verified settings and drained ownership")
    except BaseException as exc:
        error = {"type": type(exc).__name__, "error": str(exc)}
        raise
    finally:
        # All context managers have exited; never turn partial state into a
        # normal drain or replace an authoritative runtime exception.
        try:
            write_x(Path(attempt) / "GT_OPTIMIZATION_SCOPE.json",
                    _scope_report(record, prospective_sha256, optimization, ownership, result, error))
        except Exception as evidence_error:
            if error is None:
                raise
            print(f"GT optimization evidence failure: {evidence_error!r}", file=sys.stderr, flush=True)
    return result


def run_edge_only(argv, *, max_frames):
    # Keep the retained edge-only dispatch: never import the UE composition.
    sys.path.insert(0, "/cpu_partition_code")
    import edge_gate
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
    from rl_agent.splitfusion_live_parity_repair_v1 import runtime
    edge_gate.install("/partition_output", PW)
    with runtime.install_edge_evaluation(
            archive_directory=EP.profile_path(argv).parent / "prediction_archive", max_frames=max_frames):
        return ORDERED.main(argv)


def edge(argv=None):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    from rl_agent.splitfusion_live_parity_repair_v2.runtime import install_prediction_contract
    args = list(sys.argv[1:] if argv is None else argv)
    profile = EP.profile_path(args).parent
    registration = json.loads(Path("/partition_output/registration.json").read_text())
    if (registration.get("id") != C.ID or registration.get("transmitted_budget") != C.BUDGET
            or registration.get("edge_module") != C.EDGE
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("edge registration differs from repaired runtime")
    with install_prediction_contract(), \
            pose.install_edge_pose(archive_directory=profile / "pose_archive", max_frames=C.BUDGET), \
            edge_close_evidence(profile):
        return run_edge_only(args, max_frames=C.BUDGET)
