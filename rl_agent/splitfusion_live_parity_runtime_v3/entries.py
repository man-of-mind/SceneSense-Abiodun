"""Versioned startup repair; numerical/runtime evaluation semantics unchanged."""
import json
from pathlib import Path
import sys
from . import contract as C
from rl_agent.splitfusion_live_parity_runtime_v1.entries import (
    write_x, evidence_scope, edge_close_evidence,
)


def child(argv=None):
    from rl_agent.splitfusion_live_parity_repair_v1 import composition
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    from .startup import preserve_startup
    from rl_agent.splitfusion_live_parity_repair_v2.source_ownership import install_source_ownership
    from rl_agent.splitfusion_live_parity_repair_v2.gt_timing import install_gt_timing
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = BASE.attempt_dir_from_argv(args)
    config = BASE.load_config(attempt)
    if config is None or config.get("qualification_id") != C.ID:
        raise ValueError("enabled, sealed parity profiler configuration required")
    if C.canonical({k: v for k, v in config.items() if k != "sha256"}) != config.get("sha256"):
        raise ValueError("profiler configuration digest mismatch")
    capture = Path(attempt).resolve().parent.parent
    from .host_capture import verify_prospective
    record = verify_prospective(capture.parent)
    if record["budget"] != C.BUDGET:
        raise ValueError("child archive bound differs from registration")
    with install_source_ownership(attempt_dir=attempt, max_frames=C.BUDGET) as ownership, \
            install_gt_timing(attempt_dir=attempt, max_frames=C.BUDGET,
                drain_check=lambda: ownership["inspect_drain"]()["worker_drained"]), \
            pose.install_ue_pose(attempt_dir=attempt, max_frames=C.BUDGET), \
            evidence_scope(attempt), preserve_startup(attempt):
        return composition.run_ue_child(args, max_frames=C.BUDGET, edge_module=C.EDGE)


def run_edge_only(argv, *, max_frames):
    # Preserve composition.run_edge's dispatch without importing its UE half.
    sys.path.insert(0, "/cpu_partition_code")
    import edge_gate
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
    from rl_agent.splitfusion_live_parity_repair_v1 import runtime
    edge_gate.install("/partition_output", PW)
    with runtime.install_edge_evaluation(
            archive_directory=EP.profile_path(argv).parent/"prediction_archive", max_frames=max_frames):
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
            pose.install_edge_pose(archive_directory=profile/"pose_archive", max_frames=C.BUDGET), \
            edge_close_evidence(profile):
        return run_edge_only(args, max_frames=C.BUDGET)
