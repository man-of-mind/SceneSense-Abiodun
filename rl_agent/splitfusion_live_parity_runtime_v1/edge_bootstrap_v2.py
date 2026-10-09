"""Edge-only bootstrap repair, NOT activated by the sealed first attempt.

The shared repair composition eagerly imports UE-only source geometry/SciPy.
Keep its edge dispatch unchanged but avoid importing its UE half. This module
must be source-bound in a fresh prospective registration before live use.
"""
import json
from pathlib import Path
import sys

from . import contract as C


def run_edge_only(argv, *, max_frames):
    # Same dispatch as sealed composition.run_edge; no UE composition import.
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
    from . import pose
    from .entries import edge_close_evidence
    args = list(sys.argv[1:] if argv is None else argv)
    profile = EP.profile_path(args).parent
    registration = json.loads(Path("/partition_output/registration.json").read_text())
    if (registration.get("id") != C.ID or registration.get("transmitted_budget") != C.BUDGET
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("edge registration differs from repaired runtime")
    with pose.install_edge_pose(archive_directory=profile/"pose_archive", max_frames=C.BUDGET), \
            edge_close_evidence(profile):
        return run_edge_only(args, max_frames=C.BUDGET)
