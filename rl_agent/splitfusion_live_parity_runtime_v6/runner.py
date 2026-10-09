"""Prospectively bound parent; retained lifecycle and original Run5 actor."""
import json
from pathlib import Path
import sys
from . import contract as C


def install_parent(registration):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
    from rl_agent.splitfusion_run5_live_avo_v1.audit_actor import load_registered_actor
    from .host_capture import verify_prospective, verify_prerequisite
    record = verify_prospective(Path(registration).parent)
    verify_prerequisite(Path(registration).parent, record)
    root = Path(__file__).resolve().parents[2]
    if root != Path(record["worktree"]):
        raise ValueError("runner imported from a different repository")
    actor = load_registered_actor()
    digest = C.lineage(root, actor.boundary_sha256)
    sealed = json.loads(Path(registration).read_text())
    if sealed.get("controller_lineage_sha256") != digest or sealed.get("id") != C.ID:
        raise ValueError("unregistered repaired controller lineage")
    original = R.offline_preflight
    R.controller_lineage_sha256 = lambda: digest

    def preflight(*args, **kwargs):
        config, cells, legacy = original(*args, **kwargs)
        return config, cells, dict(schema="splitfusion.run5.parity.preflight.v6",
            legacy_infrastructure_preflight=legacy, state_dim=22,
            actor_boundary_sha256=actor.boundary_sha256, controller_lineage_sha256=digest,
            live_metric_contract=C.metrics(), live_command_executed=False, cuda_initialized=False)
    R.offline_preflight = preflight


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args.count("--registration") != 1 or args.index("--registration") + 1 >= len(args):
        raise ValueError("explicit sealed parity registration required")
    install_parent(Path(args[args.index("--registration") + 1]))
    from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_cpu_partition_v1 as R
    before = sys.argv
    sys.argv = [before[0], *args]
    try:
        return R.main()
    finally:
        sys.argv = before


if __name__ == "__main__":
    raise SystemExit(main())
