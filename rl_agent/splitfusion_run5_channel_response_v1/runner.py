"""Channel response v1: ``select_channel`` substitutes phase6_live_runner_v2.PROFILE (FAVORABLE_STABLE ->
FADE_RECOVERY) and the preflight verifies the single resolved carrier cell. Original header:

Retained lifecycle and instrumentation with the GC-qualification binding (+ the pre-launch retention plan digest in the lineage) (versioned copy of the full Run-5 runner;
the lineage additionally binds the registered GT GC arm)."""
import json
from pathlib import Path
import sys

from . import contract as C


def select_channel(R, sealed, root):
    """Bind the sealed channel, substitute the runner's PROFILE constant once and return the resolved-cell check."""
    if sealed.get("network_profile") != C.NETWORK_PROFILE or sealed.get("channel") != C.channel_registration(root):
        raise ValueError("sealed registration does not bind the registered channel")
    if R.PROFILE != C.ORIGINAL_PROFILE:
        raise ValueError("phase6 runner PROFILE seam drift: " + repr(R.PROFILE))
    R.PROFILE = C.NETWORK_PROFILE            # the ONLY functional change: select the pinned a71/FADE_RECOVERY cell

    def check(cells):
        selected = [(int(c.action_id), str(c.network_profile_id), str(c.trace_id), int(c.seed)) for c in cells]
        if selected != [(C.CARRIER_ACTION, C.NETWORK_PROFILE, C.TRACE["trace_id"], C.TRACE["seed"])]:
            raise ValueError("resolved carrier cell differs from the registered channel: " + repr(selected))
        return dict(cell_id=cells[0].cell_id, action_id=C.CARRIER_ACTION, network_profile_id=C.NETWORK_PROFILE,
                    trace_id=C.TRACE["trace_id"], seed=C.TRACE["seed"])
    return check


def install_parent(registration):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
    from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
    from .host import verify_prospective, verify_prerequisite
    output = Path(registration).parent
    record = verify_prospective(output)
    verify_prerequisite(output, record)
    root = Path(__file__).resolve().parents[2]
    if root != Path(record["worktree"]):
        raise ValueError("runner imported from foreign repository")
    actor = load_registered_actor()
    sealed = json.loads(Path(registration).read_bytes())
    digest = C.lineage(root, actor.boundary_sha256, sealed["support_sha256"], sealed["transmitted_budget"],
                       sealed["gt_gc"]["arm"], sealed["gt_gc"]["producer"], sealed["paired_retention"]["plan_sha256"])
    if sealed.get("controller_lineage_sha256") != digest or sealed.get("id") != C.ID:
        raise ValueError("foreign GC-qualification runtime lineage")
    check_cells = select_channel(R, sealed, root)
    original = R.offline_preflight
    R.controller_lineage_sha256 = lambda: digest

    def preflight(*args, **kwargs):
        config, cells, legacy = original(*args, **kwargs)
        legacy = dict(legacy, channel_cell=check_cells(cells))
        return config, cells, dict(schema="splitfusion.run5_channel_response_v1.live_preflight.v1",
            legacy_infrastructure_preflight=legacy, state_dim=22, actor_boundary_sha256=actor.boundary_sha256,
            controller_lineage_sha256=digest, live_metric_contract=C.metrics(), forced_action=None,
            budget=sealed["transmitted_budget"], gt_gc_arm=sealed["gt_gc"]["arm"], numeric_producer=sealed["gt_gc"]["producer"], retention_plan_sha256=sealed["paired_retention"]["plan_sha256"], live_command_executed=False, cuda_initialized=False)
    R.offline_preflight = preflight


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args.count("--registration") != 1 or args.index("--registration") + 1 >= len(args):
        raise ValueError("explicit sealed registration required")
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
