"""M3.1 runner seam (versioned copy of the channel-response runner).

* The carrier cell is the pinned a71/FAVORABLE_STABLE cell: ``phase6_live_runner_v2.PROFILE`` is verified UNCHANGED
  and the single resolved cell is checked (action, profile, trace, seed); the GOOD_ALL schedule is applied inside the
  UE child (controller-module seam), never by substituting a stochastic profile.
* G1 (``runner_v1.g1_frame_accounting``) gains the registered ROUTE_COMPLETED branch; a frame-ceiling stop is an
  incomplete route. Phase-6 P3 (``phase6_live_runner_v2.evaluate_phase6``) gains the registered final-cycle
  exception. Both are wrappers resolved through the modules' globals; neither pinned file is edited.
* ``radio_observer`` (runner process only) tails the gNB MAC statistics and keeps the softmodem logs before the retained
  scratch removal (achieved-radio evidence, labelled separately from the commanded target).
"""
import json
from pathlib import Path
import sys

from . import contract as C
from . import route_completion as RC


def select_channel(R, sealed, root):
    """Bind the sealed GOOD_ALL schedule and verify the carrier cell without substituting the runner PROFILE."""
    if sealed.get("network_profile") != C.NETWORK_PROFILE or sealed.get("channel") != C.channel_registration(root):
        raise ValueError("sealed registration does not bind the registered GOOD_ALL schedule / carrier cell")
    if R.PROFILE != C.ORIGINAL_PROFILE:
        raise ValueError("phase6 runner PROFILE seam drift: " + repr(R.PROFILE))
    carrier = C.CARRIER_TRACE

    def check(cells):
        selected = [(int(c.action_id), str(c.network_profile_id), str(c.trace_id), int(c.seed)) for c in cells]
        if selected != [(C.CARRIER_ACTION, carrier["profile_id"], carrier["trace_id"], carrier["seed"])]:
            raise ValueError("resolved carrier cell differs from the registered carrier: " + repr(selected))
        return dict(cell_id=cells[0].cell_id, action_id=C.CARRIER_ACTION, network_profile_id=carrier["profile_id"],
                    carrier_trace_id=carrier["trace_id"], seed=carrier["seed"],
                    commanded_schedule=C.SCHEDULE["schedule_id"], schedule_trace_id=C.SCHEDULE["trace_id"])
    return check


def install_gates():
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
    from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_v1 as RV
    if getattr(RV.g1_frame_accounting, "__m3_route_branch__", False) or getattr(R.evaluate_phase6, "__m3_final_cycle__",
                                                                                False):
        raise RuntimeError("M3 runner gates already installed")
    RV.g1_frame_accounting = RC.g1_route_branch(RV.g1_frame_accounting)
    R.evaluate_phase6 = RC.p3_final_cycle(R.evaluate_phase6)


def install_parent(registration, output_root=None):
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
        raise ValueError("foreign M3 runtime lineage")
    check_cells = select_channel(R, sealed, root)
    install_gates()
    if output_root is not None:
        from . import radio_observer
        radio_observer.install(output_root)
    original = R.offline_preflight
    R.controller_lineage_sha256 = lambda: digest

    def preflight(*args, **kwargs):
        config, cells, legacy = original(*args, **kwargs)
        legacy = dict(legacy, channel_cell=check_cells(cells))
        return config, cells, dict(schema="splitfusion.m3_route_b_v2.live_preflight.v1",
            legacy_infrastructure_preflight=legacy, state_dim=22, actor_boundary_sha256=actor.boundary_sha256,
            controller_lineage_sha256=digest, live_metric_contract=C.metrics(), forced_action=None,
            frame_ceiling=sealed["transmitted_budget"], route=C.route_registration(root),
            schedule=C.channel_registration(root)["schedule"], gt_gc_arm=sealed["gt_gc"]["arm"],
            numeric_producer=sealed["gt_gc"]["producer"], retention_plan_sha256=sealed["paired_retention"]["plan_sha256"],
            live_command_executed=False, cuda_initialized=False)
    R.offline_preflight = preflight


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args.count("--registration") != 1 or args.index("--registration") + 1 >= len(args):
        raise ValueError("explicit sealed registration required")
    output_root = (Path(args[args.index("--output-root") + 1]) if args.count("--output-root") == 1
                   and "--preflight" not in args and "--execute" in args else None)
    install_parent(Path(args[args.index("--registration") + 1]), output_root)
    from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_cpu_partition_v1 as R
    before = sys.argv
    sys.argv = [before[0], *args]
    try:
        return R.main()
    finally:
        sys.argv = before


if __name__ == "__main__":
    raise SystemExit(main())
