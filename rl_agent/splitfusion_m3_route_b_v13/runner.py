"""M3.3 v13 runner: copy of the v8 runner (native, not a facade over v3 ``main``) with the v13 per-role binding.

Two additions, both in the runner parent and both evidence-only: the unchanged v3 MAC-log ``radio_observer`` keeps
its separately labelled artefact, and ``radio_telemetry`` collects the calibrated per-transmission PUSCH SNR,
scheduler ``avg_snr`` and actual MCS from the gNB T-tracer port, placed on the registered radio CPUs.  Neither
touches the inference, GT, scoring or map path, and neither reaches the actor.

The v8 header follows.  ``validate_parent`` holds every registration check the runner performs; ``install_parent`` calls it before
installing any retained runner seam, and ``--verify-only`` calls it (plus the retained child-module import check of
the profiler preflight) and stops.  Runner seams, channel/carrier check, route gates and the radio observer are
unchanged from v3/v5.
"""
import importlib
import json
from pathlib import Path
import sys

from . import contract as C
from rl_agent.splitfusion_m3_route_b_v3 import route_completion as RC


def select_channel(R, sealed, root):
    if sealed.get("network_profile") != C.NETWORK_PROFILE or sealed.get("channel") != C.channel_registration(root):
        raise ValueError("sealed registration does not bind the registered GOOD_TO_BAD schedule / carrier cell")
    if R.PROFILE != C.ORIGINAL_PROFILE:
        raise ValueError("phase6 runner PROFILE seam drift: " + repr(R.PROFILE))
    carrier = C.CARRIER_TRACE

    def check(cells):
        selected = [(int(c.action_id), str(c.network_profile_id), str(c.trace_id), int(c.seed)) for c in cells]
        expected = [(C.CARRIER_ACTION, carrier["profile_id"], carrier["trace_id"], carrier["seed"])]
        if selected != expected:
            raise ValueError("resolved carrier cell differs from the registered carrier: " + repr(selected))
        return dict(cell_id=cells[0].cell_id, action_id=C.CARRIER_ACTION,
                    network_profile_id=carrier["profile_id"], carrier_trace_id=carrier["trace_id"],
                    seed=carrier["seed"], commanded_schedule=C.SCHEDULE["schedule_id"],
                    schedule_trace_id=C.SCHEDULE["trace_id"])
    return check


def install_gates():
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
    from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_v1 as RV
    if (getattr(RV.g1_frame_accounting, "__m3_route_branch__", False)
            or getattr(R.evaluate_phase6, "__m3_final_cycle__", False)):
        raise RuntimeError("M3 runner gates already installed")
    RV.g1_frame_accounting = RC.g1_route_branch(RV.g1_frame_accounting)
    R.evaluate_phase6 = RC.p3_final_cycle(R.evaluate_phase6)


def validate_parent(registration):
    """Every registration/verifier check of the runner parent (shared by launch and ``--verify-only``)."""
    from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
    from .host import verify_prerequisite, verify_prospective
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
    if (sealed.get("controller_lineage_sha256") != digest or sealed.get("id") != C.ID
            or sealed.get("launcher_binding") != C.launcher_binding()):
        raise ValueError("foreign M3 v13 launcher/runtime lineage")
    resolved = dict(validate_parent=C.module_identity(validate_parent),
                    verify_prospective=C.module_identity(verify_prospective),
                    verify_prerequisite=C.module_identity(verify_prerequisite), contract=C.__name__)
    return dict(record=record, sealed=sealed, digest=digest, actor=actor, root=root, resolved=resolved)


def install_parent(registration, output_root=None):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
    checked = validate_parent(registration)
    sealed, digest, actor, root = checked["sealed"], checked["digest"], checked["actor"], checked["root"]
    print("M3_VERSION_BINDING=" + json.dumps(dict(C.launcher_binding(), resolved=checked["resolved"]),
                                             sort_keys=True), flush=True)
    check_cells = select_channel(R, sealed, root)
    install_gates()
    if output_root is not None:
        from rl_agent.splitfusion_m3_route_b_v3 import radio_observer
        from . import radio_telemetry as RT
        radio_observer.install(output_root)                       # MAC-log power-control statistic (separate label)
        RT.install(output_root, RT.radio_cpus(output_root))       # calibrated per-transmission PUSCH SNR / MCS
    original = R.offline_preflight
    R.controller_lineage_sha256 = lambda: digest

    def preflight(*args, **kwargs):
        config, cells, legacy = original(*args, **kwargs)
        legacy = dict(legacy, channel_cell=check_cells(cells))
        return config, cells, dict(
            schema="splitfusion.m3_route_b_v13.live_preflight.v1",
            legacy_infrastructure_preflight=legacy, state_dim=22,
            actor_boundary_sha256=actor.boundary_sha256,
            controller_lineage_sha256=digest, live_metric_contract=C.metrics(), forced_action=None,
            frame_ceiling=sealed["transmitted_budget"], route=C.route_registration(root),
            schedule=C.channel_registration(root)["schedule"], gt_gc_arm=sealed["gt_gc"]["arm"],
            radio_telemetry=C.radio_telemetry_registration(root), bookkeeping=C.bookkeeping_registration(),
            numeric_producer=sealed["gt_gc"]["producer"],
            retention_plan_sha256=sealed["paired_retention"]["plan_sha256"],
            launcher_binding=C.launcher_binding(), live_command_executed=False, cuda_initialized=False)
    R.offline_preflight = preflight
    return checked


def verify_only(registration):
    checked = validate_parent(registration)
    importlib.import_module(checked["sealed"]["child_module"])     # retained profiler preflight import check
    torch = sys.modules.get("torch")
    document = dict(role="runner_parent", status="PASS", resolved=checked["resolved"], contract_id=C.ID,
                    record_schema=checked["record"]["schema"], child_module=checked["sealed"]["child_module"],
                    lineage_sha256=checked["digest"], seams_installed=False,
                    cuda_initialized=bool(torch is not None and torch.cuda.is_initialized()))
    print("M3_V13_STARTUP_VERIFICATION=" + json.dumps(document, sort_keys=True), flush=True)
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args.count("--registration") != 1 or args.index("--registration") + 1 >= len(args):
        raise ValueError("explicit sealed registration required")
    registration = Path(args[args.index("--registration") + 1])
    if "--verify-only" in args:
        return verify_only(registration)
    output_root = (Path(args[args.index("--output-root") + 1]) if args.count("--output-root") == 1
                   and "--preflight" not in args and "--execute" in args else None)
    install_parent(registration, output_root)
    from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_cpu_partition_v1 as retained
    before = sys.argv
    sys.argv = [before[0], *args]
    try:
        return retained.main()
    finally:
        sys.argv = before


if __name__ == "__main__":
    raise SystemExit(main())
