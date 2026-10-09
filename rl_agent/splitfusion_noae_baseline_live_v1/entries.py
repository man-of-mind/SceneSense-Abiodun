"""Child and edge entry points for the noAE action-0 baseline (versioned copy of the gc_v2 entries; the ONLY addition is
the forced-action layer ``forced.install_forced_action``, entered innermost in the UE child exactly as in
``splitfusion_fixed_action_screen_v3.entries``, with the registered action-0 document).

Original gc_v2 header: Child and edge entry points for the paired live GC qualification (versioned copy of
``splitfusion_run5_full_live_v1.entries``; the ONLY addition is the UE-child GT-service trampoline seam).

The early-GT composition of ``splitfusion_fixed_action_screen_v3.entries`` WITHOUT the forced-action layer: the frozen
full Run-5 actor (via the full Run-5 child entry and policy seams) drives every decision. The UE pose bound and the exact-GT
limits follow the registered frame budget.
"""
import json
import os
from pathlib import Path
import sys

from . import contract as C


def _write(path, value):
    with open(path, "x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=str)
        stream.write("\n")


def registration(path, root):
    value = json.loads(Path(path).read_bytes())
    if (value.get("id") != C.ID or value.get("transmitted_budget") not in C.BUDGETS
            or value.get("g1_boundary_budget") != value.get("transmitted_budget")
            or value.get("exact_gt_limits") != C.limits(value.get("transmitted_budget"))
            or value.get("child_module") != C.CHILD or value.get("edge_module") != C.EDGE
            or value.get("live_metric_contract") != C.metrics()
            or value.get("registration_sha256") != C.canonical({k: v for k, v in value.items() if k != "registration_sha256"})):
        raise ValueError("full Run-5 live entry requires its exact sealed registration")
    root = Path(root)
    for relative, digest in value["exact_gt_sources"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("registered source drift: " + relative)
    for name in ("actor.py", "runtime_policy.py", "child_entry.py", "entries.py", "contract.py"):
        if C.V1.PACKAGE + "/" + name not in value["exact_gt_sources"]:
            raise ValueError("registration omits full live source " + name)
    for name in ("contract.py", "entries.py", "gc_seams.py", "service_entry.py", "worker_entry.py", "gc_policy_live.py",
                 *C.GC_SOURCES, *C.PRODUCER_SOURCES):
        key = name if "/" in name else C.PACKAGE + "/" + name
        if key not in value["exact_gt_sources"]:
            raise ValueError("registration omits GC-qualification source " + key)
    gc_reg = value.get("gt_gc") or {}
    if gc_reg != C.gc_registration(gc_reg.get("arm"), value["transmitted_budget"], gc_reg.get("producer")):
        raise ValueError("registration GC arm/limits differ from the contract")
    C.check_forced_shape(value.get("forced_action"))
    if C.PACKAGE + "/forced.py" not in value["exact_gt_sources"]:
        raise ValueError("registration omits the forced-action source")
    return value


def child(argv=None):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_run5_full_live_v1 import child_entry as CHILD
    from .gc_seams import install_ue_service_seam
    from rl_agent.splitfusion_live_parity_repair_v1.gt_bridge import bind_legacy_namespace
    from rl_agent.splitfusion_live_parity_repair_v1.runtime import install_ue_perception
    from rl_agent.splitfusion_live_parity_runtime_v1.pose import install_ue_pose
    from rl_agent.splitfusion_live_parity_runtime_v8.startup import preserve_startup
    from rl_agent.splitfusion_numeric_gt_runtime_v1.live_entries import preserve_numeric_evidence
    from rl_agent.splitfusion_ue_input_prep_v1.candidate import install_bgr2rgb_candidate
    from rl_agent.splitfusion_ue_input_prep_v2.candidate import install_fused_candidate
    from rl_agent.splitfusion_exact_gt_live_v2.ue import install_early_gt_ue
    from .forced import install_forced_action
    from .host import verify_prospective
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = Path(BASE.attempt_dir_from_argv(args)).resolve(strict=True)
    config = BASE.load_config(attempt)
    if (config is None or config.get("qualification_id") != C.ID
            or config.get("sha256") != C.canonical({k: v for k, v in config.items() if k != "sha256"})):
        raise ValueError("sealed enabled full Run-5 live profiler configuration required")
    output = attempt.parent.parent.parent
    verify_prospective(output)
    root = Path(__file__).resolve().parents[2]
    reg = registration(output / "registration.json", root)
    bind_legacy_namespace()
    original = CHILD.EDGE_MODULE
    CHILD.EDGE_MODULE = C.EDGE
    try:
        with install_ue_service_seam(registration=reg, output=output), \
                preserve_numeric_evidence(), preserve_startup(attempt), install_ue_perception(), \
                install_bgr2rgb_candidate(), install_fused_candidate(), \
                install_ue_pose(attempt_dir=attempt, max_frames=int(reg["transmitted_budget"])), \
                install_early_gt_ue(attempt_dir=attempt, registration=reg), \
                install_forced_action(log_path=attempt / "forced_action" / "FORCED_ACTIONS.jsonl",
                                      forced=reg["forced_action"]):
            return CHILD.main(args)
    finally:
        CHILD.EDGE_MODULE = original


def edge(argv=None):
    from rl_agent.splitfusion_numeric_gt_runtime_v1.live_entries import handoff_numeric_evidence, _run_edge_ordered
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[2]
    reg = registration("/partition_output/registration.json", root)
    profile = EP.profile_path(args).parent
    directory = profile / "numeric_runtime"
    directory.mkdir(parents=True, exist_ok=False)
    result, error, state = None, None, None
    try:
        sys.path.insert(0, "/cpu_partition_code")
        import edge_gate
        edge_gate.install("/partition_output", PW)
        base = Path("/work/torch_cache/segmentation_evidence")
        with install_early_gt_edge(socket_path=base / reg["exact_gt_socket_name"],
                                   capability_path=base / reg["exact_gt_capability_name"]) as state:
            result = _run_edge_ordered(ORDERED.main, args)
            instance = EarlyGtEvaluatorV1._instance
            if instance is not None:
                state["evaluator_counters"] = {k: v for k, v in instance.counters.items()}
                state["frames"] = {str(k): v for k, v in instance._frames.items()}
                state["records"] = [dict(r, measurement=None) for r in instance.records]
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write(directory / "EDGE_EARLY_GT.json", dict(state=state, returncode=result, error=error, pid=os.getpid()))
        _write(profile / "numeric_edge_scope.json", dict(schema="splitfusion.noae_baseline_live_v1.edge_scope.v1",
            returncode=result, error=error, live_qualified=False))
        handoff_numeric_evidence(directory, profile / "numeric_edge_scope.json",
                                 publisher_uid=reg["publisher_uid"], publisher_gid=reg["publisher_gid"])
