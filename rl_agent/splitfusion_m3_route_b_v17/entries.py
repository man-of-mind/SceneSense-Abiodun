"""M3.7 v17 UE-child and edge entry points with explicit, version-coherent registration/verifier binding.

Versioned copy of ``splitfusion_m3_route_b_v11.entries`` (itself the v8 lineage: a copy of v3 ``child/edge/registration``
wrapped as v4 did, abort-evidence exporter around the child).  v17 differs from v11 ONLY in the contract it reads (the
OPERATING_14DB schedule and the v17 identities) and in the name of its native controller installer:

  * ``install_operating_14db_controller`` is native here and installs the **v17** OPERATING_14DB controller (constant 14.0 dB ->
    RFsim noise -7.0, Phase-14A anchor 06); the v11 installer would install the v11 23.5-dB GOOD_ALL controller.
    The installer body is the v11 one (same retained ``runtime_policy.CONTROLLER_MODULE`` seam).
  * the collector-layer scope is ``route_drain.install_collector_layer`` (the v11 bounded end-of-route terminal drain
    directly above the unchanged v3 ``M3CollectorLayer``), unchanged from v11.

Every other scope, their order and every runtime implementation are unchanged from v11; only the resolution of the
validating functions follows the package:

  v3/v4/v5 child: ``from .host import verify_prospective``   -> v3 host + v3 contract (the V5 failure)
                  ``from .retention import load_observer``   -> v3 plan id (would refuse the registered v4 plan)
                  ``from .gc_seams import ...``              -> v3 module whose global C was rebound in place by v4
  v17 child:       ``host.verify_prospective``, ``retention.load_observer``, ``gc_seams`` of THIS package, all reading
                  ``splitfusion_m3_route_b_v17.contract``; no global of an older module is rebound.

All registration checks run in ``validate_child_startup`` / ``validate_edge_startup`` BEFORE any production scope.
The verification-only entry modes (``verify_child_startup`` / ``verify_edge_startup``, reached through the real
sampler-visible trampolines with ``--m3-verify-startup``) call these same functions and then stop.
"""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys

from . import contract as C

# The v3 retained-sources clause, written out against the modules a v17 run actually executes.
REQUIRED_SOURCES = (
    *(f"{C.V1.PACKAGE}/{name}" for name in ("actor.py", "runtime_policy.py", "child_entry.py", "entries.py",
                                             "contract.py")),
    *(f"{C.PACKAGE}/{name}" for name in ("contract.py", "entries.py", "gc_seams.py", "retention.py", "host.py",
                                         "route_drain.py", "radio_telemetry.py", "operating_14db_controller.py",
                                         "postflight_p3.py")),
    # v17: the GT trampolines are the reviewed containment package's; they compose V16's accounting layer + timer
    *C.CONTAINMENT_QUALIFIED,
    *(f"{C.V16_PACKAGE}/{name}" for name in ("contract.py", "accounting_timing.py", "worker_entry.py",
                                              "service_entry.py")),
    *C.PREP_LIFETIME_QUALIFIED, *C.AVO_TILT_QUALIFIED, *C.ACCOUNTING_QUALIFIED, *C.ACCOUNTING_V1_QUALIFIED,
    *(f"{C.V8.PACKAGE}/{name}" for name in ("contract.py", "worker_entry.py", "service_entry.py")),
    *(f"{C.V7.PACKAGE}/{name}" for name in ("contract.py", "worker_entry.py")),
    *(f"rl_agent/splitfusion_m3_route_b_v4/{name}" for name in ("contract.py", "abort_evidence.py", "worker_entry.py")),
    *(f"rl_agent/splitfusion_m3_route_b_v3/{name}" for name in (
        "contract.py", "entries.py", "worker_entry.py", "gc_policy_live.py", "capacity.py",
        "pose_tolerance.py", "retention.py", "route_completion.py", "map_journal.py", "edge_guard.py",
        "good_all_controller.py", "map_class.py")),
    *C.GC_SOURCES, *C.PRODUCER_SOURCES, *C.RETENTION_SOURCES)

# Modules imported by the live child/edge before production starts; the verification modes import the same set.
CHILD_IMPORTS = (
    "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1", "rl_agent.splitfusion_run5_full_live_v1.child_entry",
    "rl_agent.splitfusion_live_parity_repair_v1.gt_bridge", "rl_agent.splitfusion_live_parity_repair_v1.runtime",
    "rl_agent.splitfusion_live_parity_runtime_v1.pose", "rl_agent.splitfusion_live_parity_runtime_v8.startup",
    "rl_agent.splitfusion_numeric_gt_runtime_v1.live_entries", "rl_agent.splitfusion_ue_input_prep_v1.candidate",
    "rl_agent.splitfusion_ue_input_prep_v2.candidate", "rl_agent.splitfusion_exact_gt_live_v2.ue",
    "rl_agent.splitfusion_m3_route_b_v3.route_completion", "rl_agent.splitfusion_m3_route_b_v3.entries",
    "rl_agent.splitfusion_m3_route_b_v4.abort_evidence", "rl_agent.splitfusion_paired_retention_v2.observer",
    "rl_agent.splitfusion_m3_route_b_v17.route_drain", "rl_agent.splitfusion_m3_route_b_v17.operating_14db_controller")
EDGE_IMPORTS = (
    "rl_agent.splitfusion_numeric_gt_runtime_v1.live_entries", "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1",
    "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_ordering",
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_prewarm_v2", "rl_agent.splitfusion_exact_gt_live_v2.edge",
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2",
    "rl_agent.splitfusion_m3_route_b_v3.edge_guard")


def registration(path, root):
    """The child/edge registration check (v3 ``registration`` reading the v17 contract)."""
    value = json.loads(Path(path).read_bytes())
    if (value.get("id") != C.ID or value.get("transmitted_budget") not in C.BUDGETS
            or value.get("g1_boundary_budget") != value.get("transmitted_budget")
            or value.get("exact_gt_limits") != C.limits(value.get("transmitted_budget"))
            or value.get("child_module") != C.CHILD or value.get("edge_module") != C.EDGE
            or value.get("live_metric_contract") != C.metrics()
            or value.get("registration_sha256") != C.canonical({k: v for k, v in value.items() if k != "registration_sha256"})):
        raise ValueError("full Run-5 live entry requires its exact sealed registration")
    if (value.get("launcher_revision") != C.LAUNCHER_REVISION or value.get("runtime_revision") != C.RUNTIME_REVISION
            or value.get("launcher_binding") != C.launcher_binding()):
        raise ValueError("registration launcher binding differs from the v17 contract")
    root = Path(root)
    for relative, digest in value["exact_gt_sources"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("registered source drift: " + relative)
    for key in REQUIRED_SOURCES:
        if key not in value["exact_gt_sources"]:
            raise ValueError("registration omits composition source " + key)
    gc_reg = value.get("gt_gc") or {}
    if gc_reg != C.gc_registration(gc_reg.get("arm"), value["transmitted_budget"], gc_reg.get("producer")):
        raise ValueError("registration GC arm/limits differ from the contract")
    if "forced_action" in value:
        raise ValueError("an actor-driven registration must not carry a forced action")
    retention = value.get("paired_retention") or {}
    if retention != C.retention_registration(retention.get("plan_sha256")):
        raise ValueError("registration retention section differs from the contract")
    return value


@contextmanager
def install_operating_14db_controller(attempt):
    """v11 ``install_good_all_controller`` reading the v17 contract: the retained ``runtime_policy`` launches the
    controller module named by its ``CONTROLLER_MODULE`` global, here the v17 constant-14.0-dB OPERATING_14DB controller."""
    from rl_agent.splitfusion_run5_full_live_v1 import runtime_policy as RP
    from rl_agent.splitfusion_m3_route_b_v3.entries import _write
    if RP.CONTROLLER_MODULE != C.RETAINED_CONTROLLER_MODULE:
        raise ValueError("retained controller module drift: " + repr(RP.CONTROLLER_MODULE))
    RP.CONTROLLER_MODULE = C.CONTROLLER_MODULE
    try:
        yield
    finally:
        RP.CONTROLLER_MODULE = C.RETAINED_CONTROLLER_MODULE
        directory = Path(attempt) / "m3_route"
        directory.mkdir(exist_ok=True)
        _write(directory / "CONTROLLER_MODULE_SEAM.json", dict(
            schema="splitfusion.m3_route_b_v17.controller_seam.v1", substituted=C.CONTROLLER_MODULE,
            retained=C.RETAINED_CONTROLLER_MODULE, restored=RP.CONTROLLER_MODULE == C.RETAINED_CONTROLLER_MODULE,
            target_snr_db=C.SCHEDULE["target_snr_db"],
            expected_rfsim_command_db=C.SCHEDULE["expected_rfsim_command_db"]))


def validate_child_startup(output, config, root):
    """Every registration/verifier check of the UE child, in the live order, before any production scope."""
    from .gc_seams import seam_parameters
    from .host import verify_prospective
    from .retention import load_observer
    if (config is None or config.get("qualification_id") != C.ID
            or config.get("sha256") != C.canonical({k: v for k, v in config.items() if k != "sha256"})):
        raise ValueError("sealed enabled full Run-5 live profiler configuration required")
    record = verify_prospective(output)
    reg = registration(Path(output) / "registration.json", root)
    observer = load_observer(output, reg)          # digest-bound pre-launch plan; refuses before any frame
    seam = seam_parameters(reg)
    return dict(record=record, registration=reg, observer=observer, seam=seam,
                resolved=dict(verify_prospective=C.module_identity(verify_prospective),
                              registration=C.module_identity(registration),
                              load_observer=C.module_identity(load_observer),
                              seam_parameters=C.module_identity(seam_parameters), contract=C.__name__))


def _child(argv=None):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_run5_full_live_v1 import child_entry as CHILD
    from rl_agent.splitfusion_live_parity_repair_v1.gt_bridge import bind_legacy_namespace
    from rl_agent.splitfusion_live_parity_repair_v1.runtime import install_ue_perception
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as POSE   # resolved after the P2 substitution
    from rl_agent.splitfusion_live_parity_runtime_v8.startup import preserve_startup
    from rl_agent.splitfusion_numeric_gt_runtime_v1.live_entries import preserve_numeric_evidence
    from rl_agent.splitfusion_ue_input_prep_v1.candidate import install_bgr2rgb_candidate
    from rl_agent.splitfusion_ue_input_prep_v2.candidate import install_fused_candidate
    from rl_agent.splitfusion_exact_gt_live_v2.ue import install_early_gt_ue
    from rl_agent.splitfusion_m3_route_b_v3 import route_completion as RC
    from rl_agent.splitfusion_m3_route_b_v3.entries import install_map_trampoline, install_ue_capacity
    from . import route_drain as RD
    from rl_agent.splitfusion_paired_retention_v2.observer import install_retention
    from .gc_seams import install_ue_service_seam
    from .retention import finalize_after_child
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = Path(BASE.attempt_dir_from_argv(args)).resolve(strict=True)
    config = BASE.load_config(attempt)
    output = attempt.parent.parent.parent
    root = Path(__file__).resolve().parents[2]
    checked = validate_child_startup(output, config, root)
    reg, observer = checked["registration"], checked["observer"]
    bind_legacy_namespace()
    original = CHILD.EDGE_MODULE
    CHILD.EDGE_MODULE = C.EDGE
    result, error = None, None
    try:
        with install_operating_14db_controller(attempt), RC.install_child_stop_rule(), install_map_trampoline(attempt), \
                install_ue_service_seam(registration=reg, output=output), \
                preserve_numeric_evidence(), preserve_startup(attempt), install_ue_perception(), \
                install_bgr2rgb_candidate(), install_fused_candidate(), install_ue_capacity(attempt), \
                POSE.install_ue_pose(attempt_dir=attempt, max_frames=int(reg["transmitted_budget"])), \
                RD.install_collector_layer(), \
                install_early_gt_ue(attempt_dir=attempt, registration=reg), \
                install_retention(observer):
            result = CHILD.main(args)
            return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        CHILD.EDGE_MODULE = original
        # every production scope has exited; persistence happens only after the owning frame worker is dead
        finalize_after_child(observer, output, child_returncode=result, child_error=error)


def child(argv=None):
    from rl_agent.splitfusion_m3_route_b_v4.abort_evidence import install_abort_evidence
    with install_abort_evidence():
        result = _child(argv)
    return result


def validate_edge_startup(path, root):
    """The edge's registration check, before any production scope."""
    return registration(path, root)


def edge(argv=None):
    from rl_agent.splitfusion_numeric_gt_runtime_v1.live_entries import handoff_numeric_evidence, _run_edge_ordered
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
    from rl_agent.splitfusion_m3_route_b_v3.entries import _write
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[2]
    reg = validate_edge_startup(C.EDGE_REGISTRATION_PATH, root)
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
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
            from rl_agent.splitfusion_m3_route_b_v3.edge_guard import build_guarded_evaluator
            if E.Run4EvaluatorV2 is not EarlyGtEvaluatorV1:
                raise RuntimeError("early-GT evaluator is not the installed edge evaluator")
            E.Run4EvaluatorV2 = build_guarded_evaluator(EarlyGtEvaluatorV1)
            state["m3_edge_guard"] = dict(installed=True, base="EarlyGtEvaluatorV1")
            result = _run_edge_ordered(ORDERED.main, args)
            instance = EarlyGtEvaluatorV1._instance
            if instance is not None:
                state["evaluator_counters"] = {k: v for k, v in instance.counters.items()}
                state["m3_edge_guard"]["guarded_errors"] = list(getattr(instance, "m3_guarded_errors", []))
                state["m3_edge_guard"]["instance_class"] = type(instance).__name__
                state["frames"] = {str(k): v for k, v in instance._frames.items()}
                state["records"] = [dict(r, measurement=None) for r in instance.records]
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write(directory / "EDGE_EARLY_GT.json", dict(state=state, returncode=result, error=error, pid=os.getpid()))
        _write(profile / "numeric_edge_scope.json", dict(schema="splitfusion.m3_route_b_v3.edge_scope.v1",
            returncode=result, error=error, live_qualified=False))
        handoff_numeric_evidence(directory, profile / "numeric_edge_scope.json",
                                 publisher_uid=reg["publisher_uid"], publisher_gid=reg["publisher_gid"])


# ---------------------------------------------------------------------------- verification-only entry modes
def _cuda_initialized():
    torch = sys.modules.get("torch")
    return bool(torch is not None and torch.cuda.is_initialized())


def _main_module():
    return getattr(getattr(sys.modules["__main__"], "__spec__", None), "name", None)


def _emit(document):
    print("M3_V17_STARTUP_VERIFICATION=" + json.dumps(document, sort_keys=True, default=str), flush=True)


def verify_child_startup(argv):
    """Run the UE child's startup validation against a prospective root, then stop (no production scope).

    The profiler configuration is produced by the retained ``runner_v1.profiler_config`` (the function whose output
    the live runner writes as PROFILER_CONFIG.json), from the sealed registration."""
    if len(argv) != 2 or argv[0] != "--output":
        raise SystemExit("usage: <child trampoline> --m3-verify-startup --output <prospective root>")
    output = Path(argv[1]).resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    for name in CHILD_IMPORTS:
        importlib.import_module(name)
    from rl_agent.splitfusion_single_gpu_profiler_v1.runner_v1 import profiler_config
    config = profiler_config(json.loads((output / "registration.json").read_bytes()), "m3_v17_startup_verification")
    checked = validate_child_startup(output, config, root)
    _emit(dict(role="ue_child", status="PASS", module=_main_module(), pid=os.getpid(),
               executable=sys.executable, root=str(root), output=str(output), resolved=checked["resolved"],
               contract_id=C.ID, record_schema=checked["record"]["schema"], seam=checked["seam"],
               retention_plan_id=checked["observer"].plan["plan_id"], cuda_initialized=_cuda_initialized(),
               production_scopes_entered=False))
    return 0


def verify_edge_startup(argv):
    """Run the edge's startup validation against the mounted registration, then stop (no production scope)."""
    path = Path(argv[1] if len(argv) == 2 and argv[0] == "--registration" else C.EDGE_REGISTRATION_PATH)
    root = Path(__file__).resolve().parents[2]
    for name in EDGE_IMPORTS:
        importlib.import_module(name)
    partition_code = Path("/cpu_partition_code")
    edge_gate_imported = False
    if partition_code.is_dir():
        sys.path.insert(0, str(partition_code))
        importlib.import_module("edge_gate")          # imported only; install() patches threads at production start
        edge_gate_imported = True
    reg = validate_edge_startup(path, root)
    _emit(dict(role="edge", status="PASS", module=_main_module(), pid=os.getpid(),
               uid=os.getuid(), executable=sys.executable, python=sys.version.split()[0], root=str(root),
               registration=str(path), registration_id=reg["id"],
               resolved=dict(validate_edge_startup=C.module_identity(validate_edge_startup),
                             registration=C.module_identity(registration), contract=C.__name__),
               edge_gate_imported=edge_gate_imported, cuda_initialized=_cuda_initialized(),
               production_scopes_entered=False))
    return 0
