"""Static-mode v13 UE-child and edge entry points (versioned copy of ``splitfusion_m3_route_b_v12.entries``).

Differences from V12, all registered in the v13 contract:
  * NO controller substitution: ``runtime_policy.CONTROLLER_MODULE`` must equal the retained module, which commands
    the FAVORABLE_STABLE trace from step 0 (V12 installed its BAD_ALL controller here).
  * NO route-completion stop rule: ``route_completion.install_child_stop_rule`` (substitution R1) is not installed, so
    the retained ``phase6_live_child_v2.run`` budget rule decides the terminal.  The V3 ``M3CollectorLayer``
    (FinishResult repair) and this package's bounded drain stay installed through ``route_drain.install_collector_layer``.
  * NO paired-retention observer and NO ``finalize_after_child`` (retention registered disabled).
  * Mode arms: ``fixed_engine.install_fixed_engine`` (actor-free, identity-checked on every decision and hold); the
    agent arm runs the unchanged deployed actor.
Every registration check runs in ``validate_child_startup`` / ``validate_edge_startup`` before any production scope,
and the verification-only modes call the same functions and stop.
"""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys

from . import contract as C

REQUIRED_SOURCES = (
    *(f"{C.V1.PACKAGE}/{name}" for name in ("actor.py", "runtime_policy.py", "child_entry.py", "entries.py",
                                             "contract.py")),
    *(f"{C.PACKAGE}/{name}" for name in ("contract.py", "entries.py", "gc_seams.py", "host.py", "service_entry.py",
                                         "worker_entry.py", "accounting_timing.py", "route_drain.py",
                                         "radio_telemetry.py", "postflight_p3.py", "fixed_engine.py",
                                         "FIXED_IDENTITIES.json")),
    *C.PREP_LIFETIME_QUALIFIED, *C.AVO_TILT_QUALIFIED, *C.ACCOUNTING_QUALIFIED, *C.ACCOUNTING_V1_QUALIFIED,
    *(f"{C.V8.PACKAGE}/{name}" for name in ("contract.py", "worker_entry.py", "service_entry.py")),
    *(f"{C.V7.PACKAGE}/{name}" for name in ("contract.py", "worker_entry.py")),
    *(f"rl_agent/splitfusion_m3_route_b_v4/{name}" for name in ("contract.py", "abort_evidence.py", "worker_entry.py")),
    *(f"rl_agent/splitfusion_m3_route_b_v3/{name}" for name in (
        "contract.py", "entries.py", "worker_entry.py", "gc_policy_live.py", "capacity.py", "pose_tolerance.py",
        "route_completion.py", "map_journal.py", "edge_guard.py", "map_class.py")),
    *C.GC_SOURCES, *C.PRODUCER_SOURCES)

CHILD_IMPORTS = (
    "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1", "rl_agent.splitfusion_run5_full_live_v1.child_entry",
    "rl_agent.splitfusion_live_parity_repair_v1.gt_bridge", "rl_agent.splitfusion_live_parity_repair_v1.runtime",
    "rl_agent.splitfusion_live_parity_runtime_v1.pose", "rl_agent.splitfusion_live_parity_runtime_v8.startup",
    "rl_agent.splitfusion_numeric_gt_runtime_v1.live_entries", "rl_agent.splitfusion_ue_input_prep_v1.candidate",
    "rl_agent.splitfusion_ue_input_prep_v2.candidate", "rl_agent.splitfusion_exact_gt_live_v2.ue",
    "rl_agent.splitfusion_m3_route_b_v3.route_completion", "rl_agent.splitfusion_m3_route_b_v3.entries",
    "rl_agent.splitfusion_m3_route_b_v4.abort_evidence", "rl_agent.splitfusion_static_modes_v13.route_drain",
    "rl_agent.splitfusion_static_modes_v13.fixed_engine")
EDGE_IMPORTS = (
    "rl_agent.splitfusion_numeric_gt_runtime_v1.live_entries", "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1",
    "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_ordering",
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_prewarm_v2", "rl_agent.splitfusion_exact_gt_live_v2.edge",
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2",
    "rl_agent.splitfusion_m3_route_b_v3.edge_guard")
MARKER = "STATIC_V13_STARTUP_VERIFICATION="


def registration(path, root):
    """The child/edge registration check (reads the v13 contract)."""
    value = json.loads(Path(path).read_bytes())
    if (value.get("id") != C.ID or value.get("transmitted_budget") not in C.BUDGETS
            or value.get("g1_boundary_budget") != value.get("transmitted_budget")
            or value.get("exact_gt_limits") != C.limits(value.get("transmitted_budget"))
            or value.get("lifecycle") != C.LIFECYCLE
            or value.get("child_module") != C.CHILD or value.get("edge_module") != C.EDGE
            or value.get("live_metric_contract") != C.metrics()
            or value.get("registration_sha256") != C.canonical({k: v for k, v in value.items() if k != "registration_sha256"})):
        raise ValueError("static-mode v13 entry requires its exact sealed registration")
    if (value.get("launcher_revision") != C.LAUNCHER_REVISION or value.get("runtime_revision") != C.RUNTIME_REVISION
            or value.get("launcher_binding") != C.launcher_binding()):
        raise ValueError("registration launcher binding differs from the v13 contract")
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
        raise ValueError("a v13 registration must not carry a forced action")
    if value.get("paired_retention") != C.retention_registration():
        raise ValueError("registration retention section differs from the contract (disabled)")
    arm = value.get("static_arm")
    if arm not in C.ARM_KEYS:
        raise ValueError("registration names no registered static-mode arm")
    semantics = C.SELECTION_SEMANTICS_AGENT if arm == C.AGENT_ARM else C.SELECTION_SEMANTICS_FIXED
    if (value.get("external_fixed_action") != C.fixed_action(arm, root)
            or value.get("selection_semantics") != semantics
            or value.get("zero_completion_semantics") != C.ZERO_COMPLETION):
        raise ValueError("registration arm identity / selection semantics differ from the v13 contract")
    if value.get("channel") != C.channel_registration(root) or value.get("network_profile") != C.NETWORK_PROFILE:
        raise ValueError("registration channel differs from the registered FAVORABLE_STABLE trace")
    return value


@contextmanager
def require_retained_controller(attempt):
    """No substitution: the retained ``runtime_policy`` launches the retained controller (FAVORABLE_STABLE)."""
    from rl_agent.splitfusion_run5_full_live_v1 import runtime_policy as RP
    from rl_agent.splitfusion_m3_route_b_v3.entries import _write
    if RP.CONTROLLER_MODULE != C.RETAINED_CONTROLLER_MODULE:
        raise ValueError("retained controller module drift: " + repr(RP.CONTROLLER_MODULE))
    try:
        yield
    finally:
        directory = Path(attempt) / "m3_route"
        directory.mkdir(exist_ok=True)
        _write(directory / "CONTROLLER_MODULE_SEAM.json", dict(
            schema="splitfusion.static_modes_v13.controller_seam.v1", substituted=None,
            retained=C.RETAINED_CONTROLLER_MODULE, unchanged=RP.CONTROLLER_MODULE == C.RETAINED_CONTROLLER_MODULE,
            schedule=C.SCHEDULE["schedule_id"], trace_id=C.SCHEDULE["trace_id"],
            trace_sha256=C.FAVORABLE_TRACE_SHA256))


@contextmanager
def _no_scope():
    yield None


def validate_child_startup(output, config, root):
    """Every registration/verifier check of the UE child, in the live order, before any production scope."""
    from .gc_seams import seam_parameters
    from .host import verify_prospective
    if (config is None or config.get("qualification_id") != C.ID
            or config.get("sha256") != C.canonical({k: v for k, v in config.items() if k != "sha256"})):
        raise ValueError("sealed enabled static-mode v13 profiler configuration required")
    record = verify_prospective(output)
    reg = registration(Path(output) / "registration.json", root)
    seam = seam_parameters(reg)
    arm = reg["static_arm"]
    resolved = None if arm == C.AGENT_ARM else C.resolved_fixed_action(arm)
    if resolved is not None and resolved != reg["external_fixed_action"]:
        raise ValueError("live resolver differs from the registered fixed identity for " + arm)
    return dict(record=record, registration=reg, seam=seam, arm=arm, resolved_fixed_action=resolved,
                resolved=dict(verify_prospective=C.module_identity(verify_prospective),
                              registration=C.module_identity(registration),
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
    from rl_agent.splitfusion_m3_route_b_v3.entries import install_map_trampoline, install_ue_capacity
    from . import route_drain as RD
    from .fixed_engine import install_fixed_engine
    from .gc_seams import install_ue_service_seam
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = Path(BASE.attempt_dir_from_argv(args)).resolve(strict=True)
    config = BASE.load_config(attempt)
    output = attempt.parent.parent.parent
    root = Path(__file__).resolve().parents[2]
    checked = validate_child_startup(output, config, root)
    reg, arm = checked["registration"], checked["arm"]
    fixed = (_no_scope() if arm == C.AGENT_ARM
             else install_fixed_engine(arm=arm, fixed_action=reg["external_fixed_action"], attempt=attempt))
    bind_legacy_namespace()
    original = CHILD.EDGE_MODULE
    CHILD.EDGE_MODULE = C.EDGE
    try:
        with fixed, require_retained_controller(attempt), install_map_trampoline(attempt), \
                install_ue_service_seam(registration=reg, output=output), \
                preserve_numeric_evidence(), preserve_startup(attempt), install_ue_perception(), \
                install_bgr2rgb_candidate(), install_fused_candidate(), install_ue_capacity(attempt), \
                POSE.install_ue_pose(attempt_dir=attempt, max_frames=int(reg["transmitted_budget"])), \
                RD.install_collector_layer(), \
                install_early_gt_ue(attempt_dir=attempt, registration=reg):
            return CHILD.main(args)
    finally:
        CHILD.EDGE_MODULE = original


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
    print(MARKER + json.dumps(document, sort_keys=True, default=str), flush=True)


def verify_child_startup(argv):
    if len(argv) != 2 or argv[0] != "--output":
        raise SystemExit("usage: <child trampoline> --m3-verify-startup --output <prospective root>")
    output = Path(argv[1]).resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    for name in CHILD_IMPORTS:
        importlib.import_module(name)
    from rl_agent.splitfusion_single_gpu_profiler_v1.runner_v1 import profiler_config
    config = profiler_config(json.loads((output / "registration.json").read_bytes()), "static_v13_startup_verification")
    checked = validate_child_startup(output, config, root)
    _emit(dict(role="ue_child", status="PASS", module=_main_module(), pid=os.getpid(),
               executable=sys.executable, root=str(root), output=str(output), resolved=checked["resolved"],
               contract_id=C.ID, record_schema=checked["record"]["schema"], seam=checked["seam"],
               arm=checked["arm"], resolved_fixed_action=checked["resolved_fixed_action"],
               paired_retention_enabled=False, cuda_initialized=_cuda_initialized(),
               production_scopes_entered=False))
    return 0


def verify_edge_startup(argv):
    path = Path(argv[1] if len(argv) == 2 and argv[0] == "--registration" else C.EDGE_REGISTRATION_PATH)
    root = Path(__file__).resolve().parents[2]
    for name in EDGE_IMPORTS:
        importlib.import_module(name)
    partition_code = Path("/cpu_partition_code")
    edge_gate_imported = False
    if partition_code.is_dir():
        sys.path.insert(0, str(partition_code))
        importlib.import_module("edge_gate")
        edge_gate_imported = True
    reg = validate_edge_startup(path, root)
    _emit(dict(role="edge", status="PASS", module=_main_module(), pid=os.getpid(),
               uid=os.getuid(), executable=sys.executable, python=sys.version.split()[0], root=str(root),
               registration=str(path), registration_id=reg["id"], arm=reg["static_arm"],
               resolved=dict(validate_edge_startup=C.module_identity(validate_edge_startup),
                             registration=C.module_identity(registration), contract=C.__name__),
               edge_gate_imported=edge_gate_imported, cuda_initialized=_cuda_initialized(),
               production_scopes_entered=False))
    return 0
