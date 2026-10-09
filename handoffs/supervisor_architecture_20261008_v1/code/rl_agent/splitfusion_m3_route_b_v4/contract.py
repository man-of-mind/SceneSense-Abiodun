"""M3.1 v4 registration layered on the sealed, unmodified v3 launcher.

V4 changes only two integration seams: the formerly rejected GT-worker pose
range is exercised through both real workers, and already-observed Phase-6 UE
state is exported if shutdown loses the exact-GT service.  All scientific and
runtime policy constants are inherited unchanged from v3.
"""
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v3.contract import *  # noqa: F401,F403

PACKAGE = "rl_agent/splitfusion_m3_route_b_v4"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v4"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v4"
RUNNER = "rl_agent.splitfusion_m3_route_b_v4.runner"
PARTITION = "rl_agent.splitfusion_m3_route_b_v4.partition"
SERVICE_ENTRY = "rl_agent.splitfusion_m3_route_b_v4.service_entry"
WORKER_ENTRY = "rl_agent.splitfusion_m3_route_b_v4.worker_entry"
ID = "M3_1_ROUTE_B_GOOD_ALL_V4"
RETENTION_PLAN_ID = "M3_1_ROUTE_B_GOOD_ALL_V4_P3000_STRIDE15"
ABORT_EXPORT = dict(
    artifact="PHASE6_UE_EVIDENCE.json",
    status="PARTIAL_ABORTED",
    once_only=True,
    no_controller_poll=True,
    no_handoff_close_for_export=True,
    unresolved_preserved=True,
    original_error_precedence=True,
    synchronous_per_frame_persistence=False,
)


def token(arm, budget=FRAME_CEILING, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    return f"M3_1_ROUTE_B_GOOD_ALL_V4_{arm.upper()}_{producer.upper()}_C{int(budget):04d}_ONCE_EXECUTE_V1"


def own_sources(root):
    root = Path(root).resolve(strict=True)
    result = dict(V3.own_sources(root))
    result.update({str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))})
    return result


def exact_sources(root):
    result = dict(V3.E.exact_sources(root))
    result.update(own_sources(root))
    return dict(sorted(result.items()))


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = dict(V3.E.source_inventory(root))
    result.update(own_sources(root))
    shims = [root / (CHILD.replace(".", "/") + ".py"), root / (EDGE.replace(".", "/") + ".py"),
             root / MAP_SERVER_TRAMPOLINE]
    for path in shims:
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing real M3 v4 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M3 v4 source drift: " + relative)
    return dict(sorted(result.items()))


def evidence_registration():
    value = dict(V3.evidence_registration())
    value["abort_export"] = dict(ABORT_EXPORT)
    return value


def lineage(root, actor_boundary, support_sha256, budget, arm, producer="v2", retention_plan_sha256=None):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    if not isinstance(retention_plan_sha256, str) or len(retention_plan_sha256) != 64:
        raise ValueError("retention launcher lineage requires the pre-launch retention plan digest")
    return canonical(dict(
        retention_plan_sha256=retention_plan_sha256,
        retention_registration=retention_registration(retention_plan_sha256),
        channel=channel_registration(root), route=route_registration(root), capacity=capacity_registration(),
        evidence_additions=evidence_registration(), pose_repair=pose_registration(), map_class=map_class_registration(),
        actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=None, budget=int(budget),
        gt_gc_arm=arm, gc_policy=GC_POLICY, gc_safety=SAFETY, numeric_producer=producer,
        producer_substitutions=list(SUBSTITUTIONS) if producer == "v2" else []))


def retention_plan(budget):
    from rl_agent.splitfusion_paired_retention_v2 import plan as RP
    if int(budget) not in BUDGETS:
        raise ValueError("budget is not the registered M3 frame ceiling")
    return RP.build_plan(plan_id=RETENTION_PLAN_ID, planned_frames=RETENTION_PLANNED_FRAMES, **RETENTION_PLAN_ARGS)
