"""Prospective M3.1 launcher revision over the sealed v4 scientific runtime.

Only the host-side runner/verifier/partition binding changes.  The registered
actor, state/action spaces, route, radio schedule, GT semantics, pose repair,
abort exporter, child, edge, service, and workers remain the v4 runtime.
"""
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v4.contract import *  # noqa: F401,F403

PACKAGE = "rl_agent/splitfusion_m3_route_b_v5"
LAUNCHER_REVISION = "M3_1_ROUTE_B_LAUNCHER_BINDING_V5"
RUNTIME_REVISION = V4.ID

# The live runtime is deliberately unchanged.  Only these host-side entry
# points belong to the repair revision.
ID = V4.ID
CHILD = V4.CHILD
EDGE = V4.EDGE
SERVICE_ENTRY = V4.SERVICE_ENTRY
WORKER_ENTRY = V4.WORKER_ENTRY
RUNNER = "rl_agent.splitfusion_m3_route_b_v5.runner"
PARTITION = "rl_agent.splitfusion_m3_route_b_v5.partition"


def own_sources(root):
    root = Path(root).resolve(strict=True)
    result = dict(V4.own_sources(root))
    result.update({str(path.relative_to(root)): sha(path)
                   for path in sorted((root / PACKAGE).glob("*.py"))})
    return result


def exact_sources(root):
    result = dict(V4.exact_sources(root))
    result.update(own_sources(root))
    return dict(sorted(result.items()))


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = dict(V4.source_inventory(root))
    result.update(own_sources(root))
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M3 v5 launcher source drift: " + relative)
    return dict(sorted(result.items()))


def lineage(root, actor_boundary, support_sha256, budget, arm, producer="v2",
            retention_plan_sha256=None):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    if not isinstance(retention_plan_sha256, str) or len(retention_plan_sha256) != 64:
        raise ValueError("retention launcher lineage requires the pre-launch retention plan digest")
    return canonical(dict(
        launcher_revision=LAUNCHER_REVISION, runtime_revision=RUNTIME_REVISION,
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
