"""Prospective identity of the replacement fixed-action screen on the repaired (v2) early-GT pipeline.

Pipeline = ``splitfusion_exact_gt_live_v2`` (anchor provenance + refusal cleanup + archive-completion fix),
whose exact sources are bound here together with this package. The file-level delta from the launched v1
pipeline is recorded in every registration. Same seven selected actions, limits, stop rules as v1.
"""
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_exact_gt_live_v2 import contract as E

PACKAGE = "rl_agent/splitfusion_fixed_action_screen_v2"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_fixed_action_v2"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_fixed_action_v2"
RUNNER = "rl_agent.splitfusion_fixed_action_screen_v2.runner"
PARTITION = "rl_agent.splitfusion_fixed_action_screen_v2.partition"
ID = "RUN5_FIXED_ACTION_SCREEN_EARLY_GT_V2"
BUDGET = 100
MAX_ARMS = 7
SOCKET_NAME, CAPABILITY_NAME, LIMITS = E.SOCKET_NAME, E.CAPABILITY_NAME, dict(E.LIMITS)
EXPECTED_CHANGED_FROM_V1 = frozenset((
    "rl_agent/splitfusion_exact_gt_live_v2/contract.py", "rl_agent/splitfusion_exact_gt_live_v2/service.py",
    "rl_agent/splitfusion_exact_gt_live_v2/ue.py", "rl_agent/splitfusion_exact_gt_live_v2/rehearsal.py",
    "rl_agent/splitfusion_exact_gt_live_v2/prereq.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/deferred.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/prep_core.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/scorer_core.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/worker.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/service.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/verify_composition.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/parity_common.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/test_anchor.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/verify_anchor.py"))
sha, canonical = E.sha, E.canonical


def token(action_id):
    return f"RUN5_FIXED_ACTION_SCREEN_V2_A{int(action_id):02d}_ONCE_EXECUTE_V1"


def metrics():
    return dict(E.metrics(), action_selection="REGISTERED_FIXED_ACTION_FORCED_FROZEN_ACTOR_IN_SHADOW_V1",
                executed_action_drives_bookkeeping=True, proposal_logged_and_discarded=True,
                fallback_semantics="RETAINED_UNCHANGED_MODE11_Q9800_A71_WHEN_STATE_UNAVAILABLE",
                policy_decision_agreement_gate_claimed=False)


def own_sources(root):
    root = Path(root).resolve(strict=True)
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))}


def exact_sources(root):
    result = dict(E.exact_sources(root))
    result.update(own_sources(root))
    return dict(sorted(result.items()))


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = dict(E.source_inventory(root))
    result.update(own_sources(root))
    for module in (CHILD, EDGE):
        path = root / (module.replace(".", "/") + ".py")
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing real fixed-action v2 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("fixed-action screen v2 source drift: " + relative)
    return dict(sorted(result.items()))


def pipeline_delta(root):
    """File-level delta of the v2 pipeline from the launched v1 packages; must be exactly the registered set."""
    delta = E.delta_from_v1(root)
    changed = {k for k, v in delta.items() if v in ("CHANGED", "ADDED")}
    if changed != EXPECTED_CHANGED_FROM_V1:
        raise ValueError("v2 pipeline delta differs from the registered repair scope: "
                         + json.dumps(sorted(changed ^ EXPECTED_CHANGED_FROM_V1)))
    return delta


def forced_document(action):
    keys = ("action_id", "profile_id", "family", "quantizer", "mode_id", "q_e4", "keep_count")
    return {k: action[k] for k in keys}


def lineage(root, actor_boundary, forced):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root), metrics=metrics(),
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", controller="RewardHoldControllerV2", deadline_ns=170_000_000,
        deadline_inclusive=True, fixed_fallback="mode11_q9800_a71", no_retraining=True,
        forced_action=forced_document(forced)))
