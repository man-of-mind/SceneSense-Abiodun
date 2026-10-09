"""Prospective identity of the bounded fixed-action screen on the integrated early-GT pipeline.

The pipeline is the launched early-GT one, unchanged: its exact sources (``E.exact_sources``) are part of
this binding and must still equal what that experiment registered. This package only adds the forced
action seam, thin entry/host/runner/partition wrappers bound to this identity, and analysis code.
"""
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_exact_gt_live_v1 import contract as E

PACKAGE = "rl_agent/splitfusion_fixed_action_screen_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_fixed_action"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_fixed_action"
RUNNER = "rl_agent.splitfusion_fixed_action_screen_v1.runner"
PARTITION = "rl_agent.splitfusion_fixed_action_screen_v1.partition"
ID = "RUN5_FIXED_ACTION_SCREEN_EARLY_GT_V1"
BUDGET = 100
MAX_ARMS = 7
SOCKET_NAME, CAPABILITY_NAME, LIMITS = E.SOCKET_NAME, E.CAPABILITY_NAME, dict(E.LIMITS)
EARLY_GT_LAUNCHED_REGISTRATION = ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                                  "20261005_exact_gt_early_service_live100_v1/registration.json")
sha, canonical = E.sha, E.canonical


def token(action_id):
    return f"RUN5_FIXED_ACTION_SCREEN_A{int(action_id):02d}_ONCE_EXECUTE_V1"


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
            raise ValueError("missing real fixed-action trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("fixed-action screen source drift: " + relative)
    return dict(sorted(result.items()))


def early_gt_pipeline_unchanged(root):
    """The integrated pipeline sources equal those the launched early-GT experiment registered."""
    reg = json.loads((Path(root) / EARLY_GT_LAUNCHED_REGISTRATION).read_bytes())
    return reg["exact_gt_sources"] == E.exact_sources(root)


def forced_document(action):
    keys = ("action_id", "profile_id", "family", "quantizer", "mode_id", "q_e4", "keep_count")
    return {k: action[k] for k in keys}


def lineage(root, actor_boundary, forced):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root), metrics=metrics(),
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", controller="RewardHoldControllerV2", deadline_ns=170_000_000,
        deadline_inclusive=True, fixed_fallback="mode11_q9800_a71", no_retraining=True,
        forced_action=forced_document(forced)))
