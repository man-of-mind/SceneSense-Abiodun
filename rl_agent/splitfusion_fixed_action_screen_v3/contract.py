"""Prospective identity of the three-arm mode-11 frontier measurement (D10) on the unchanged v2 early-GT pipeline.

Differences from ``splitfusion_fixed_action_screen_v2`` (whose sources and evidence stay untouched):

* a forced action is any executable (mode_id, q_e4) whose identity is EXACTLY the dynamic execution
  contract's own ``resolve_q_e4`` resolution -- catalogue anchors keep their catalogue action/profile id,
  an off-anchor q (here mode 11, q_e4 8000) carries ``action_id=None, profile_id=None,
  measurement_status=UNMEASURED_OFF_ANCHOR``. No catalogue id is invented;
* arms are keyed by (mode_id, q_e4), not by catalogue id; three arms, registered order;
* the pipeline, actor, controller, fallback, deadline, limits and lifecycle are the v2 ones, unchanged.
"""
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_exact_gt_live_v2 import contract as E

PACKAGE = "rl_agent/splitfusion_fixed_action_screen_v3"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_fixed_action_v3"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_fixed_action_v3"
RUNNER = "rl_agent.splitfusion_fixed_action_screen_v3.runner"
PARTITION = "rl_agent.splitfusion_fixed_action_screen_v3.partition"
ID = "RUN5_FIXED_ACTION_M11_FRONTIER_EARLY_GT_V3"
BUDGET = 100
MAX_ARMS = 3
ARMS = (("arm0_m11_q3000_a67", 11, 3000), ("arm1_m11_q7000_a69", 11, 7000), ("arm2_m11_q8000_offanchor", 11, 8000))
CATALOGUE_Q = (0, 3000, 5000, 7000, 9000, 9800)        # six catalogue anchors per mode; action_id = 6*mode + index
IDENTITY_SOURCE = "DYNAMIC_EXECUTION_CONTRACT_RESOLVE_Q_E4"
FORCED_KEYS = ("action_id", "profile_id", "family", "quantizer", "mode_id", "q_e4", "keep_count", "drop_count",
               "measurement_status", "execution_bundle_sha256", "identity_source")
SOCKET_NAME, CAPABILITY_NAME, LIMITS = E.SOCKET_NAME, E.CAPABILITY_NAME, dict(E.LIMITS)
EXPECTED_CHANGED_FROM_V1 = frozenset((                  # the registered v2 repair scope (unchanged pipeline)
    "rl_agent/splitfusion_exact_gt_live_v2/contract.py", "rl_agent/splitfusion_exact_gt_live_v2/service.py",
    "rl_agent/splitfusion_exact_gt_live_v2/ue.py", "rl_agent/splitfusion_exact_gt_live_v2/rehearsal.py",
    "rl_agent/splitfusion_exact_gt_live_v2/prereq.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/deferred.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/prep_core.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/scorer_core.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/worker.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/service.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/verify_composition.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/parity_common.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/test_anchor.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/verify_anchor.py"))
sha, canonical = E.sha, E.canonical


def arm_key(mode_id, q_e4):
    for key, mode, q in ARMS:
        if (mode, q) == (int(mode_id), int(q_e4)):
            return key
    raise ValueError(f"(mode {mode_id}, q_e4 {q_e4}) is not a registered arm")


def token(forced):
    return f"RUN5_FIXED_ACTION_M11_FRONTIER_V3_M{int(forced['mode_id']):02d}_Q{int(forced['q_e4']):04d}_ONCE_EXECUTE_V1"


def metrics():
    return dict(E.metrics(), action_selection="REGISTERED_FIXED_ACTION_FORCED_FROZEN_ACTOR_IN_SHADOW_V1",
                executed_action_drives_bookkeeping=True, proposal_logged_and_discarded=True,
                fallback_semantics="RETAINED_UNCHANGED_MODE11_Q9800_A71_WHEN_STATE_UNAVAILABLE",
                policy_decision_agreement_gate_claimed=False,
                forced_identity_rule="EXACT_DYNAMIC_CONTRACT_RESOLUTION_INCLUDING_OFF_ANCHOR_V1")


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
            raise ValueError("missing real fixed-action v3 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("fixed-action v3 source drift: " + relative)
    return dict(sorted(result.items()))


def pipeline_delta(root):
    delta = E.delta_from_v1(root)
    changed = {k for k, v in delta.items() if v in ("CHANGED", "ADDED")}
    if changed != EXPECTED_CHANGED_FROM_V1:
        raise ValueError("v2 pipeline delta differs from the registered repair scope: "
                         + json.dumps(sorted(changed ^ EXPECTED_CHANGED_FROM_V1)))
    return delta


def resolved_document(profile):
    """The forced-action document from an ``ExecutableDispatchProfile`` (the contract's own resolution)."""
    return dict(action_id=profile.action_id, profile_id=profile.profile_id, family=profile.family,
                quantizer=profile.quantizer, mode_id=int(profile.mode_id), q_e4=int(profile.q_e4),
                keep_count=int(profile.keep_count), drop_count=int(profile.drop_count),
                measurement_status=str(profile.measurement_status),
                execution_bundle_sha256=profile.execution_bundle_sha256, identity_source=IDENTITY_SOURCE)


def resolve(mode_id, q_e4, contract=None):
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
    contract = dec.load_dynamic_execution_contract() if contract is None else contract
    return resolved_document(contract.resolve_q_e4(int(mode_id), int(q_e4)))


def check_forced_shape(forced):
    """Contract-free consistency check usable in the child and the edge container.

    The authoritative check (equality with ``resolve``) runs at preparation; at runtime the retained engine
    resolves the profile itself and the logged executed identity is compared after the arm.
    """
    if set(forced) != set(FORCED_KEYS) or forced["identity_source"] != IDENTITY_SOURCE:
        raise ValueError("forced action is not a registered dynamic-contract resolution document")
    mode, q = forced["mode_id"], forced["q_e4"]
    if type(mode) is not int or type(q) is not int or not 0 <= mode < 12 or not 0 <= q <= 9800:
        raise ValueError("forced (mode_id, q_e4) outside the executable range")
    if (mode, q) not in {(m, v) for _, m, v in ARMS}:
        raise ValueError("forced action is not one of the registered three arms")
    if q in CATALOGUE_Q:
        if forced["action_id"] != 6 * mode + CATALOGUE_Q.index(q) or forced["measurement_status"] != "MEASURED_ANCHOR" \
                or not forced["profile_id"]:
            raise ValueError("catalogue-anchor q without its exact catalogue identity")
    elif forced["action_id"] is not None or forced["profile_id"] is not None \
            or forced["measurement_status"] != "UNMEASURED_OFF_ANCHOR":
        raise ValueError("off-anchor q must carry action_id=None, profile_id=None, UNMEASURED_OFF_ANCHOR")
    bundle = forced["execution_bundle_sha256"]
    if (type(forced["keep_count"]) is not int or type(forced["drop_count"]) is not int
            or forced["keep_count"] + forced["drop_count"] != 21504
            or type(bundle) is not str or len(bundle) != 64 or any(c not in "0123456789abcdef" for c in bundle)):
        raise ValueError("forced keep/drop counts or execution bundle digest invalid")
    return forced


def forced_document(action):
    return {k: action[k] for k in FORCED_KEYS}


def lineage(root, actor_boundary, forced):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root), metrics=metrics(),
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", controller="RewardHoldControllerV2", deadline_ns=170_000_000,
        deadline_inclusive=True, fixed_fallback="mode11_q9800_a71", no_retraining=True,
        forced_action=forced_document(forced)))
