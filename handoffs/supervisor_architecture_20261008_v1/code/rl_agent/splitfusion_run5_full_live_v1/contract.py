"""Prospective identity of the actor-driven live evaluation of the frozen full Run-5 actor.

Pipeline: the unchanged exact-GT v2 early-GT composition (as used by the v2/v3 fixed-action screens), the retained
Run-5 host lifecycle, CPU partition and thread caps, FAVORABLE_STABLE, inclusive 170-ms deadline. Differences from
the v3 screen: NO forced-action layer (the actor drives every decision); the actor is the frozen full Run-5 actor
(``actor.FrozenFullRun5Actor``) loaded by versioned copies of the Run-5 policy seam and child entry; the frame
budget is a registered parameter (100-frame diagnostic, 1,000-frame validation) and sets the exact-GT
``max_frames`` limit accordingly. Retained gates are all binding (G8 actor replay with the full Run-5 loader included);
the only registered exception is the retained whole-cycle frame-count boundary (G1 on TRANSMITTED_BUDGET_REACHED or
the verified decision-cycle boundary).
"""
import json
from pathlib import Path

from rl_agent.splitfusion_exact_gt_live_v2 import contract as E

PACKAGE = "rl_agent/splitfusion_run5_full_live_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_full_live_v1"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_full_live_v1"
RUNNER = "rl_agent.splitfusion_run5_full_live_v1.runner"
PARTITION = "rl_agent.splitfusion_run5_full_live_v1.partition"
ID = "RUN5_FULL_LIVE_FROZEN_INFERENCE_EARLY_GT_V1"
BUDGETS = (100, 1000)
EXPORT_FILES = ("actor_export/ACTOR_BINDING.json", "actor_export/actor_state_dict.pt")
SUPPORT_SOURCES = ("rl_agent/splitfusion_run5_full_v1/__init__.py",
                   "rl_agent/splitfusion_run5_full_v1/support.py",
                   # archive capacity repair v1: registered original->repaired substitution used by the host
                   "rl_agent/splitfusion_gt_archive_capacity_v1/__init__.py",
                   "rl_agent/splitfusion_gt_archive_capacity_v1/contract.py",
                   # mask-lookup candidate v1: combined registered substitution used by the host
                   "rl_agent/splitfusion_gt_mask_lookup_v1/__init__.py",
                   "rl_agent/splitfusion_gt_mask_lookup_v1/contract.py",
                   # UE input single-buffer candidate v2 (bit-exact to the installed v1 BGR->RGB candidate)
                   "rl_agent/splitfusion_ue_input_prep_v2/__init__.py",
                   "rl_agent/splitfusion_ue_input_prep_v2/candidate.py")
SOCKET_NAME, CAPABILITY_NAME = E.SOCKET_NAME, E.CAPABILITY_NAME
# Sources bound by the input-prep v2 proof (must equal splitfusion_ue_input_prep_v2.prove.SOURCES).
INPUT_SOURCES = ("rl_agent/splitfusion_ue_input_prep_v2/__init__.py", "rl_agent/splitfusion_ue_input_prep_v2/candidate.py",
                 "rl_agent/splitfusion_ue_input_prep_v1/candidate.py", "rl_agent/splitfusion_live_parity_repair_v1/perception.py")
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


def limits(budget):
    if budget not in BUDGETS:
        raise ValueError("budget is not a registered frame budget")
    return dict(E.LIMITS, max_frames=int(budget))


def token(budget):
    return f"RUN5_FULL_LIVE_FROZEN_INFERENCE_V1_B{int(budget):04d}_ONCE_EXECUTE_V1"


def metrics():
    return dict(E.metrics(), ue_input_function="CORRECTED_PERCEPTION_BGR2RGB_CV2_V1_PLUS_SINGLE_BUFFER_V2_BIT_EXACT",
                action_selection="FROZEN_FULL_RUN5_ACTOR_DETERMINISTIC_ARGMAX_MODE_CONDITIONAL_MEAN_Q",
                executed_action_drives_bookkeeping=True, forced_action_layer=False, online_learning=False,
                fallback_semantics="RETAINED_UNCHANGED_MODE11_Q9800_A71_WHEN_STATE_UNAVAILABLE",
                state="ORIGINAL_RUN5_22D_QUALITY_AWARE_PREV_QPERC_CAUSAL_SNR",
                reward_post_processing="CURRENT_RUN4_REWARD_LIVE_UNCHANGED__AGREED_REWARD_OFFLINE_ONLY")


def own_sources(root):
    root = Path(root).resolve(strict=True)
    out = {str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))}
    out.update({f"{PACKAGE}/{name}": sha(root / PACKAGE / name) for name in EXPORT_FILES})
    out.update({name: sha(root / name) for name in SUPPORT_SOURCES})
    return out


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
            raise ValueError("missing real full Run-5 live trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("full Run-5 live source drift: " + relative)
    return dict(sorted(result.items()))


def pipeline_delta(root):
    delta = E.delta_from_v1(root)
    changed = {k for k, v in delta.items() if v in ("CHANGED", "ADDED")}
    if changed != EXPECTED_CHANGED_FROM_V1:
        raise ValueError("v2 pipeline delta differs from the registered repair scope: "
                         + json.dumps(sorted(changed ^ EXPECTED_CHANGED_FROM_V1)))
    return delta


def lineage(root, actor_boundary, support_sha256, budget):
    return canonical(dict(actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=None, budget=int(budget)))
