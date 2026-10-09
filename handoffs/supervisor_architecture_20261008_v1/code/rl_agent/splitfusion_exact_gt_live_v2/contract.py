"""Source binding and metric contract of the repaired (v2) early-GT pipeline.

v2 = the launched early-GT pipeline with exactly two narrow changes:
  * catalogue-anchor captures bind through an explicit, versioned anchor-provenance mapping
    (proto_v2 deferred/prep/scorer/worker) instead of being refused;
  * BIND refusals (both arrival orders) and expected scorer capacity refusals produce one explicit
    no-score outcome and release scorer capacity (live_v2 service).
No live entry/host/runner is defined here; the fixed-action screen v2 composes this pipeline.
"""
import hashlib
import json
from pathlib import Path

PACKAGE = "rl_agent/splitfusion_exact_gt_live_v2"
BUDGET = 100
SOCKET_NAME = "exact_gt.sock"
CAPABILITY_NAME = "exact_gt.cap"
NEW_PACKAGES = ("rl_agent/splitfusion_exact_gt_live_v2", "rl_agent/splitfusion_exact_gt_service_proto_v2",
                "rl_agent/splitfusion_ue_input_prep_v1")
LIMITS = dict(max_frames=BUDGET, prep_jobs=4, scorer_jobs=4, freeze_queue=1, max_waiting=2, ue_freeze_queue=2,
              horizon_ms=2000, drain_s=120, prep_cpu=4, score_cpu=5, service_cpu=6)
V1_PACKAGES = ("rl_agent/splitfusion_exact_gt_live_v1", "rl_agent/splitfusion_exact_gt_service_proto_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def metrics():
    from rl_agent.splitfusion_live_parity_runtime_v1.contract import metrics as original
    return dict(original(), gt_readiness="NUMERIC_READY_TARGET_V2_EARLY_SERVICE",
        source_ownership="IMMUTABLE_OWNED_TICK_BEFORE_ACTION_UNCHANGED",
        source_handoff="SEALED_MEMFD_DESCRIPTOR_ON_PERSISTENT_UE_THREAD_AFTER_ACTION_OPEN",
        producer_dispatch="REWARD_DECISIONS_ONLY_AT_ACTION_OPEN_DEFERRED_BINDING_COMMIT_AFTER_SEND",
        prediction_submission="IMMEDIATELY_AFTER_EDGE_COMPUTE_BEFORE_MAP",
        ue_input_function="CORRECTED_PERCEPTION_WITH_BIT_EXACT_BGR2RGB_CV2_V1",
        archive="DEFERRED_DURABLE_AFTER_REAL_UE_TERMINAL",
        archive_durability_inside_reward_deadline=False,
        actual_scoring_and_radio_feedback_inside_reward_deadline=True,
        deadline_exceedance_is_performance_not_architecture_failure=True,
        anchor_binding="ACTION_NEUTRAL_SOURCE_IDENTITY_PLUS_EXPLICIT_ANCHOR_PROVENANCE_V1",
        refusal_handling="EXPLICIT_NO_SCORE_ONCE_RELEASE_SCORER_CAPACITY_V2")


def exact_sources(root):
    from rl_agent.splitfusion_numeric_gt_runtime_v1.live_contract import worker_sources as retained_workers
    root = Path(root).resolve(strict=True)
    result = dict(retained_workers())
    for package in NEW_PACKAGES + ("rl_agent/splitfusion_live_gt_handoff_v1",
                                   "rl_agent/splitfusion_live_gt_optimization_v4"):
        result.update({str(p.relative_to(root)): sha(p) for p in sorted((root / package).glob("*.py"))})
    return dict(sorted(result.items()))


def source_inventory(root):
    from rl_agent.splitfusion_live_parity_runtime_v1.contract import source_inventory as retained
    root = Path(root).resolve(strict=True)
    result = retained(root)
    result.update(exact_sources(root))
    for package in ("rl_agent/splitfusion_live_parity_repair_v2",):
        result.update({str(p.relative_to(root)): sha(p) for p in (root / package).glob("*.py")})
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != digest:
            raise ValueError("early-GT v2 source drift: " + relative)
    return dict(sorted(result.items()))


def delta_from_v1(root):
    """Exact file-level change list from the launched v1 packages to v2 (renamed module paths)."""
    root = Path(root).resolve(strict=True)
    out = {}
    for old, new in zip(V1_PACKAGES, NEW_PACKAGES[:2]):
        old_files = {p.name: p for p in (root / old).glob("*.py")}
        new_files = {p.name: p for p in (root / new).glob("*.py")}
        for name in sorted(set(old_files) | set(new_files)):
            a = old_files.get(name)
            b = new_files.get(name)
            if a is None or b is None:
                out[f"{new}/{name}"] = "ADDED" if a is None else "NOT_CARRIED_TO_V2"
                continue
            renamed = a.read_text().replace("splitfusion_exact_gt_service_proto_v1", "splitfusion_exact_gt_service_proto_v2") \
                .replace("splitfusion_exact_gt_live_v1", "splitfusion_exact_gt_live_v2")
            out[f"{new}/{name}"] = "IDENTICAL_AFTER_PACKAGE_RENAME" if renamed == b.read_text() else "CHANGED"
    return out
