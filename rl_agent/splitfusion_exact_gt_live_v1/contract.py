"""Prospective identity for one bounded early-GT live engineering check (100 frames)."""
import hashlib
import json
from pathlib import Path

PACKAGE = "rl_agent/splitfusion_exact_gt_live_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_exact_gt"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_exact_gt"
RUNNER = "rl_agent.splitfusion_exact_gt_live_v1.runner"
PARTITION = "rl_agent.splitfusion_exact_gt_live_v1.partition"
ID = "RUN5_EXACT_GT_EARLY_SERVICE_LIVE_100_ENGINEERING_V1"
TOKEN = "RUN5_EXACT_GT_EARLY_SERVICE_LIVE_100_ONCE_EXECUTE_V1"
BUDGET = 100
SOCKET_NAME = "exact_gt.sock"
CAPABILITY_NAME = "exact_gt.cap"
NEW_PACKAGES = ("rl_agent/splitfusion_exact_gt_live_v1", "rl_agent/splitfusion_exact_gt_service_proto_v1",
                "rl_agent/splitfusion_ue_input_prep_v1")
LIMITS = dict(max_frames=BUDGET, prep_jobs=4, scorer_jobs=4, freeze_queue=1, max_waiting=2, ue_freeze_queue=2,
              horizon_ms=2000, drain_s=120, prep_cpu=4, score_cpu=5, service_cpu=6)


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
        deadline_exceedance_is_performance_not_architecture_failure=True)


def exact_sources(root):
    """Every source the integrated path executes beyond the retained Run-5 tree."""
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
    for module in (CHILD, EDGE):
        path = root / (module.replace(".", "/") + ".py")
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing real early-GT trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != digest:
            raise ValueError("early-GT live source drift: " + relative)
    return dict(sorted(result.items()))


def lineage(root, actor_boundary):
    return canonical(dict(actor_boundary_sha256=actor_boundary, sources=source_inventory(root), metrics=metrics(),
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", controller="RewardHoldControllerV2", deadline_ns=170_000_000,
        deadline_inclusive=True, fixed_fallback="mode11_q9800_a71", no_retraining=True))
