"""Prospective identity for one bounded, original Run-5 numeric-GT trial.

This is a new runtime contract, not a claim that the historical AVO READY
clocks or the offline timing replay were observed on a live route.
"""
import hashlib
import json
from pathlib import Path

PACKAGE = "rl_agent/splitfusion_numeric_gt_runtime_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_numeric_gt"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_numeric_gt"
RUNNER = "rl_agent.splitfusion_numeric_gt_runtime_v1.live_runner"
PARTITION = "rl_agent.splitfusion_numeric_gt_runtime_v1.live_partition"
ID = "RUN5_NUMERIC_GT_LIVE_100_ENGINEERING_V1"
TOKEN = "RUN5_NUMERIC_GT_LIVE_100_ONCE_EXECUTE_V1"
BUDGET = 100
SOCKET_NAME = "numeric_gt.sock"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def metrics():
    from rl_agent.splitfusion_live_parity_runtime_v1.contract import metrics as original
    from .live_clock import ASYNC_HANDOFF_ID
    return dict(original(), gt_readiness="NUMERIC_READY_TARGET_V2",
        archive="OBSERVED_RAW_LIVE_ARCHIVE_V2_NOT_HISTORICAL_READY",
        source_stage=ASYNC_HANDOFF_ID,
        source_ownership="IMMUTABLE_OWNED_TICK_BEFORE_ACTION",
        producer_dispatch="REWARD_FRAMES_ONLY_AFTER_ACTUAL_LAST_SEND",
        producer_lifetime="PERSISTENT_CREATOR_THREAD_UNTIL_CLOSE_OR_ABORT",
        producer_startup="COMPLETE_STARTUP_POPULATION_REGISTRY_BEFORE_FIRST_FRAME",
        archive_durability_inside_reward_deadline=False,
        actual_scoring_and_radio_feedback_inside_reward_deadline=True)


def worker_sources():
    from .replay import inventory
    from rl_agent.splitfusion_numeric_gt_producer_v1.originals import startup_binding
    return inventory(dict(startup=startup_binding()))


def source_inventory(root):
    from rl_agent.splitfusion_live_parity_runtime_v1.contract import source_inventory as retained
    root = Path(root).resolve(strict=True)
    result = retained(root)
    result.update(worker_sources())
    # These adapters are deliberately new, not edits to a sealed old runner.
    for package in ("rl_agent/splitfusion_live_parity_repair_v2",
                    "rl_agent/splitfusion_live_gt_handoff_v1",
                    "rl_agent/splitfusion_live_gt_optimization_v4"):
        result.update({str(p.relative_to(root)): sha(p)
                       for p in (root / package).glob("*.py")})
    for module in (CHILD, EDGE):
        path = root / (module.replace(".", "/") + ".py")
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing real numeric live trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root)
                or sha(path) != digest):
            raise ValueError("numeric live source drift: " + relative)
    return dict(sorted(result.items()))


def lineage(root, actor_boundary):
    return canonical(dict(actor_boundary_sha256=actor_boundary,
        sources=source_inventory(root), metrics=metrics(),
        state="ORIGINAL_RUN5_22D_QUALITY_AWARE", controller="RewardHoldControllerV2",
        deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True))
