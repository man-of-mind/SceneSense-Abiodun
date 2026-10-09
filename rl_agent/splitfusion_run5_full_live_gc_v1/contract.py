"""Prospective identity of the paired live qualification of the GT prep-worker GC policy.

Everything is the retained full Run-5 live evaluation (``splitfusion_run5_full_live_v1``: frozen seed-43 actor, 12 modes,
original q support, 22 features incl. causal SNR, live reward, inclusive 170-ms deadline, early-GT v2 pipeline, CPU
partition, thread caps, OAI configuration, FAVORABLE_STABLE) with ONE registered difference between the two arms:

  arm "control"   : the GT service and both workers are started through this package's trampolines with
                    GC tracing only (stock CPython GC).
  arm "candidate" : identical trampolines and tracing; the prep worker ALONE installs the bounded GC policy
                    (``splitfusion_gt_worker_spike_fix_v1.gc_policy``) with the registered safety stops below.

Correction of earlier wording: the gen2 threshold (200) is NOT a memory cap. In CPython 3.10 it only bounds the number
of gen1 collections between automatic full collections, and the full collection is additionally gated by the
long-lived-pending > long_lived_total/4 heuristic. Memory is bounded here only by the explicit RSS/growth safety stops.
"""
import json
from pathlib import Path

from rl_agent.splitfusion_exact_gt_live_v2 import contract as E
from rl_agent.splitfusion_run5_full_live_v1 import contract as V1

PACKAGE = "rl_agent/splitfusion_run5_full_live_gc_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_full_live_gc_v1"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_full_live_gc_v1"
RUNNER = "rl_agent.splitfusion_run5_full_live_gc_v1.runner"
PARTITION = "rl_agent.splitfusion_run5_full_live_gc_v1.partition"
SERVICE_MODULE = "rl_agent.splitfusion_exact_gt_live_v2.service"
SERVICE_ENTRY = "rl_agent.splitfusion_run5_full_live_gc_v1.service_entry"
WORKER_MODULE = "rl_agent.splitfusion_exact_gt_service_proto_v2.worker"
WORKER_ENTRY = "rl_agent.splitfusion_run5_full_live_gc_v1.worker_entry"
ID = "RUN5_FULL_LIVE_GT_GC_QUALIFICATION_V1"
ARMS = ("control", "candidate")
BUDGETS = (1000,)
TRACE_DIR = "gt_gc_trace"
GC_SOURCES = ("rl_agent/splitfusion_gt_worker_spike_fix_v1/__init__.py",
              "rl_agent/splitfusion_gt_worker_spike_fix_v1/gc_policy.py")
INPUT_SOURCES = V1.INPUT_SOURCES
EXPECTED_CHANGED_FROM_V1 = V1.EXPECTED_CHANGED_FROM_V1
SOCKET_NAME, CAPABILITY_NAME = V1.SOCKET_NAME, V1.CAPABILITY_NAME
sha, canonical = E.sha, E.canonical
limits, metrics, pipeline_delta = V1.limits, V1.metrics, V1.pipeline_delta

# Measured references (valid_b1000, 493 decisions; offline bench 20261006): prep RSS 686-720 MiB, growth after
# warm-up 6.7 MiB; in-request gen2 25-31 ms offline / +34-87 ms live; candidate idle collections <= 0.375 ms.
GC_POLICY = dict(warmup_decisions=2, idle_min_gen1=1, gen2_threshold=200, idle_ops=sorted((
    "EVIDENCE_DURABLE", "UNSCORED_DURABLE", "CANCEL_UNBOUND", "CANCEL_BOUND", "VOID")))
# SAFETY STOPS (candidate only, in-process): on breach the policy uninstalls itself (stock GC restored), records
# SAFETY_STOP and the run continues unchanged; the arm is then a qualification REJECT. Not performance criteria.
SAFETY = dict(prep_rss_max_bytes=1536 * 2**20,            # ~2.1x the live maximum (720 MiB)
              prep_rss_growth_after_freeze_max_bytes=256 * 2**20,
              idle_collection_wall_max_ms=50.0,           # a pause this long defeats the purpose
              host_mem_available_min_bytes=4 * 2**30)    # checked by the campaign before each arm
# PERFORMANCE ACCEPTANCE (evaluated after both arms; registered now).
ACCEPTANCE = dict(
    control_reproduces_min_in_request_gen2=10,      # control FREEZE/BIND full collections after warm-up, else INCONCLUSIVE
    candidate_in_request_gen2_after_warmup=0,       # FREEZE/BIND, decisions > warmup
    idle_collection_p99_max_ms=5.0, idle_collection_max_ms=20.0,
    idle_arrival_collision_max_fraction=0.01,
    rss_growth_after_warmup_max_excess_over_control_bytes=32 * 2**20,
    rss_slope_second_half_max_bytes_per_decision=64 * 1024,
    spike_definition="FREEZE wall > arm median + 30 ms (same rule both arms)",
    both_arms_integration="splitfusion_run5_full_live_analysis_v1.integration.evaluate == PASS_ACTOR_DRIVEN_INTEGRATION")


def trace_limits(budget):
    """Tracing is bounded by the configured run length; overflow is counted, never silent."""
    budget = int(budget)
    return dict(max_ops=16 * budget, max_gen2_events=4 * budget, max_policy_events=4 * budget)


def token(arm, budget=1000):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return f"RUN5_FULL_LIVE_GT_GC_QUAL_V1_{arm.upper()}_B{int(budget):04d}_ONCE_EXECUTE_V1"


def own_sources(root):
    root = Path(root).resolve(strict=True)
    out = dict(V1.own_sources(root))
    out.update({str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))})
    out.update({name: sha(root / name) for name in GC_SOURCES})
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
            raise ValueError("missing real GC-qualification trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("GC-qualification source drift: " + relative)
    return dict(sorted(result.items()))


def lineage(root, actor_boundary, support_sha256, budget, arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return canonical(dict(actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=None, budget=int(budget),
        gt_gc_arm=arm, gc_policy=GC_POLICY, gc_safety=SAFETY))


def gc_registration(arm, budget):
    return dict(arm=arm, policy=GC_POLICY if arm == "candidate" else None, safety=SAFETY, acceptance=ACCEPTANCE,
                trace_limits=trace_limits(budget), trace_dir=TRACE_DIR, role="prep",
                trampolines=dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY),
                substituted=dict(service=SERVICE_MODULE, worker=WORKER_MODULE),
                threshold_wording="gen2 threshold bounds gen1 collections between full collections; NOT a memory cap")
