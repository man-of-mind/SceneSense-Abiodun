"""Prospective identity of ONE short live measurement of the external noAE/UINT8/no-drop split baseline (catalogue
action 0) on the current qualified runtime.

Versioned copy of ``splitfusion_run5_full_live_gc_v2`` (adopted GT prep-worker GC policy + numeric producer v2, fused
UE input prep v2, early-GT v2 pipeline, CPU partition, thread caps, FAVORABLE_STABLE, inclusive 170-ms deadline).
The ONLY functional addition is the forced-action layer (``forced.py``, adapted from
``splitfusion_fixed_action_screen_v3.forced``): the frozen full Run-5 actor still runs in shadow on every decision
(its proposal is logged and discarded) and the registered action 0 = (mode 0, q_e4 0) is executed on every
POLICY_DECISION and its holds. Action 0 is an EXTERNAL fixed baseline: it lies outside the actor's mode-0 q support
[8812, 9800] and is NOT clamped into it; the forced layer records the out-of-support status instead of refusing.
Fallback frames keep the retained a71 fallback unchanged and are reported separately (never counted as baseline).
Budget 100 only, producer v2, GC arm "candidate" (= adopted policy) only. No retention, no retry.

Header of the copied gc_v2 contract follows.

Prospective identity of the guarded full Run-5 live launcher with the adopted GC policy AND numeric producer v2.

Versioned copy of ``splitfusion_run5_full_live_gc_v1`` (the GC-qualification launcher, whose registered experiment
verdict REJECT and separate engineering-adoption decision are unchanged). Additions only:
* producer selection ``producer`` in {"v1", "v2"} for the GT prep worker. "v2" installs, inside the guarded prep
  worker only and before its engine is constructed, the two registered substitutions of
  ``splitfusion_numeric_gt_producer_v2.worker_switch``:
    1. ``splitfusion_exact_gt_service_proto_v2.deferred.NS`` -> proxy whose ``NumericSourceProducer`` is V2
       (producer construction);
    2. ``splitfusion_numeric_gt_integration_v2.ownership.NumericSourceProducer`` -> V2 (the ownership guard's
       accepted EXACT type; it still refuses any other class).
  "v1" is the rollback: no substitution. The worker records the actually constructed producer class.
* the V2 package sources are part of the registered exact sources.

Original header of the copied contract follows.

Prospective identity of the paired live qualification of the GT prep-worker GC policy.

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

PACKAGE = "rl_agent/splitfusion_noae_baseline_live_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_noae_baseline_live_v1"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_noae_baseline_live_v1"
RUNNER = "rl_agent.splitfusion_noae_baseline_live_v1.runner"
PARTITION = "rl_agent.splitfusion_noae_baseline_live_v1.partition"
SERVICE_MODULE = "rl_agent.splitfusion_exact_gt_live_v2.service"
SERVICE_ENTRY = "rl_agent.splitfusion_noae_baseline_live_v1.service_entry"
WORKER_MODULE = "rl_agent.splitfusion_exact_gt_service_proto_v2.worker"
WORKER_ENTRY = "rl_agent.splitfusion_noae_baseline_live_v1.worker_entry"
ID = "RUN5_NOAE_BASELINE_A0_FAVORABLE_V1"
PRODUCERS = ("v1", "v2")
PRODUCER_CLASS = {"v1": "NumericSourceProducer", "v2": "NumericSourceProducerV2"}
PRODUCER_SOURCES = ("rl_agent/splitfusion_numeric_gt_producer_v2/__init__.py",
                    "rl_agent/splitfusion_numeric_gt_producer_v2/exact_frame.py",
                    "rl_agent/splitfusion_numeric_gt_producer_v2/source_v2.py",
                    "rl_agent/splitfusion_numeric_gt_producer_v2/worker_switch.py")
SUBSTITUTIONS = (
    dict(target="rl_agent.splitfusion_exact_gt_service_proto_v2.deferred.NS", value="proxy module, NumericSourceProducer = NumericSourceProducerV2",
         purpose="producer construction"),
    dict(target="rl_agent.splitfusion_numeric_gt_integration_v2.ownership.NumericSourceProducer", value="NumericSourceProducerV2",
         purpose="ownership guard accepted exact type (type(producer) is ... still exact)"))
ARMS = ("control", "candidate")
BUDGETS = (100,)
TRACE_DIR = "gt_gc_trace"
GC_SOURCES = ("rl_agent/splitfusion_gt_worker_spike_fix_v1/__init__.py",
              "rl_agent/splitfusion_gt_worker_spike_fix_v1/gc_policy.py")
INPUT_SOURCES = V1.INPUT_SOURCES
EXPECTED_CHANGED_FROM_V1 = V1.EXPECTED_CHANGED_FROM_V1
SOCKET_NAME, CAPABILITY_NAME = V1.SOCKET_NAME, V1.CAPABILITY_NAME
sha, canonical = E.sha, E.canonical
limits, pipeline_delta = V1.limits, V1.pipeline_delta

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


# This launcher runs ONE forced external-baseline measurement (adopted GC policy + producer v2). Its validity conditions
# (full registered verdict: experiments/splitfusion_noae_baseline_live_v1/ACCEPTANCE.json):
VALIDATION = dict(
    integration="splitfusion_noae_baseline_live_v1_analysis.verdict.evaluate == VALID_BASELINE_MEASUREMENT (forced-screen "
                "amendment-02 conditions with G8 forced-vs-proposal mismatch and P0 report-only; timeliness, delivery and "
                "coverage are outcomes)",
    gc_trace="traces present, roles as registered, GC policy only in the prep worker, GC state restored, no overflow, no safety stop",
    producer="prep worker constructed exactly the registered producer class; both substitutions restored at exit",
    reported="GT readiness, feedback latency, timeliness, memory, exact rescoring, existing quality metrics; earlier sessions descriptive only",
    frame_budget="retained rules: G1 on TRANSMITTED_BUDGET_REACHED or the verified whole-cycle boundary")


def trace_limits(budget):
    """Tracing is bounded by the configured run length; overflow is counted, never silent."""
    budget = int(budget)
    return dict(max_ops=16 * budget, max_gen2_events=4 * budget, max_policy_events=4 * budget)


def token(arm, budget=100, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    if (arm, producer, int(budget)) != ("candidate", "v2", 100):
        raise ValueError("the baseline runs only the adopted GC policy, producer v2, budget 100")
    return "RUN5_NOAE_BASELINE_A0_FAVORABLE_CANDIDATE_V2_B0100_ONCE_EXECUTE_V1"


def own_sources(root):
    root = Path(root).resolve(strict=True)
    out = dict(V1.own_sources(root))
    out.update({str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))})
    out.update({name: sha(root / name) for name in GC_SOURCES})
    out.update({name: sha(root / name) for name in PRODUCER_SOURCES})
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
            raise ValueError("missing real noAE-baseline trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("noAE-baseline source drift: " + relative)
    return dict(sorted(result.items()))


def lineage(root, actor_boundary, support_sha256, budget, arm, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    return canonical(dict(actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=forced_document(), budget=int(budget),
        gt_gc_arm=arm, gc_policy=GC_POLICY, gc_safety=SAFETY, numeric_producer=producer,
        producer_substitutions=list(SUBSTITUTIONS) if producer == "v2" else []))


def gc_registration(arm, budget, producer="v2"):
    if producer not in PRODUCERS:
        raise ValueError("unknown producer")
    return dict(arm=arm, policy=GC_POLICY if arm == "candidate" else None, safety=SAFETY, validation=VALIDATION,
                producer=producer, producer_class=PRODUCER_CLASS[producer],
                producer_substitutions=list(SUBSTITUTIONS) if producer == "v2" else [],
                producer_rollback="producer v1 through the same launcher (no substitution)",
                trace_limits=trace_limits(budget), trace_dir=TRACE_DIR, role="prep",
                trampolines=dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY),
                substituted=dict(service=SERVICE_MODULE, worker=WORKER_MODULE),
                threshold_wording="gen2 threshold bounds gen1 collections between full collections; NOT a memory cap")


# ---- the registered external fixed baseline (the only functional addition over gc_v2)
FORCED_ACTION = dict(action_id=0, profile_id="split_noae_uint8_q0000", family="noAE", quantizer="UINT8", mode_id=0, q_e4=0,
                     keep_count=21504, drop_count=0, measurement_status="MEASURED_ANCHOR",
                     execution_bundle_sha256="583cbab6b68449495e779259507fdf7dcdfba56b5d31894ab9f8955da4deb10a",
                     identity_source="DYNAMIC_EXECUTION_CONTRACT_RESOLVE_Q_E4")
FORCED_KEYS = tuple(FORCED_ACTION)
FALLBACK_IDENTITY = (11, 9800, 71, "split_ae32_uint4_q9800")
SUPPORT_POLICY = "EXTERNAL_BASELINE_OUTSIDE_ACTOR_SUPPORT_RECORDED_NOT_CLAMPED"


def forced_document():
    return dict(FORCED_ACTION)


def resolved_forced(contract=None):
    """The dynamic execution contract's own resolution of (0, 0); must equal FORCED_ACTION exactly (checked at prepare)."""
    from rl_agent.splitfusion_fixed_action_screen_v3 import contract as V3
    return V3.resolve(FORCED_ACTION["mode_id"], FORCED_ACTION["q_e4"], contract)


def check_forced_shape(forced):
    """Contract-free check usable in the child and in the edge container: exactly the registered action-0 document."""
    if forced != FORCED_ACTION:
        raise ValueError("forced action is not the registered noAE/UINT8/no-drop action 0 document")
    return forced


def metrics():
    return dict(V1.metrics(), action_selection="REGISTERED_EXTERNAL_FIXED_BASELINE_A0_FORCED_FROZEN_ACTOR_IN_SHADOW_V1",
                forced_action_layer=True, proposal_logged_and_discarded=True,
                forced_support_policy=SUPPORT_POLICY, policy_decision_agreement_gate_claimed=False)
