"""Prospective identity of the ONE bounded Run-5 channel-adaptation validation (channel response v1).

Versioned copy of ``splitfusion_run5_full_live_retention_v1`` (gc_v2 launcher + paired-input retention). The ONLY
functional change is the network profile of the single registered carrier cell:
  FAVORABLE_STABLE -> FADE_RECOVERY (action 71 cell, already pinned by the unchanged direct-edge-map live config and
  its network_profile_design_v2 design; trace GM_V2_FADE_RECOVERY_SEED_2026082104, seed 2026082104, prefix digest
  ef2ae2d6...). It is selected by substituting ``phase6_live_runner_v2.PROFILE`` inside this package's runner (the
  pinned runner module is not edited) and the resolved cell is verified (action, profile, trace, seed) before any
  process starts. The unchanged target-SNR runtime regenerates the trace from the pinned design, verifies its digest
  against the pinned config and the accepted traces.csv, and replays it from step 0 at the first-capture gate.
No radio code, mapping, actuator, RFsim configuration, actor, q support, state, reward, deadline, session semantics,
GC policy, producer, partition, thread caps or retention plan rule changes. The plan rule is identical; only its id
differs so the two captures cannot be confused.

Header of the copied retention contract follows.

Prospective identity of the guarded full Run-5 live launcher (adopted GC policy + numeric producer v2) WITH bounded
paired-input retention (quality diagnostic).

Versioned copy of ``splitfusion_run5_full_live_gc_v2``. The ONLY addition is the opt-in, pre-registered paired-input
retention of ``splitfusion_paired_retention_v2`` in the UE child:
* a sampling plan written create-only BEFORE launch (``RETENTION_PLAN.json``; rule CENTERED_STRIDE_ON_POLICY_DECISION_
  ORDINAL_V1, 1,000 planned frames -> 500 planned decisions -> stride 5, offset 2, 100 ordinals). The ordinal is an
  attempt-wide count of POLICY_DECISION materializations on the owned UE frame worker; it does NOT reset with the
  controller session. Selection never reads latency, outcome, action, payload or quality.
* POST_DRAIN persistence: the sampled prepared CPU tensor (by reference, no clone/D2H/hash/I/O on the decision path)
  and the exact transmitted SFD4 wire are held in UE-child RAM (cap 1.25 GiB) and written only after the UE frame
  worker that performed every materialization is shown dead (``RETENTION_DIR`` at the run root).
* GT, native predicted labels and complete prediction records stay in the numeric-GT service archive and are bound
  postflight by path + SHA-256 (``gt_adapter.resolve_gt_references``), with exact rescoring.
Everything else (actor, 12 modes, q support, 22 features incl. causal SNR, reward, inclusive 170-ms deadline, early-GT
pipeline, GC policy, producer v2, CPU partition, thread caps, perception weights) is the gc_v2 launcher unchanged.
A run with retention is a QUALITY DIAGNOSTIC with host-memory pressure, not an uninstrumented timing measurement.

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

PACKAGE = "rl_agent/splitfusion_run5_channel_response_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_channel_response_v1"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_channel_response_v1"
RUNNER = "rl_agent.splitfusion_run5_channel_response_v1.runner"
PARTITION = "rl_agent.splitfusion_run5_channel_response_v1.partition"
SERVICE_MODULE = "rl_agent.splitfusion_exact_gt_live_v2.service"
SERVICE_ENTRY = "rl_agent.splitfusion_run5_channel_response_v1.service_entry"
WORKER_MODULE = "rl_agent.splitfusion_exact_gt_service_proto_v2.worker"
WORKER_ENTRY = "rl_agent.splitfusion_run5_channel_response_v1.worker_entry"
ID = "RUN5_CHANNEL_RESPONSE_V1_FADE_RECOVERY"
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
BUDGETS = (1000,)
TRACE_DIR = "gt_gc_trace"
# ---- paired-input retention (the only addition over gc_v2)
RETENTION_PACKAGE = "rl_agent/splitfusion_paired_retention_v2"
RETENTION_SOURCES = tuple(f"{RETENTION_PACKAGE}/{name}" for name in ("__init__.py", "plan.py", "observer.py", "writer.py"))
RETENTION_POSTFLIGHT_SOURCES = tuple(f"{RETENTION_PACKAGE}/{name}" for name in ("validate.py", "gt_adapter.py"))
RETENTION_PLAN_FILE = "RETENTION_PLAN.json"
RETENTION_DIR = "paired_retention_v2"
RETENTION_RESULT_FILE = "RETENTION_RESULT.json"
RETENTION_PLAN_ID = "RUN5_CHANNEL_RESPONSE_V1_B1000_STRIDE5"
RETENTION_PLAN_ARGS = dict(decision_period_tensors=2, target_records=100, persistence="POST_DRAIN",
                           require_cuda_transfer=True)
RETENTION_ADMISSION = dict(
    sample_rule="ordinals fixed by the create-only pre-launch plan; never resampled; NOT_REACHED for a shorter run",
    missing_samples="every planned ordinal that is not RETAINED_COMPLETE, and every retained record whose GT reference "
                    "is not RESOLVED (unscored decision, join failure, inexact rescore), is REPORTED as missing; "
                    "comparisons use admitted samples only",
    per_sample_admission="RESOLVED GT reference with exact rescoring AND selected-branch replay reproducing the archived "
                         "inner payload bytes, native labels and complete records (sealed replay_pair gates)",
    claim_scope="QUALITY_DIAGNOSTIC__HOST_MEMORY_PRESSURE__NOT_AN_UNINSTRUMENTED_TIMING_RUN")
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


# This launcher runs ONE validation (adopted GC policy + producer v2), not an A/B qualification. Its pass conditions:
VALIDATION = dict(
    integration="splitfusion_run5_full_live_analysis_v1.integration.evaluate == PASS_ACTOR_DRIVEN_INTEGRATION (unchanged R/X/A/E/S/Q)",
    gc_trace="traces present, roles as registered, GC policy only in the prep worker, GC state restored, no overflow, no safety stop",
    producer="prep worker constructed exactly the registered producer class; both substitutions restored at exit",
    reported="GT readiness, feedback latency, timeliness, memory, exact rescoring, existing quality metrics; earlier sessions descriptive only",
    frame_budget="retained rules: G1 on TRANSMITTED_BUDGET_REACHED or the verified whole-cycle boundary")


# ---- channel selection (the only functional change over the retention launcher)
ORIGINAL_PROFILE = "FAVORABLE_STABLE"      # value of phase6_live_runner_v2.PROFILE that is substituted
NETWORK_PROFILE = "FADE_RECOVERY"
CARRIER_ACTION = 71
LIVE_CONFIG = "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
TRACE = dict(profile_id="FADE_RECOVERY", trace_id="GM_V2_FADE_RECOVERY_SEED_2026082104", seed=2026082104,
             trace_sha256="ef2ae2d6448d9c7f3208bc81123d301a1a9f5d9b16be3bac4412c8cecf94e7c0")


def channel_registration(root):
    """Hash-bound channel identity (pinned config, design, accepted traces, target->RFsim mapping); never edited here."""
    root = Path(root).resolve(strict=True)
    config = json.loads((root / LIVE_CONFIG).read_bytes())
    network = config["network"]
    frozen = [row for row in network["profiles"] if row["profile_id"] == NETWORK_PROFILE]
    if len(frozen) != 1 or {k: frozen[0][k] for k in TRACE} != TRACE:
        raise ValueError("FADE_RECOVERY trace is not pinned by the live config as registered")
    for key, digest in (("design_config", "design_config_sha256"), ("traces_csv", "traces_sha256"),
                        ("mapping_csv", "mapping_sha256")):
        if sha(root / network[key]) != network[digest]:
            raise ValueError("pinned channel input drift: " + network[key])
    return dict(network_profile=NETWORK_PROFILE, substituted_from=ORIGINAL_PROFILE, carrier_action=CARRIER_ACTION,
                live_config=LIVE_CONFIG, live_config_sha256=sha(root / LIVE_CONFIG), trace=dict(TRACE),
                design_config=network["design_config"], design_config_sha256=network["design_config_sha256"],
                traces_csv=network["traces_csv"], traces_sha256=network["traces_sha256"],
                mapping_csv=network["mapping_csv"], mapping_sha256=network["mapping_sha256"],
                sample_period_ms=int(network["sample_period_ms"]), catch_up_policy=network["catch_up_policy"],
                continuation=network["continuation"], replay="FROM_STEP_ZERO_AT_FIRST_CAPTURE_GATE",
                seam="phase6_live_runner_v2.PROFILE substituted in this package's runner.install_parent; resolved "
                     "cell verified (action 71, profile, trace_id, seed) before any process starts",
                transitions_depend_on_actor=False)


def trace_limits(budget):
    """Tracing is bounded by the configured run length; overflow is counted, never silent."""
    budget = int(budget)
    return dict(max_ops=16 * budget, max_gen2_events=4 * budget, max_policy_events=4 * budget)


def token(arm, budget=1000, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    return f"RUN5_CHANNEL_RESPONSE_V1_FADE_RECOVERY_{arm.upper()}_{producer.upper()}_B{int(budget):04d}_ONCE_EXECUTE_V1"


def own_sources(root):
    root = Path(root).resolve(strict=True)
    out = dict(V1.own_sources(root))
    out.update({str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))})
    out.update({name: sha(root / name) for name in GC_SOURCES})
    out.update({name: sha(root / name) for name in PRODUCER_SOURCES})
    out.update({name: sha(root / name) for name in RETENTION_SOURCES})
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


def lineage(root, actor_boundary, support_sha256, budget, arm, producer="v2", retention_plan_sha256=None):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    if not isinstance(retention_plan_sha256, str) or len(retention_plan_sha256) != 64:
        raise ValueError("retention launcher lineage requires the pre-launch retention plan digest")
    return canonical(dict(
        retention_plan_sha256=retention_plan_sha256, retention_registration=retention_registration(retention_plan_sha256),
        channel=channel_registration(root),
        actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=None, budget=int(budget),
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


def retention_plan(budget):
    """The plan is a pure function of the registered frame budget (no outcome/latency/action/quality input)."""
    from rl_agent.splitfusion_paired_retention_v2 import plan as RP
    return RP.build_plan(plan_id=RETENTION_PLAN_ID, planned_frames=int(budget), **RETENTION_PLAN_ARGS)


def retention_registration(plan_sha256):
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise ValueError("retention plan digest required")
    return dict(enabled=True, plan_file=RETENTION_PLAN_FILE, plan_sha256=plan_sha256, plan_id=RETENTION_PLAN_ID,
                plan_args=dict(RETENTION_PLAN_ARGS), directory=RETENTION_DIR, result_file=RETENTION_RESULT_FILE,
                observer="splitfusion_run5_channel_response_v1.retention.AttemptRetentionObserver (subclass of "
                         "splitfusion_paired_retention_v2.observer.PairedRetentionObserverV2; records the owning UE "
                         "frame-worker Thread and bounded hot-path bookkeeping durations only)",
                scope="innermost child context, entered after install_fused_candidate, install_ue_pose, install_early_gt_ue",
                drain="finalize only after CHILD.main returned AND the owning UE frame-worker thread is_alive() is False",
                gt="numeric-GT service archive by reference; resolved postflight with exact rescoring",
                admission=RETENTION_ADMISSION)
