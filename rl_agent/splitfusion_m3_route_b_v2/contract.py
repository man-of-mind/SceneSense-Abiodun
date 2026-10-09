"""M3.1 v2 (second GOOD_ALL full-loop attempt): the v1 launcher plus exactly two registered integration repairs.

  R-POSE  reconstructed-vs-CARLA pose comparisons (owned ego, ego/mount RGB camera, ego/mount radar, wire/source camera)
          use a magnitude-aware float32 translation tolerance: max(1e-5 m, 2 x ULP32(max |translation|)), valid only for
          max |translation| <= 512 m (=> at most 1.22e-4 m; beyond the range the comparison is REJECTED). Rotation block,
          homogeneous row, rigidity, finiteness, identity, timestamp and calibration-hash checks are unchanged (1e-5 /
          existing). Copy-of-the-same-source comparisons (owned RGB camera/radar vs snapshot, preparation sensor records,
          live-GT and producer camera checks) stay strict. Evidence: on the v1 capture all 6,028 archived reconstructed
          comparisons are <= 0.997 ULP of the largest translation (registered factor 2).
  R-CLASS the map process adds ``type`` to each installed object from the model record's own ``class_name``
          (vehicle -> Vehicle, person -> Pedestrian) on a copy, before the frozen baseline normalizer; counts, order,
          scores, coordinates and every other field are unchanged; edge records, scorer inputs and hashes are untouched.
Everything else is the v1 registration (frozen actor, GOOD_ALL 21.5 dB, radio, CPU partition, reward, deadline, capacity,
route termination, map journal, edge guard, retention rule). Header of the copied v1 contract follows.

Prospective identity of M3.1: ONE complete Town10 Route-B loop with the frozen deployed Run-5 actor under the
constant-target schedule GOOD_ALL (21.5 dB commanded target), with direct edge-to-map ingestion.

Versioned copy of ``splitfusion_run5_channel_response_v1`` (= retention launcher + one carrier-cell seam). Functional
changes, each registered below and in the lineage:
  1. SCHEDULE: the carrier cell stays the pinned a71/FAVORABLE_STABLE cell (``phase6_live_runner_v2.PROFILE`` is NOT
     substituted); inside the UE child the target-SNR controller module is substituted by ``good_all_controller``,
     which replaces only ``prepare_sequence`` with the constant GOOD_ALL sequence (state 2, 21.5 dB every 100-ms step,
     own trace_id/digest). Mapping (21.5 -> noise_power_dB -11.0, Phase-14a anchor 02), RFsim command/ACK
     validation, skip policy, causal-SNR publication and the -50 dB restore are the unchanged runtime.
  2. ROUTE TERMINATION: the run ends at completion of the planned loop (``ROUTE_COMPLETED``), not at a frame budget.
     ``FRAME_CEILING`` is a finite safety ceiling; reaching it, the safety timeout or any watchdog is an INCOMPLETE
     route. The retained post-budget ``TypeError`` is repaired at its cause (``route_completion``).
  3. CAPACITY: the 1,000-frame literals of the exact-GT pipeline are raised to FRAME_CEILING by registered,
     count-asserted source substitutions inside the owning processes (no pinned file edited); the GT archive is sized
     at its unchanged 4,000-job cap.
  4. EVIDENCE: the exact installed map records are teed to a durable map journal; the edge evaluator overflow branch
     can no longer suppress publication of a completed prediction (``edge_guard``).
Actor, 12 modes, q support, 22-D state, reward, deterministic inference, fallback, inclusive 170-ms deadline, early GT,
GC policy, producer v2, CPU partition, thread caps and the retention rule are unchanged.

Header of the copied channel-response contract follows.

Prospective identity of the ONE bounded Run-5 channel-adaptation validation (channel response v1).

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

PACKAGE = "rl_agent/splitfusion_m3_route_b_v2"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v2"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v2"
RUNNER = "rl_agent.splitfusion_m3_route_b_v2.runner"
PARTITION = "rl_agent.splitfusion_m3_route_b_v2.partition"
SERVICE_MODULE = "rl_agent.splitfusion_exact_gt_live_v2.service"
SERVICE_ENTRY = "rl_agent.splitfusion_m3_route_b_v2.service_entry"
WORKER_MODULE = "rl_agent.splitfusion_exact_gt_service_proto_v2.worker"
WORKER_ENTRY = "rl_agent.splitfusion_m3_route_b_v2.worker_entry"
ID = "M3_1_ROUTE_B_GOOD_ALL_V2"
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
# Finite frame ceiling (transmitted frames). Derived from completed full loops of this route: 2,990-3,060 frames in the
# 14 direct-edge-map validation loops (298.75-309.2 s simulated, 20 Hz ticks, one prepared frame per 2 ticks) and up
# to ~3,615 frames in the slowest 288-campaign loops (361.5 s). 3,900 = 1.28 x typical, 1.08 x slowest observed, and
# keeps the GT archive (2 jobs/decision measured: 986 jobs / 500 decisions) under its unchanged 4,000-job cap.
FRAME_CEILING = 3900
BUDGETS = (FRAME_CEILING,)
ARCHIVE_JOB_CAP = 4000          # = unchanged splitfusion_live_gt_handoff_v1.archive.MAX_CUMULATIVE_JOBS
SAFETY_TIMEOUT_S = 900.0        # route simulated-time watchdog AND collector wall timeout (same retained argument)
CHILD_TIMEOUT_S = 2400.0
HOST_ROUTE_LIFECYCLE_S = 2400   # host_capture "Bounded route lifecycle" (retained 1000 s)
TRACE_DIR = "gt_gc_trace"
# ---- paired-input retention (the only addition over gc_v2)
RETENTION_PACKAGE = "rl_agent/splitfusion_paired_retention_v2"
RETENTION_SOURCES = tuple(f"{RETENTION_PACKAGE}/{name}" for name in ("__init__.py", "plan.py", "observer.py", "writer.py"))
RETENTION_POSTFLIGHT_SOURCES = tuple(f"{RETENTION_PACKAGE}/{name}" for name in ("validate.py", "gt_adapter.py"))
RETENTION_PLAN_FILE = "RETENTION_PLAN.json"
RETENTION_DIR = "paired_retention_v2"
RETENTION_RESULT_FILE = "RETENTION_RESULT.json"
RETENTION_PLAN_ID = "M3_1_ROUTE_B_GOOD_ALL_V2_P3000_STRIDE15"
# Retention is planned over the registered TYPICAL loop length (3,000 frames -> 1,500 decisions -> stride 15, offset 7),
# not the ceiling, so the 100 ordinals span the whole loop; decisions past 1,500 are counted SKIPPED_BEYOND_PLAN.
RETENTION_PLANNED_FRAMES = 3000
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
metrics, pipeline_delta = V1.metrics, V1.pipeline_delta


def limits(budget):
    """Retained exact-GT limits with max_frames = the registered frame ceiling (no other limit changes)."""
    if int(budget) not in BUDGETS:
        raise ValueError("budget is not the registered M3 frame ceiling")
    return dict(E.LIMITS, max_frames=int(budget))

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


# ---- GOOD_ALL constant-target schedule (replaces the channel-response FADE_RECOVERY carrier substitution)
ORIGINAL_PROFILE = "FAVORABLE_STABLE"      # phase6_live_runner_v2.PROFILE: carrier cell identity, NOT substituted
NETWORK_PROFILE = ORIGINAL_PROFILE          # the resolved carrier cell (a71__favorable_stable) is a launch identity only
CARRIER_ACTION = 71
LIVE_CONFIG = "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
CARRIER_TRACE = dict(profile_id="FAVORABLE_STABLE", trace_id="GM_V2_FAVORABLE_STABLE_SEED_2026082101", seed=2026082101)
CONTROLLER_MODULE = "rl_agent.splitfusion_m3_route_b_v2.good_all_controller"
RETAINED_CONTROLLER_MODULE = "rl_agent.splitfusion_run5_live_avo_v1.snr_live"
SCHEDULE = dict(schedule_id="GOOD_ALL", trace_id="CONST_GOOD_ALL_21P5DB_V1", seed=0, target_snr_db=21.5,
                state_index=2, kind="CONSTANT_TARGET_EVERY_STEP",
                expected_rfsim_command_db=-11.0, mapping_point="PHASE14A_100MHZ_ANCHOR_02",
                not_the_stochastic_profile="FAVORABLE_STABLE is a Gaussian-Markov trace (profile mean ~20.0 dB); GOOD_ALL "
                                           "is a new deterministic constant schedule and is reported under its own name",
                hold_semantics="constant by design; the config's forbidden_after_prefix (wrap/hold_final_value/reseed) "
                               "governs the Gaussian-Markov traces and is not violated: no stochastic trace is replayed",
                actor_observation="the retained causal SNR proxy is the ACKed COMMANDED TARGET (controller publication), "
                                  "constant 21.5 dB -> feature 22 = (21.5-5.5)/19 = 0.842105...; it is not a measured "
                                  "achieved SNR; schedule identity/step/route progress never enter the actor state")
LATER_SCHEDULES_NOT_IMPLEMENTED = ("BAD_ALL", "GOOD_TO_BAD", "BAD_TO_GOOD")


def schedule_prefix(count):
    """The GOOD_ALL prefix in the retained (state, target) form, and its digest computed like prepare_sequence."""
    import numpy as np
    prefix = [(int(SCHEDULE["state_index"]), float(SCHEDULE["target_snr_db"]))] * int(count)
    states = np.asarray([row[0] for row in prefix], dtype="<i4")
    targets = np.asarray([row[1] for row in prefix], dtype="<f8")
    return prefix, __import__("hashlib").sha256(states.tobytes() + targets.tobytes()).hexdigest()


def channel_registration(root):
    """Hash-bound GOOD_ALL identity: pinned config/design/traces/mapping (unchanged), the carrier cell, the mapped
    command of the constant target, and the constant-schedule digest."""
    import csv
    root = Path(root).resolve(strict=True)
    config = json.loads((root / LIVE_CONFIG).read_bytes())
    network = config["network"]
    carrier = [row for row in network["profiles"] if row["profile_id"] == ORIGINAL_PROFILE]
    if len(carrier) != 1 or {k: carrier[0][k] for k in CARRIER_TRACE} != CARRIER_TRACE:
        raise ValueError("carrier FAVORABLE_STABLE cell is not pinned by the live config as registered")
    for key, digest in (("design_config", "design_config_sha256"), ("traces_csv", "traces_sha256"),
                        ("mapping_csv", "mapping_sha256")):
        if sha(root / network[key]) != network[digest]:
            raise ValueError("pinned channel input drift: " + network[key])
    with (root / network["mapping_csv"]).open(newline="") as handle:
        points = [row for row in csv.DictReader(handle)]
    anchor = [row for row in points if float(next(iter(row.values()))) == SCHEDULE["target_snr_db"]]
    if len(anchor) != 1 or float(list(anchor[0].values())[1]) != SCHEDULE["expected_rfsim_command_db"]:
        raise ValueError("GOOD_ALL target is not the registered measured mapping point")
    count = int(network["prefix_samples"])
    _, digest = schedule_prefix(count)
    return dict(schedule=dict(SCHEDULE, prefix_samples=count, trace_sha256=digest,
                              continuation="CONSTANT (same value every 100-ms step after the prefix)"),
                carrier_cell=dict(action=CARRIER_ACTION, network_profile=ORIGINAL_PROFILE, trace=dict(CARRIER_TRACE),
                                  profile_substituted=False),
                controller_module=CONTROLLER_MODULE, retained_controller_module=RETAINED_CONTROLLER_MODULE,
                seam="UE child sets splitfusion_run5_full_live_v1.runtime_policy.CONTROLLER_MODULE; the controller "
                     "replaces only ue_target_snr_cell_runtime_v1.prepare_sequence and rewrites --profile-id to GOOD_ALL",
                live_config=LIVE_CONFIG, live_config_sha256=sha(root / LIVE_CONFIG),
                design_config=network["design_config"], design_config_sha256=network["design_config_sha256"],
                traces_csv=network["traces_csv"], traces_sha256=network["traces_sha256"],
                mapping_csv=network["mapping_csv"], mapping_sha256=network["mapping_sha256"],
                sample_period_ms=int(network["sample_period_ms"]), catch_up_policy=network["catch_up_policy"],
                radio_profile=network["radio_baseline"]["profile_id"],
                restore="retained runtime: noise_power_dB -50 commanded, read back and verified at teardown",
                later_schedules_not_implemented=list(LATER_SCHEDULES_NOT_IMPLEMENTED),
                transitions_depend_on_actor=False)


# ---- route completion (registered before launch)
ROUTE_JSON = "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json"
PROGRESS_CSV = "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.progress.csv"
PROGRESS_MATCH_M = 8.0
PROGRESS_COVERAGE_MIN = 0.98
ROUTE_COMPLETION = dict(
    terminal="ROUTE_COMPLETED: run_route_b returned accepted with route_completed, the durable route summary persisted "
             "with matching identity, and the collector made no early stop (no frame-ceiling / safety-timeout / "
             "decision-cycle stop)",
    runner_criteria=dict(all_ordered_waypoints_reached=True, waypoints_expected=19, completion_radius_m=6.0,
                         completion_heading_tolerance_deg=25.0, abort_reason="", watchdog_aborted=False, loops=1,
                         route_outcome="classification in the retained ue_route_b_split_cell_adapter_v1."
                                       "ROUTE_ACCEPTABLE_OUTCOMES (PASS; INTERVENED with permitted roadblock clearing; "
                                       "proven collision-only; proven no-pedestrian-braking) -- the run_route_b "
                                       "'accepted' result, unchanged"),
    progress_evidence=dict(source="UE pose archive ego_world per transmitted frame (scientific, measured pose)",
                           reference=PROGRESS_CSV, match_radius_m=PROGRESS_MATCH_M,
                           ordered=True, minimum_ordered_coverage=PROGRESS_COVERAGE_MIN),
    incomplete=("FRAME_CEILING reached, SAFETY_TIMEOUT_EXPIRED, route watchdog/abort, missing/mismatched route summary, "
                "collector failure: INCOMPLETE_ROUTE (evidence kept, never a pass, no retry)"),
    final_cycle_exception=("P3_HOLD_TICKET admits exactly one single-frame decision group iff it is the last decision of "
                           "the last session, its frame is the last transmitted frame and it is reward-requested "
                           "(the loop can end between a decision and its hold); every other group needs >= 2 frames"))


def route_registration(root):
    root = Path(root).resolve(strict=True)
    config = json.loads((root / LIVE_CONFIG).read_bytes())
    route = config["route_b"]
    if route["route_json"] != ROUTE_JSON or route.get("progress_csv") != PROGRESS_CSV:
        raise ValueError("live config does not bind the registered full-map loop")
    document = json.loads((root / ROUTE_JSON).read_bytes())
    if len(document["intermediate_waypoints"]) + 1 != ROUTE_COMPLETION["runner_criteria"]["waypoints_expected"]:
        raise ValueError("route waypoint count drift")
    return dict(route_json=ROUTE_JSON, route_json_sha256=sha(root / ROUTE_JSON), route_name=document["name"],
                progress_csv=PROGRESS_CSV, progress_csv_sha256=sha(root / PROGRESS_CSV),
                scenario_seed=route["scenario_seed"], traffic_manager_seed=route["traffic_manager_seed"],
                completion=ROUTE_COMPLETION, frame_ceiling=FRAME_CEILING, safety_timeout_s=SAFETY_TIMEOUT_S,
                child_timeout_s=CHILD_TIMEOUT_S, host_route_lifecycle_s=HOST_ROUTE_LIFECYCLE_S,
                cadence="20 Hz world ticks (fixed_delta 0.05 s, real-time paced), one prepared frame per 2 ticks",
                derivation="full loops 298.75-309.2 s simulated (2,990-3,060 frames; 14 direct-edge-map loops), slowest "
                           "observed 361.5 s (~3,615 frames, 288 campaign); ceiling 3,900 frames; watchdog 900 s")


# ---- capacity substitutions (registered; count-asserted source text inside the owning process; files unchanged)
CAPACITY_SUBSTITUTIONS = (
    dict(id="C1_PREP_DEFERRED_ENGINE", process="GT prep worker (worker_entry)",
         target="rl_agent.splitfusion_exact_gt_service_proto_v2.deferred:DeferredProducerEngine.__init__",
         replacements=[["1 <= max_frames <= 1000", f"1 <= max_frames <= {FRAME_CEILING}"],
                       ["1 <= source_max_frames <= 1000", f"1 <= source_max_frames <= {FRAME_CEILING}"]]),
    dict(id="C2_PREP_PRODUCER_SOURCE", process="GT prep worker (worker_entry)",
         target="rl_agent.splitfusion_numeric_gt_producer_v1.source:NumericSourceProducer.__init__",
         replacements=[["1 <= max_frames <= 1000", f"1 <= max_frames <= {FRAME_CEILING}"]]),
    dict(id="C3_SERVICE_ARCHIVE_SIZING", process="GT service (service_entry)",
         target="rl_agent.splitfusion_exact_gt_live_v2.service:Service.__init__",
         replacements=[['max_frames=4 * self.limits["max_frames"]', f"max_frames={ARCHIVE_JOB_CAP}"]]),
    dict(id="C4_UE_TICKETS", process="UE child (entries.child)",
         target="rl_agent.splitfusion_exact_gt_live_v2.ue:EarlyGtUeHandoff.stage_before_action",
         replacements=[["len(self._tickets) < 1000", f"len(self._tickets) < {FRAME_CEILING}"]]),
    dict(id="C5_UE_POSE_ARCHIVE", process="UE child (entries.child)",
         target="rl_agent.splitfusion_live_parity_runtime_v1.pose:PoseArchive.__init__",
         replacements=[["1 <= max_frames <= 1000", f"1 <= max_frames <= {FRAME_CEILING}"]]),
)
POSE = dict(floor_m=1e-5, ulps=2, coordinate_range_m=512.0,
            max_tolerance_m=2 * 6.103515625e-05,
            scale="ULP32 of the largest absolute translation component of the two compared matrices (norm-wise; the "
                  "composed float32 translation error is set by the largest term, not by each element's own magnitude)",
            evidence="v1 capture: 2,412 ego/mount radar + 2,412 ego/mount RGB camera + 1,204 wire/source camera "
                     "comparisons, max 0.988 / 0.997 / 0.963 ULP; failing frame 3765 1.043e-5 m at ~128.6 m = 0.68 ULP",
            strict_unchanged=["rotation 3x3 block <= 1e-5", "homogeneous row <= 1e-5 (and _matrix 1e-8)",
                              "rigid rotation / finite / shape (_matrix)", "owned RGB camera vs snapshot",
                              "owned radar vs snapshot", "preparation sensor records", "live-GT camera checks",
                              "numeric producer camera checks", "identity/frame/timestamp/calibration hashes"])
POSE_SUBSTITUTIONS = (
    dict(id="P1_OWNED_POSE_RECONSTRUCTED", process="UE child (entries.child)",
         target="rl_agent.splitfusion_live_parity_runtime_v1.pose:capture_owned_pose",
         replacements=[['_agree(ego, owned["ego"].get_matrix(), "owned ego")',
                        '_M3_AGREE_RECONSTRUCTED(ego, owned["ego"].get_matrix(), "owned ego")'],
                       ['_agree(ego @ np.asarray(calibration.camera_to_ego_values), camera, "ego/mount RGB camera")',
                        '_M3_AGREE_RECONSTRUCTED(ego @ np.asarray(calibration.camera_to_ego_values), camera, "ego/mount RGB camera")'],
                       ['_agree(ego @ carla_transform_matrix(RADAR_MOUNT), radar, "ego/mount radar")',
                        '_M3_AGREE_RECONSTRUCTED(ego @ carla_transform_matrix(RADAR_MOUNT), radar, "ego/mount radar")']]),
    dict(id="P2_WIRE_SOURCE_CAMERA", process="UE child (entries.child)",
         target="rl_agent.splitfusion_live_parity_runtime_v1.pose:install_ue_pose", nested_classes=True,
         replacements=[['delta = _agree(context_camera(context), source.matrix("camera"), "wire/source camera")',
                        'delta = _M3_AGREE_RECONSTRUCTED(context_camera(context), source.matrix("camera"), "wire/source camera")']]),
)
MAP_SERVER_TRAMPOLINE = "rl_agent/splitfusion_single_gpu_profiler_v1/m3v2_map_profiled_v1.py"
RETAINED_MAP_SERVER = "rl_agent/splitfusion_single_gpu_profiler_v1/map_profiled_v1.py"
MAP_CLASS = dict(source_field="class_name", target_field="type", mapping={"vehicle": "Vehicle", "person": "Pedestrian"},
                 unknown_class="left without type (baseline -> Unknown) and counted", gt_used=False,
                 scope="every Run-4 map update installed (policy decisions, holds, fallbacks) in the map process only",
                 unchanged="object count, order, scores, coordinates, all other fields; edge records, scorer inputs, "
                           "hashes, confidence thresholds and filtering",
                 seam="seams_ue_v1.MAP_SERVER -> m3v2_map_profiled_v1.py (sampler/marker substring map_profiled_v1.py "
                      "kept), which wraps phase6_map_server_v2.install to convert on a copy before the frozen install")


def pose_registration():
    return dict(rule=dict(POSE), substitutions=[dict(item) for item in POSE_SUBSTITUTIONS])


def map_class_registration():
    return dict(MAP_CLASS, trampoline=MAP_SERVER_TRAMPOLINE, retained=RETAINED_MAP_SERVER)


UNCHANGED_LIMITS = ("archive MAX_CUMULATIVE_JOBS 4000 and MAX_BOOKKEEPING_BYTES 64 MiB (archive child, file unchanged)",
                    "outstanding job/byte credits, prep_jobs 4, scorer_jobs 4, freeze_queue 1, max_waiting 2, "
                    "ue_freeze_queue 2, horizon_ms 2000, drain_s 120", "duplicate refusal, durability, back-pressure",
                    "pose archive 32,768 events / 64 MiB", "map installed-frame history 4,096 (rolling)",
                    "SNR provider history 8,192 snapshots (~819 s of 100-ms steps)")


def capacity_registration():
    return dict(frame_ceiling=FRAME_CEILING, archive_job_cap=ARCHIVE_JOB_CAP,
                substitutions=[dict(item) for item in CAPACITY_SUBSTITUTIONS], unchanged=list(UNCHANGED_LIMITS),
                mechanism="inspect.getsource of the exact pinned function, each replacement must occur exactly once, "
                          "no zero-argument super()/closure, compiled under a registered filename and installed on "
                          "the owning class in that process only; every installation is recorded")


# ---- evidence additions (no runtime decision path change)
MAP_JOURNAL_DIR = "m3_map"
def evidence_registration():
    return dict(
        finish_result="route_completion.FinishResult: the retained EarlyGt finish assigns into the inner finish result, "
                      "which the base collector returns as bool -> TypeError after the GT handoff closed; a dict with "
                      "__bool__ == the cleanup flag is returned below EarlyGt (cause repaired, nothing suppressed)",
        map_journal=dict(directory=MAP_JOURNAL_DIR, files=["installed_map_records.jsonl", "map_state_latest.json"],
                         source="exact installed record the UE adapter already retrieves from the map's bounded "
                                "installed-frame history (GET /api/fusion_streams/installed/<stream>/<frame>) on its "
                                "retrieval worker; raw bytes teed, never re-fetched, never on the decision path",
                         coordinates="measured edge-output world coordinates as installed; no display prediction or "
                                     "extrapolation exists on this path (render off)"),
        edge_guard=dict(change="an exception from evaluator.submit (incl. the retained queue-full synchronous fault send) "
                               "is counted and recorded and the completed prediction still proceeds to map publication",
                        normal_path="unchanged (submit returns normally)", feedback_on_guarded_error="none sent; the UE "
                        "resolves the ticket by its retained timeout"),
        map_delivery_separation=("map publication follows edge compute only; it never waits for GT, scoring, reward or "
                                 "feedback; the 170-ms decision deadline is not a map rule; map freshness = retained "
                                 "500-ms capture-age horizon + newest-capture supersession"))


def trace_limits(budget):
    """Tracing is bounded by the configured run length; overflow is counted, never silent."""
    budget = int(budget)
    return dict(max_ops=16 * budget, max_gen2_events=4 * budget, max_policy_events=4 * budget)


def token(arm, budget=FRAME_CEILING, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    return f"M3_1_ROUTE_B_GOOD_ALL_V2_{arm.upper()}_{producer.upper()}_C{int(budget):04d}_ONCE_EXECUTE_V1"


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
    for path in [root / (module.replace(".", "/") + ".py") for module in (CHILD, EDGE)] + [root / MAP_SERVER_TRAMPOLINE]:
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
        channel=channel_registration(root), route=route_registration(root), capacity=capacity_registration(),
        evidence_additions=evidence_registration(), pose_repair=pose_registration(), map_class=map_class_registration(),
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
    if int(budget) not in BUDGETS:
        raise ValueError("budget is not the registered M3 frame ceiling")
    return RP.build_plan(plan_id=RETENTION_PLAN_ID, planned_frames=RETENTION_PLANNED_FRAMES, **RETENTION_PLAN_ARGS)


def retention_registration(plan_sha256):
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise ValueError("retention plan digest required")
    return dict(enabled=True, plan_file=RETENTION_PLAN_FILE, plan_sha256=plan_sha256, plan_id=RETENTION_PLAN_ID,
                plan_args=dict(RETENTION_PLAN_ARGS), directory=RETENTION_DIR, result_file=RETENTION_RESULT_FILE,
                observer="splitfusion_m3_route_b_v2.retention.AttemptRetentionObserver (subclass of "
                         "splitfusion_paired_retention_v2.observer.PairedRetentionObserverV2; records the owning UE "
                         "frame-worker Thread and bounded hot-path bookkeeping durations only)",
                scope="innermost child context, entered after install_fused_candidate, install_ue_pose, install_early_gt_ue",
                drain="finalize only after CHILD.main returned AND the owning UE frame-worker thread is_alive() is False",
                gt="numeric-GT service archive by reference; resolved postflight with exact rescoring",
                admission=RETENTION_ADMISSION)
