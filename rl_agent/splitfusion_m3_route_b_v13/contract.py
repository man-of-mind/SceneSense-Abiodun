"""M3.3 v13 prospective identity: the qualified V11 runtime with ONE registered change, the channel schedule
(one BAD_ALL Route-B evaluation capture).

Relative to V11 (``M3_1_ROUTE_B_GOOD_ALL_V11``, the completed 2026-10-09 capture, registration content
``56bd0bc1...``) only the commanded channel schedule changes, from the constant GOOD_ALL 23.5 dB (anchor 01) to the
constant BAD_ALL 8.5 dB (anchor 09), together with the identity/binding fields that must name this revision.
Everything else -- actor, reward, 12 modes, q support, 22 causal features and their normalisation, route, density,
seeds, permitted interventions, 170-ms inclusive deadline, corrected prep-only V3.v2 storage accounting + per-call
timer + cache bounds, GT definitions, tilt-aware AVO, registered P3, GC policy, input processing, map
publication/freshness, retention, capacity C1-C5, radio telemetry, real-ACK end-of-route drain, abort exporter,
100 MHz / 273 PRB / 4D5U radio profile, scheduler, power control, MCS policy, CPU partition and the per-role launcher
binding pattern -- is the V11 value, re-declared here under the V13 identity.

1. BAD_ALL = constant 8.5 dB nominal -> RFsim noise_power_dB -4.0, the existing Phase-14A 100-MHz anchor 09 of the
   pinned mapping (``target_to_rfsim_mapping.csv``) whose calibration record (``anchor_summary.csv`` row 09:
   per-transmission PUSCH median 8.5 / P05 7.5 / P95 8.5 dB, selected/final MCS median 9) is hash-bound below and
   re-read at registration.  The calibration is reused unchanged: no sweep, no outcome-driven adjustment, MCS NOT
   forced.  BAD_ALL is a new deterministic constant schedule (state index 0 = ADVERSE of the design's state order),
   reported under its own name and kept separate from the historical stochastic ADVERSE_STABLE profile.  The
   registered schedule, its digest and the causal commanded-target actor input change together (native v13
   controller module reading this contract); the actor sees (8.5 - 5.5) / 19 = 0.157894..., inside its unchanged
   [5.5, 24.5] dB support.  No gNB measurement reaches the actor; normalisation, weights, power control, MCS policy
   and support are unchanged.
2. Channel fidelity (evidence only, registered BEFORE launch, never a gate on the integration verdict and never a
   trigger for a retry): see ``CHANNEL_FIDELITY``.

Everything else is v11 (and through it v10/v8/v7/v4/v3).
"""
from pathlib import Path
import sys

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_m3_route_b_v8 import contract as V8
from rl_agent.splitfusion_m3_route_b_v11 import contract as V11
from rl_agent.splitfusion_m3_route_b_v12 import contract as V12
from rl_agent.splitfusion_m3_route_b_v8.contract import *  # noqa: F401,F403

PACKAGE = "rl_agent/splitfusion_m3_route_b_v13"
MODULE = "rl_agent.splitfusion_m3_route_b_v13"
LAUNCHER_REVISION = "M3_3_ROUTE_B_STARTUP_BINDING_V13"
ID = "M3_3_ROUTE_B_GOOD_TO_BAD_V13"
PREVIOUS_ID = V12.ID
RUNTIME_REVISION = ID
SCIENTIFIC_RUNTIME_BASIS = V4.ID

CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v13"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v13"
RUNNER = MODULE + ".runner"
PARTITION = MODULE + ".partition"
HOST = MODULE + ".host"
ENTRIES = MODULE + ".entries"
GC_SEAM = MODULE + ".gc_seams"
RETENTION = MODULE + ".retention"
SERVICE_ENTRY = MODULE + ".service_entry"           # native copy of the v8 service_entry reading this contract
WORKER_ENTRY = MODULE + ".worker_entry"             # prep: storage accounting + timer; both roles -> V8_WORKER_ENTRY
V8_WORKER_ENTRY = V8.WORKER_ENTRY                   # unchanged v8 trampoline: prep tilt-aware AVO -> v7 repair
V7_WORKER_ENTRY = V7.WORKER_ENTRY
EDGE_REGISTRATION_PATH = "/partition_output/registration.json"
VERIFY_FLAG = "--m3-verify-startup"
SHIMS = {f"{PACKAGE}/runtime_shims/child_v1_m3_route_b_v13.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_m3_route_b_v13.py",
         f"{PACKAGE}/runtime_shims/edge_profiled_v1_m3_route_b_v13.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_m3_route_b_v13.py"}

# ---- 1. corrected exact GT storage accounting V3.v2 (prep only), at the reviewed final digests
#         (experiments/splitfusion_gt_storage_accounting_v3/SOURCES_SHA256_02.txt and EVIDENCE_SHA256_02.txt)
ACCOUNTING_PACKAGE = "rl_agent/splitfusion_gt_storage_accounting_v3"
ACCOUNTING_MODULE = "rl_agent.splitfusion_gt_storage_accounting_v3"
ACCOUNTING_QUALIFIED = {
    f"{ACCOUNTING_PACKAGE}/__init__.py": "871029a0f75c9115b42642403f2d23eba48a58f149d25b0e0e0b3407177f5391",
    f"{ACCOUNTING_PACKAGE}/accounting.py": "b6f4a655b65f98ac10d31227372eba78c60672fba7c25a55debe03199f391baa",
    f"{ACCOUNTING_PACKAGE}/switch.py": "3510b0f118eab1ea6ac2b94db9b9cd0cf18ce40d3d7ab17836c4ec76890eeb79"}
# V3 imports V1 for the unchanged scalar/logical-byte definition. V2 stays registered as the comparison package
# used by the reviewed qualification harnesses, but is never installed by the V11/V12 production worker.
ACCOUNTING_V1_PACKAGE = "rl_agent/splitfusion_gt_storage_accounting_v1"
ACCOUNTING_V1_QUALIFIED = {
    f"{ACCOUNTING_V1_PACKAGE}/__init__.py": "9e497c924f8546282affea1655498e1e44eaf7ab1b66de89dcf870538522e28e",
    f"{ACCOUNTING_V1_PACKAGE}/accounting.py": "79a8966b5c127d52a7f00e295ab0225eb290b6616b021c3b5986950d90b1b5a0",
    f"{ACCOUNTING_V1_PACKAGE}/switch.py": "d86bdc9fec57cd23a9a42ff84eec9ffe7b489c1881074d8024cf60545c0631cb"}
ACCOUNTING_V2_PACKAGE = "rl_agent/splitfusion_gt_storage_accounting_v2"
ACCOUNTING_V2_QUALIFIED = {
    f"{ACCOUNTING_V2_PACKAGE}/__init__.py": "8a0e1a50fbb7f5a54e838ef9573078a620466ed0864f3e7af26ad213a6f1c7d2",
    f"{ACCOUNTING_V2_PACKAGE}/accounting.py": "f8e66a8e209e99366207ea9ab97ce24591dd67ee27f69c89b3271acc634c8e18",
    f"{ACCOUNTING_V2_PACKAGE}/switch.py": "7033b5345ffa80e4463a5087ec64c2a15331b0017d3d1645e8b3ba8d3a48f5e5"}
ACCOUNTING_VERSION = "splitfusion_gt_storage_accounting_v3.per_object_reachable_set.v2"
ACCOUNTING_SWITCH = ACCOUNTING_MODULE + ".switch"
ACCOUNTING_SUBSTITUTION = dict(target="rl_agent.splitfusion_numeric_gt_integration_v2.ownership:payload_storage_bytes",
                               replacement=ACCOUNTING_MODULE + ".accounting:AccountingSession",
                               process="GT prep worker only", default="OFF")
ACCOUNTING_MAX_OBJECTS = 32768
ACCOUNTING_REQUIRED_CAPABILITIES = ("dict_tags", "mappingproxy", "pair_history")
ACCOUNTING_BINDING_RECORD = "storage_accounting_binding.json"
ACCOUNTING_TIMING_RECORD = "storage_accounting_timing.json"
ACCOUNTING_TIMING_CAPACITY = 16 * FRAME_CEILING     # = trace_limits max_ops; overflow counted, never silent
# Candidate evidence reused (main checkout, each sealed by its OUTPUT_SHA256.json): RESULT/SUMMARY digests.
ACCOUNTING_EVIDENCE_DIR = "experiments/splitfusion_gt_storage_accounting_v3"
ACCOUNTING_EVIDENCE_FILES = {
    "replay_qualification_V10_02": "SUMMARY.json",
    "replay_qualification_V8_02": "SUMMARY.json",
    "process_qualification_V10_03": "SUMMARY.json",
    "process_qualification_V8_02": "SUMMARY.json",
    "output_parity_03": "SUMMARY.json",
    "service_proof_02": "RESULT.json",
    "final_tests_02": "FAULT_INJECTION.json"}

# ---- 1. GOOD_TO_BAD at 50% of ordered planned-route distance; carrier cell unchanged
CONTROLLER_MODULE = MODULE + ".good_to_bad_controller"
ROUTE_PROGRESS_MODULE = MODULE + ".route_progress"
GOOD_SCHEDULE = dict(V11.SCHEDULE)
BAD_SCHEDULE = dict(V12.SCHEDULE)
TRANSITION_FRACTION = 0.5
PROGRESS_MAX_SAMPLE_AGE_MS = 1000.0
PROGRESS_STARTUP_GRACE_MS = 180_000.0
PROGRESS_TO_COMMAND_MAX_MS = 1100.0
PROGRESS_SNAPSHOT_RELATIVE = "run5_policy/ROUTE_PROGRESS.json"
PROGRESS_MAX_CROSS_TRACK_M = 8.0
PROGRESS_INITIAL_SEGMENTS = 20
PROGRESS_BACKTRACK_SEGMENTS = 4
PROGRESS_FORWARD_SEGMENTS = 40
PROGRESS_MAX_FORWARD_MARGIN_M = 10.0
PROGRESS_DISPLACEMENT_MULTIPLIER = 3.0
PROGRESS_FORWARD_ALLOWANCE_M = 5.0
ROUTE_JSON_SHA256 = "fc4518a8746b9417a64616b8e544f59b16b5a31b7585298a316a59662ecfd6e5"
PROGRESS_CSV_SHA256 = "974593859368f24ee2bc4ac31b82118bf2e932d0de1c96858b8771e2dd4d90c0"
ANCHOR_EVIDENCE = dict(
    path="experiments/splitfusion_phase14a_100mhz_calibration_v1/20260904_phase14a_live_calibration_retry3/"
         "anchor_summary.csv",
    sha256="279027290439a2fce3600675b0057df6bcc36a65cf4ae8400cdb4ed148a0829b",
    anchor_index=9, applied_noise_power_db=-4.0, achieved_median_pusch_snr_db=8.5, achieved_pusch_snr_p05_db=7.5,
    achieved_pusch_snr_p95_db=8.5, uncertainty_half_width_db=0.5, selected_mcs_median=9.0, final_mcs_median=9.0,
    pusch_sample_count=649, mcs_sample_count=652, status="ANCHOR_ACCEPTED")
SCHEDULE = dict(
    schedule_id="GOOD_TO_BAD", trace_id="ROUTE_HALF_GOOD23P5_TO_BAD8P5_V1",
    kind="ORDERED_PLANNED_ROUTE_DISTANCE_LATCHED_TWO_PHASE",
    transition_fraction=TRANSITION_FRACTION,
    phases=[
        dict(phase="GOOD", interval="[0, 50%)", state_index=int(GOOD_SCHEDULE["state_index"]),
             target_snr_db=float(GOOD_SCHEDULE["target_snr_db"]),
             expected_rfsim_command_db=float(GOOD_SCHEDULE["expected_rfsim_command_db"]),
             mapping_point=GOOD_SCHEDULE["mapping_point"]),
        dict(phase="BAD", interval="[50%, 100%]", state_index=int(BAD_SCHEDULE["state_index"]),
             target_snr_db=float(BAD_SCHEDULE["target_snr_db"]),
             expected_rfsim_command_db=float(BAD_SCHEDULE["expected_rfsim_command_db"]),
             mapping_point=BAD_SCHEDULE["mapping_point"]),
    ],
    trigger="first causal route-progress sample whose monotone cumulative planned-route distance is >= 50%",
    latch="BAD is permanent; backtracking, jitter and loop closure cannot restore GOOD",
    startup="GOOD while no route sample exists, bounded by the registered startup grace",
    stale_progress="after the first active route sample, age > 1000 ms aborts the controller fail-closed",
    command_failure="the retained command/response validator aborts and restores RFsim fail-closed",
    actor_observation="only the ACKed commanded target reaches feature 22; route progress, phase identity and future "
                      "settings never enter actor state; measured gNB SNR is evaluation telemetry only")
SCHEDULE_FIELDS_CHANGED = tuple(sorted(SCHEDULE))
ACTOR_SNR_FEATURES = {"23.5": (23.5 - 5.5) / 19.0, "8.5": (8.5 - 5.5) / 19.0}
LATER_SCHEDULES_NOT_IMPLEMENTED = ("BAD_TO_GOOD",)
DESIGN_STATE_ORDER = ("ADVERSE", "INTERMEDIATE", "FAVORABLE")     # network_profile_design_v2 target_snr.state_order

# ---- 2. channel fidelity, registered before launch (evidence only; the integration verdict never reads it)
CHANNEL_FIDELITY = dict(
    window="route-only, split at the registered actual transition; startup attach and restoration excluded",
    measured="calibrated per-transmission PUSCH snrx10/10 from m3_radio/ttracer/live_pusch_with_ingest.csv",
    checks=dict(
        F1_good_pusch_median_db=dict(rule="GOOD phase median within qualified 23.0..24.0 dB", low=23.0, high=24.0),
        F2_bad_pusch_median_db=dict(rule="BAD phase median within anchor-09 8.0..9.0 dB", low=8.0, high=9.0),
        F3_commanded_phases=dict(rule="23.5/-12 before the single transition; 8.5/-4 after it"),
        F4_transition_delay=dict(rule="first BAD command issue no later than 1100 ms after the crossing sample",
                                 high_ms=PROGRESS_TO_COMMAND_MAX_MS),
        F5_selected_mcs_medians=dict(rule="reported outcomes of automatic MCS, never forced",
                                     good_reference=27, bad_reference=9)),
    reported_alongside=["scheduler avg_snr (EMA) P50/P95", "selected/final MCS histogram", "granted PRBs split by "
                        "gNB-reported buffer", "independent pre-action RLC backlog", "ULSCH rounds / retransmissions "
                        "/ errors / DTX deltas", "MAC-log power-control SNR, separately labelled, never mixed"],
    on_mismatch="reported as CHANNEL_FIDELITY_MISMATCH with phase-specific values; no retry or adjustment; the "
                "integration verdict is unaffected and the capture is still analysed under its registered name")


def schedule_definition_sha256(root):
    return canonical(dict(schedule=SCHEDULE, route_transition=route_transition_registration(root)))


def route_transition_registration(root):
    from . import route_progress
    return route_progress.registration(root)


def channel_registration(root):
    """Register both already-qualified anchors and the causal route-distance trigger."""
    import csv
    import json
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
    for phase in SCHEDULE["phases"]:
        anchor = [row for row in points if float(next(iter(row.values()))) == phase["target_snr_db"]]
        if (len(anchor) != 1 or float(list(anchor[0].values())[1]) != phase["expected_rfsim_command_db"]
                or list(anchor[0].values())[2] != phase["mapping_point"]):
            raise ValueError("GOOD_TO_BAD phase is not a registered measured mapping point")
    from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as R5
    if any(R5.scale_snr_db(float(value)) != scaled for value, scaled in ACTOR_SNR_FEATURES.items()):
        raise ValueError("GOOD_TO_BAD target is outside the actor's registered SNR support or scaling drifted")
    design = json.loads((root / network["design_config"]).read_bytes())
    if (tuple(design["target_snr"]["state_order"]) != DESIGN_STATE_ORDER
            or DESIGN_STATE_ORDER[GOOD_SCHEDULE["state_index"]] != "FAVORABLE"
            or DESIGN_STATE_ORDER[BAD_SCHEDULE["state_index"]] != "ADVERSE"):
        raise ValueError("GOOD_TO_BAD state indices differ from the pinned design")
    anchor_path = root / ANCHOR_EVIDENCE["path"]
    if sha(anchor_path) != ANCHOR_EVIDENCE["sha256"]:
        raise ValueError("Phase-14A anchor calibration record drift: " + ANCHOR_EVIDENCE["path"])
    with anchor_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["anchor_index"]) == ANCHOR_EVIDENCE["anchor_index"]]
    measured = None if len(rows) != 1 else dict(
        applied_noise_power_db=float(rows[0]["applied_noise_power_db"]),
        achieved_median_pusch_snr_db=float(rows[0]["achieved_median_pusch_snr_db"]),
        achieved_pusch_snr_p05_db=float(rows[0]["achieved_pusch_snr_p05_db"]),
        achieved_pusch_snr_p95_db=float(rows[0]["achieved_pusch_snr_p95_db"]),
        uncertainty_half_width_db=float(rows[0]["uncertainty_half_width_db"]),
        selected_mcs_median=float(rows[0]["selected_mcs_median"]), final_mcs_median=float(rows[0]["final_mcs_median"]),
        pusch_sample_count=int(rows[0]["pusch_sample_count"]), mcs_sample_count=int(rows[0]["mcs_sample_count"]),
        status=rows[0]["status"])
    if measured is None or any(measured[key] != ANCHOR_EVIDENCE[key] for key in measured) \
            or measured["applied_noise_power_db"] != BAD_SCHEDULE["expected_rfsim_command_db"] \
            or measured["achieved_median_pusch_snr_db"] != BAD_SCHEDULE["target_snr_db"]:
        raise ValueError("anchor 09 calibration row differs from the registered BAD_ALL anchor evidence")
    count = int(network["prefix_samples"])
    transition = route_transition_registration(root)
    return dict(schedule=dict(SCHEDULE, carrier_validation_prefix_samples=count, dynamic_prefix_samples=0,
                              trace_sha256=schedule_definition_sha256(root),
                              max_progress_sample_age_ms=PROGRESS_MAX_SAMPLE_AGE_MS,
                              startup_grace_ms=PROGRESS_STARTUP_GRACE_MS,
                              max_progress_to_command_ms=PROGRESS_TO_COMMAND_MAX_MS,
                              route_transition=transition),
                carrier_cell=dict(action=CARRIER_ACTION, network_profile=ORIGINAL_PROFILE, trace=dict(CARRIER_TRACE),
                                  profile_substituted=False),
                controller_module=CONTROLLER_MODULE, retained_controller_module=RETAINED_CONTROLLER_MODULE,
                seam="UE child installs the route-progress tick observer and sets runtime_policy.CONTROLLER_MODULE; "
                     "the controller replaces prepare_sequence and rewrites the carrier profile to GOOD_TO_BAD",
                live_config=LIVE_CONFIG, live_config_sha256=sha(root / LIVE_CONFIG),
                design_config=network["design_config"], design_config_sha256=network["design_config_sha256"],
                traces_csv=network["traces_csv"], traces_sha256=network["traces_sha256"],
                mapping_csv=network["mapping_csv"], mapping_sha256=network["mapping_sha256"],
                sample_period_ms=int(network["sample_period_ms"]), catch_up_policy=network["catch_up_policy"],
                radio_profile=network["radio_baseline"]["profile_id"],
                restore="retained runtime: noise_power_dB -50 commanded, read back and verified at teardown",
                later_schedules_not_implemented=list(LATER_SCHEDULES_NOT_IMPLEMENTED),
                transitions_depend_on_actor=False,
                actor_input=dict(source="ACKed commanded target (retained controller publication)",
                                 feature_index=22, values=[8.5, 23.5], scaled_values=ACTOR_SNR_FEATURES,
                                 route_progress_to_actor=False, phase_identity_to_actor=False,
                                 future_settings_to_actor=False, support_db=[5.5, 24.5],
                                 normalisation_changed=False, gnb_measurement_to_actor=False),
                anchor_evidence=dict(ANCHOR_EVIDENCE), design_state_order=list(DESIGN_STATE_ORDER),
                fidelity=CHANNEL_FIDELITY,
                changed_from_v12=dict(change="constant BAD_ALL becomes one route-distance-triggered GOOD_TO_BAD "
                                             "transition; both anchor levels and all non-channel runtime stay fixed",
                                      v12_schedule=V12.SCHEDULE, v12_actor_feature=V12.ACTOR_SNR_FEATURE,
                                      power_control_changed=False, mcs_policy_changed=False, mcs_forced=False,
                                      calibration_resweep=False))


# ---- 3. calibrated radio telemetry (evidence only; the actor input is unchanged)
RADIO_TELEMETRY_MODULE = MODULE + ".radio_telemetry"
RADIO_TELEMETRY_MAX_ROWS = 2_000_000                 # per stream (~110 rows/s measured); overflow counted
TRACER_DIR = "OAI/openairinterface5g/common/utils/T/tracer"
T_MESSAGES = "OAI/openairinterface5g/common/utils/T/T_messages.txt"
T_MESSAGES_SHA256 = "2f4945814ad2f47197816756819afc0d087156c499e8c746cf0d9a81653005c5"
MAC_LOG_OBSERVER = "rl_agent.splitfusion_m3_route_b_v3.radio_observer"   # unchanged; separately labelled


def _t_event_fields(root, event):
    """The field names of one T event, read from the tracer database that the collectors are given."""
    lines = (Path(root) / T_MESSAGES).read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "ID = " + event:
            for follow in lines[index + 1:]:
                if follow.strip().startswith("FORMAT = "):
                    return tuple(item.split(",")[1].strip()
                                 for item in follow.split("FORMAT = ", 1)[1].split(" : "))
                if follow.strip().startswith("ID = "):
                    break
    raise ValueError("T event not found in the tracer database: " + event)


def radio_telemetry_registration(root):
    """Hash-bound identity of the calibrated channel telemetry: database, events, fields, placement and semantics."""
    from . import radio_telemetry as RT
    root = Path(root).resolve(strict=True)
    if sha(root / T_MESSAGES) != T_MESSAGES_SHA256:
        raise ValueError("T-tracer database drift: " + T_MESSAGES)
    for event, fields in ((RT.PUSCH_EVENT, RT.PUSCH_FIELDS), (RT.MCS_EVENT, RT.MCS_FIELDS)):
        if ("time",) + _t_event_fields(root, event) != tuple(fields):
            raise ValueError("collector field list differs from the tracer database: " + event)
    for name in ("multi", "csv"):
        if not (root / TRACER_DIR / name).is_file():
            raise ValueError("T-tracer tool missing: " + name)
    return dict(
        module=RADIO_TELEMETRY_MODULE, database=T_MESSAGES, database_sha256=T_MESSAGES_SHA256,
        relay=dict(tool=f"{TRACER_DIR}/multi", gnb_port=RT.GNB_T_PORT, relay_port=RT.RELAY_PORT,
                   why="the gNB T server accepts one client and closes its listener while connected"),
        client=dict(tool=f"{TRACER_DIR}/csv", flags=["-f", "-s", ",", "-t", "time"]),
        events={RT.PUSCH_EVENT: list(RT.PUSCH_FIELDS), RT.MCS_EVENT: list(RT.MCS_FIELDS)},
        quantities=dict(
            pusch_snr_db=f"{RT.PUSCH_EVENT}.snrx10/10, per transmission (Phase-14A calibrated channel SNR)",
            scheduler_avg_snr_db=f"{RT.MCS_EVENT}.avg_snr_x10/10 (EMA; read by the SINR MCS lookup)",
            mcs=f"{RT.MCS_EVENT}.selected_mcs and .final_mcs; {RT.PUSCH_EVENT}.mcs",
            mac_log_snr=dict(kept=True, where="m3_radio/gnb_stats_observed.jsonl", observer=MAC_LOG_OBSERVER,
                             label="power-control SNR (avg_snr + tpc_in_flight), NOT channel SNR")),
        placement=dict(cpus="the registered radio CPUs of this attempt (arm.json radio_cpus)", mechanism="taskset",
                       rationale="the gNB tracer worker these read from is already pinned there"),
        ingest="runner-parent daemon drain threads; each line prefixed with host wall_ns and CLOCK_MONOTONIC_RAW ns",
        bounds=dict(max_rows_per_stream=RADIO_TELEMETRY_MAX_ROWS, overflow="counted, never silent"),
        lifecycle=dict(start="after _start_live_radio returned (radio attached, its cold check already run)",
                       stop="before the retained _stop_live_radio teardown; the relay port must be free again "
                            "for the post-teardown cold proof"),
        decision_path=dict(inference=False, gt=False, scoring=False, map=False, synchronous_work=False),
        actor_input="unchanged: the ACKed commanded target; no gNB measurement reaches the actor",
        evidence=["m3_radio/ttracer/live_pusch_with_ingest.csv", "m3_radio/ttracer/live_mcs_with_ingest.csv",
                  "m3_radio/RADIO_TELEMETRY.json"],
        qualification="experiments/splitfusion_m3_good_all_snr_qual_v1/20261008_run01")


# ---- 4. prospective bookkeeping repairs
DRAIN_SETTLE_MARGIN_S = 0.25        # after an obligation's own ack_timeout_at, before the retained watchdog closes it
DRAIN_IN_FLIGHT_CAP_S = 10.0        # the one token already inside _process_token (base slot drain is 10 s)
DRAIN_ABSOLUTE_CAP_S = 20.0         # hard bound on the whole drain
ROUTE_DRAIN_MODULE = MODULE + ".route_drain"
POSTFLIGHT_P3_MODULE = MODULE + ".postflight_p3"
# The already-qualified explicit registration binding for the postflight assessment (first built and
# regression-covered in the V10 registered assessment): every expectation the integration verdict compares
# against is derived from THIS verified registration, so no older revision's module globals can supply a schedule.
OFFLINE_ASSESSMENT_PACKAGE = "rl_agent/splitfusion_m3_route_b_v13_offline_v1"
OFFLINE_ASSESSMENT_MODULE = "rl_agent.splitfusion_m3_route_b_v13_offline_v1.postflight"
OFFLINE_ASSESSMENT_BINDING = "rl_agent.splitfusion_m3_route_b_v13_offline_v1.binding"
OFFLINE_ASSESSMENT_INTEGRATION = "rl_agent.splitfusion_m3_route_b_v13_offline_v1.integration"
V3_PINNED_VERDICT_TOOL = "rl_agent.splitfusion_m3_route_b_v3_analysis.integration"


def assessment_registration():
    """The registered postflight assessment binding: the verdict is evaluated against THIS registration."""
    return dict(module=OFFLINE_ASSESSMENT_MODULE, binding=OFFLINE_ASSESSMENT_BINDING,
                integration=OFFLINE_ASSESSMENT_INTEGRATION, package=OFFLINE_ASSESSMENT_PACKAGE,
                derives="schedule, controller module, progress coverage, trace dir, pose rule, map class and map "
                        "journal directory from the verified registration, after verify_prospective and "
                        "verify_prerequisite",
                refuses="any declared revision other than this one; any registration whose assessment fields differ "
                        "from the declared contract; any storage-accounting version other than the registered one",
                numeric_host_exit_status="CAPTURED at launch and required by the integration verdict",
                v3_pinned_verdict_tool=dict(module=V3_PINNED_VERDICT_TOOL, authority=False,
                                            why="its module-global contract supplies the V3 21.5-dB schedule to "
                                                "three SCHEDULE_* checks; it is "
                                                "still run unmodified and recorded beside the bound assessment"),
                qualification="V10 registration-bound assessment pattern (binding regression: the legitimate "
                              "prospective accepted, six isolated negatives rejected -- wrong target, controller, "
                              "trace identity, runtime/registration revision, exact-source binding and runtime "
                              "source inventory)")


def bookkeeping_registration():
    from . import postflight_p3 as P3
    from . import route_drain as RD
    return dict(end_of_route_drain=RD.drain_registration(), offline_p3=P3.postflight_registration(),
                registered_assessment=assessment_registration(),
                v8_verdict_preserved="experiments/splitfusion_m3_route_b_v8 (REPORT.md, LIVE_CAPTURE_REPORT.md, "
                                     "capture_01, analysis_01_ROUTE_COMPLETED, presentation_route_layers_01_"
                                     "ROUTE_COMPLETED) is unchanged; V8's registered verdict and evidence stand")


# ---- identity, sources and binding
ROLE_CONSTANTS = (
    ("rl_agent.splitfusion_m3_route_b_v3.entries", "v3", ("MAP_SERVER_TRAMPOLINE", "RETAINED_MAP_SERVER")),
    ("rl_agent.splitfusion_m3_route_b_v3.capacity", "v3", ("CAPACITY_SUBSTITUTIONS", "POSE_SUBSTITUTIONS")),
    ("rl_agent.splitfusion_m3_route_b_v3.pose_tolerance", "v3", ("POSE",)),
    ("rl_agent.splitfusion_m3_route_b_v3.route_completion", "v3", ("MAP_JOURNAL_DIR",)),
    ("rl_agent.splitfusion_m3_route_b_v3.map_class", "v3", ("MAP_CLASS",)),
    ("rl_agent.splitfusion_m3_route_b_v3.retention", "v3", ("RETENTION_DIR", "RETENTION_RESULT_FILE")),
    ("rl_agent.splitfusion_m3_route_b_v3.worker_entry", "v4",
     ("ARMS", "BUDGETS", "GC_POLICY", "POSE", "PRODUCERS", "PRODUCER_CLASS", "SAFETY", "trace_limits")),
)
# Neither the v3 nor the v11 GOOD_ALL controller is executed in a v13 run; the v13 BAD_ALL controller is, and it reads
# this contract.  The controller installer is likewise the native v13 one.
CONTROLLER_CONSTANTS = ("ORIGINAL_PROFILE", "SCHEDULE", "CONTROLLER_MODULE", "RETAINED_CONTROLLER_MODULE")
V3_ENTRY_HELPERS = ("install_ue_capacity", "install_map_trampoline", "_write")


def repo_root():
    return Path(__file__).resolve().parents[2]


def module_identity(function):
    """Import name of the module that defines ``function`` (a ``-m`` entry point executes as ``__main__``)."""
    spec = getattr(sys.modules.get(function.__module__), "__spec__", None)
    return spec.name if spec is not None else function.__module__


def role_constant_audit():
    """Per (module, name): the value the retained module reads equals the v13 value (functions: same object)."""
    contracts = dict(v3=V3, v4=V4)
    out = {}
    for module, bound, names in ROLE_CONSTANTS:
        for name in names:
            seen, mine = getattr(contracts[bound], name), globals()[name]
            out[f"{module}[C={bound}].{name}"] = (seen is mine) if callable(mine) else (seen == mine)
    from . import good_to_bad_controller as GC
    for name in CONTROLLER_CONSTANTS:
        out[f"{CONTROLLER_MODULE}.{name}"] = getattr(GC.C, name) == globals()[name]
    out[f"{CONTROLLER_MODULE}.contract"] = GC.C.__name__ == __name__
    return out


def token(arm, budget=FRAME_CEILING, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    return f"M3_3_ROUTE_B_GOOD_TO_BAD_V13_{arm.upper()}_{producer.upper()}_C{int(budget):04d}_ONCE_EXECUTE_V1"


def accounting_registration():
    return dict(version=ACCOUNTING_VERSION, switch=ACCOUNTING_SWITCH, substitution=dict(ACCOUNTING_SUBSTITUTION),
                roles=["prep"], sources=dict(ACCOUNTING_QUALIFIED), v1_sources=dict(ACCOUNTING_V1_QUALIFIED),
                v2_comparison_sources=dict(ACCOUNTING_V2_QUALIFIED),
                value_contract="returns exactly the original payload_storage_bytes integer; same exceptions",
                cache=dict(max_objects=ACCOUNTING_MAX_OBJECTS,
                           owner_cache="the last owner call only (stage_products -> bind_plan -> set_ready, or "
                                       "stage_products -> set_ready, with identical staged/products identities)",
                           snapshot_context="computed afresh from reachable(snapshot) minus the caller's current "
                                            "seen set; no partial snapshot subtotal is reusable",
                           mutation_guard="whole-graph witness: every reached dict/list/set/object identity and "
                                          "ndarray (dtype.hasobject, nbytes); mismatch performs the exact traversal",
                           required_capabilities=list(ACCOUNTING_REQUIRED_CAPABILITIES),
                           cleanup="worker finally: timer removed, session.snapshot() recorded, session restored as "
                                   "the owner global, then switch.uninstall() -> AccountingSession.close()",
                           cleanup_record=f"{TRACE_DIR}/{ACCOUNTING_BINDING_RECORD}:accounting_session"),
                reservation_bytes_unchanged=True, capacity_rule_unchanged=True,
                timing=dict(module=MODULE + ".accounting_timing", capacity=ACCOUNTING_TIMING_CAPACITY,
                            records="per call: CLOCK_MONOTONIC_RAW begin/end, thread CPU ns, returned bytes, the "
                                    "ProducerOwner method that called _account",
                            why="the diagnosis fitted the late-route accounting cost; it was never measured per call",
                            overflow="counted in dropped, never silent", writes_on_request_path=False),
                binding_record=f"{TRACE_DIR}/{ACCOUNTING_BINDING_RECORD}",
                timing_record=f"{TRACE_DIR}/{ACCOUNTING_TIMING_RECORD}",
                qualification=dict(directory=ACCOUNTING_EVIDENCE_DIR, records=dict(ACCOUNTING_EVIDENCE_FILES)))


def gc_registration(arm, budget, producer="v2"):
    value = V8.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(value["implementations"], service=SERVICE_ENTRY, v8_worker=V8_WORKER_ENTRY)
    value["storage_accounting"] = accounting_registration()
    value["seam"] = GC_SEAM
    return value


def retention_registration(plan_sha256):
    value = V3.retention_registration(plan_sha256)
    value["plan_id"] = RETENTION_PLAN_ID
    value["loader"] = RETENTION + ".load_observer"
    return value


def retention_plan(budget):
    return V4.retention_plan(budget)


def own_sources(root):
    """v4 sources, every file of this package and of the sealed v7/v8/v11 packages (v11: this contract imports the
    v11 contract for the schedule it replaces), the FREEZE lifetime repair, the tilt-aware AVO and the exact
    storage-accounting runtime files (V3, the V1 base it imports and the V2 comparison package)."""
    root = Path(root).resolve(strict=True)
    result = dict(V4.own_sources(root))
    for package in (V7.PACKAGE, V8.PACKAGE, V11.PACKAGE, V12.PACKAGE, PACKAGE, OFFLINE_ASSESSMENT_PACKAGE):
        result.update({str(p.relative_to(root)): sha(p) for p in sorted((root / package).glob("*.py"))})
    for qualified in (PREP_LIFETIME_QUALIFIED, AVO_TILT_QUALIFIED, ACCOUNTING_QUALIFIED,
                      ACCOUNTING_V1_QUALIFIED, ACCOUNTING_V2_QUALIFIED):
        result.update({name: sha(root / name) for name in qualified})
    return result


def exact_sources(root):
    result = dict(V3.E.exact_sources(root))
    result.update(own_sources(root))
    return dict(sorted(result.items()))


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = dict(V3.E.source_inventory(root))
    result.update(own_sources(root))
    for path in [root / (module.replace(".", "/") + ".py") for module in (CHILD, EDGE)] + [root / MAP_SERVER_TRAMPOLINE]:
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing real M3 v13 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M3 v13 source drift: " + relative)
    for label, qualified in (("prep lifetime repair", PREP_LIFETIME_QUALIFIED), ("tilt-aware AVO", AVO_TILT_QUALIFIED),
                             ("storage accounting", ACCOUNTING_QUALIFIED),
                             ("storage accounting v1 base", ACCOUNTING_V1_QUALIFIED),
                             ("storage accounting v2 comparison", ACCOUNTING_V2_QUALIFIED)):
        for relative, digest in qualified.items():
            if result[relative] != digest:
                raise ValueError(f"{label} differs from the qualified source: " + relative)
    return dict(sorted(result.items()))


def launcher_binding():
    """The exact module identity of every validating function, per process role (registered and re-checked)."""
    return dict(
        launcher_revision=LAUNCHER_REVISION, runtime_revision=RUNTIME_REVISION,
        scientific_runtime_basis=SCIENTIFIC_RUNTIME_BASIS, contract_module=MODULE + ".contract",
        host=dict(verify_prospective=HOST, verify_prerequisite=HOST, check_prerequisites=HOST,
                  verify_startup_proof=HOST),
        runner=dict(module=RUNNER, validate_parent=RUNNER, verify_prospective=HOST, verify_prerequisite=HOST,
                    radio_telemetry=RADIO_TELEMETRY_MODULE, mac_log_observer=MAC_LOG_OBSERVER),
        child=dict(module=CHILD, entries=ENTRIES, validate_child_startup=ENTRIES, verify_prospective=HOST,
                   registration=ENTRIES, load_observer=RETENTION, seam_parameters=GC_SEAM,
                   controller=CONTROLLER_MODULE, route_progress=ROUTE_PROGRESS_MODULE,
                   collector_layer=ROUTE_DRAIN_MODULE),
        edge=dict(module=EDGE, entries=ENTRIES, validate_edge_startup=ENTRIES, registration=ENTRIES,
                  registration_path=EDGE_REGISTRATION_PATH),
        gt_service=dict(seam=GC_SEAM, trampoline=SERVICE_ENTRY, implementation=SERVICE_ENTRY,
                        retained_service=SERVICE_MODULE,
                        registration_check="rl_agent.splitfusion_exact_gt_live_v2.service.verify_sources"),
        gt_workers=dict(trampoline=WORKER_ENTRY, v8_trampoline=V8_WORKER_ENTRY, v7_trampoline=V7_WORKER_ENTRY,
                        retained_trampoline=RETAINED_WORKER_ENTRY,
                        implementation="rl_agent.splitfusion_m3_route_b_v3.worker_entry",
                        prep_core=PREP_LIFETIME_MODULE + ".PrepCoreLifetime",
                        prep_avo=AVO_SUBSTITUTION["replacement"], prep_avo_switch=AVO_SWITCH,
                        prep_accounting=ACCOUNTING_SUBSTITUTION["replacement"], prep_accounting_switch=ACCOUNTING_SWITCH,
                        score_core="rl_agent.splitfusion_exact_gt_service_proto_v2.worker.ScoreCore",
                        registration_check=None),
        postflight=dict(p3=POSTFLIGHT_P3_MODULE, assessment=OFFLINE_ASSESSMENT_MODULE,
                        assessment_binding=OFFLINE_ASSESSMENT_BINDING,
                        assessment_integration=OFFLINE_ASSESSMENT_INTEGRATION),
        partition=PARTITION)


def lineage(root, actor_boundary, support_sha256, budget, arm, producer="v2", retention_plan_sha256=None):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    if not isinstance(retention_plan_sha256, str) or len(retention_plan_sha256) != 64:
        raise ValueError("retention launcher lineage requires the pre-launch retention plan digest")
    return canonical(dict(
        launcher_revision=LAUNCHER_REVISION, runtime_revision=RUNTIME_REVISION,
        scientific_runtime_basis=SCIENTIFIC_RUNTIME_BASIS, launcher_binding=launcher_binding(),
        retention_plan_sha256=retention_plan_sha256,
        retention_registration=retention_registration(retention_plan_sha256),
        channel=channel_registration(root), route=route_registration(root), capacity=capacity_registration(),
        evidence_additions=evidence_registration(), pose_repair=pose_registration(), map_class=map_class_registration(),
        avo_geometry=avo_registration(), storage_accounting=accounting_registration(),
        radio_telemetry=radio_telemetry_registration(root), bookkeeping=bookkeeping_registration(),
        actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=None, budget=int(budget),
        gt_gc_arm=arm, gc_policy=GC_POLICY, gc_safety=SAFETY, numeric_producer=producer,
        producer_substitutions=list(SUBSTITUTIONS) if producer == "v2" else []))
