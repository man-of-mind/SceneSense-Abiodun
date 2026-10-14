"""V19: approved position-indexed spatial field on the actual qualified V17/R1 runtime.
Native version-coherent launchers; unchanged containment/accounting/GT/radio/actor.
Only causal position -> channel target and its evidence/assessment are new.
"""
import json
from pathlib import Path
import sys

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_m3_route_b_v8 import contract as V8
from rl_agent.splitfusion_m3_route_b_v11 import contract as V11
from rl_agent.splitfusion_m3_route_b_v17 import contract as V17
from rl_agent.splitfusion_m3_route_b_v8.contract import *  # noqa: F401,F403

PACKAGE = "rl_agent/splitfusion_m3_route_b_v19"
MODULE = "rl_agent.splitfusion_m3_route_b_v19"
LAUNCHER_REVISION = "M3_SPATIAL_ROUTE_B_STARTUP_BINDING_V19"
ID = "M3_SPATIAL_ROUTE_B_V19"
PREVIOUS_ID = V11.ID
RUNTIME_REVISION = ID
SCIENTIFIC_RUNTIME_BASIS = V4.ID

CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v19"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v19"
RUNNER = MODULE + ".runner"
PARTITION = MODULE + ".partition"
HOST = MODULE + ".host"
ENTRIES = MODULE + ".entries"
GC_SEAM = MODULE + ".gc_seams"
RETENTION = MODULE + ".retention"
# ---- 0. GT geometry-refusal containment (the ONE runtime change vs V16): the reviewed package's own trampolines
CONTAINMENT_PACKAGE = "rl_agent/splitfusion_gt_geometry_containment_v1"
CONTAINMENT_MODULE = "rl_agent.splitfusion_gt_geometry_containment_v1"
SERVICE_ENTRY = CONTAINMENT_MODULE + ".service_entry"   # reviewed; code-identical to V16's, reads the containment contract
WORKER_ENTRY = CONTAINMENT_MODULE + ".worker_entry"     # reviewed: V16 accounting -> V8 AVO -> V7 lifetime -> containment
SERVICE_CONTRACT_MODULE = CONTAINMENT_MODULE + ".contract"
V16_PACKAGE = "rl_agent/splitfusion_m3_route_b_v16"
V16_WORKER_ENTRY = "rl_agent.splitfusion_m3_route_b_v16.worker_entry"   # the base stack the containment worker composes
ACCOUNTING_TIMING_MODULE = "rl_agent.splitfusion_m3_route_b_v16.accounting_timing"   # the V16 timer, unchanged
ACCOUNTING_TIMING_SCHEMA = "splitfusion.m3_route_b_v16.storage_accounting_timing.v1"
CONTAINMENT_BINDING_RECORD = "geometry_containment_binding.json"
CONTAINMENT_DIAGNOSTICS_DIR = "geometry_refusal_v1"
CONTAINMENT_VERSION = "splitfusion.gt_geometry_containment.v1"
CONTAINMENT_PREP_CORE = CONTAINMENT_MODULE + ".core.PrepCoreContained"
CONTAINMENT_ENGINE = CONTAINMENT_MODULE + ".engine.ContainedEngine"
CONTAINMENT_REASONS = {"TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET": "GT_GEOMETRY_ENVELOPE_TARGET",
                       "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_POSSIBLE_COMPETITOR": "GT_GEOMETRY_ENVELOPE_POSSIBLE_COMPETITOR"}
CONTAINMENT_OUTCOME_PREFIX = "NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_"
# Every source of the reviewed package (experiments/splitfusion_gt_geometry_containment_v1/MANIFEST_02.json).
CONTAINMENT_QUALIFIED = {
    "rl_agent/splitfusion_gt_geometry_containment_v1/__init__.py": "c9367a7dbf5cfb14ea9d6b91b95c66e243b0d2713cc5cc6962e6bd0764561811",
    "rl_agent/splitfusion_gt_geometry_containment_v1/classify.py": "428fe5afb77ad00db904de17ee3de62b9a8512ae66b594f5fc8bfadd594e6951",
    "rl_agent/splitfusion_gt_geometry_containment_v1/contract.py": "90c9b6778d22b63d99334787e5c18fbe1edc868e52fbb3ebfb6e1966c161c990",
    "rl_agent/splitfusion_gt_geometry_containment_v1/core.py": "e5b57f1ebe22a1d7537ed59aa8343210903f95fc727923e496c2c286891f140e",
    "rl_agent/splitfusion_gt_geometry_containment_v1/engine.py": "b3a99b5e5d47f8d8929dd3eae905302a428582a2f0002d0ab00647b9ebf8f594",
    "rl_agent/splitfusion_gt_geometry_containment_v1/fixtures.py": "7f70c4d05f6b753d7fbc4af9ccf50b27ce97842581c8e1a9918f38bcf4ad8d41",
    "rl_agent/splitfusion_gt_geometry_containment_v1/gc_seams.py": "8fea0336c5c09afc1415a22bd9c331ce82a4e95a726067b4c17370356f98411c",
    "rl_agent/splitfusion_gt_geometry_containment_v1/metrics.py": "9b966ca363d98cb1ade2483b6cf56613bc461e8653723a49084667014f31f665",
    "rl_agent/splitfusion_gt_geometry_containment_v1/replay_s1.py": "c0079c14bad6fcba7db4ecd37c9105278aaa1ddd4b400c8e62b31c5cfe1f6734",
    "rl_agent/splitfusion_gt_geometry_containment_v1/rerun_review_probes.py": "de2d372d9ad83ca0213c15baebb9c64559d358e94f7454a973597de6ed89ba34",
    "rl_agent/splitfusion_gt_geometry_containment_v1/service_entry.py": "1ad2806d03c17f82fb27b30873cfcfe5ce176850c56f181a861eae40d234b60a",
    "rl_agent/splitfusion_gt_geometry_containment_v1/service_proof.py": "13604caac7f0a7bdcac6864f3ccad050107e14153f77adc05ee8bdb40a190f59",
    "rl_agent/splitfusion_gt_geometry_containment_v1/test_containment.py": "28b1d8611851091014ceaa5ec9e9471661a7de8935d6dcb579eb559be366361d",
    "rl_agent/splitfusion_gt_geometry_containment_v1/verify_engine.py": "bf97f0ad9390be5ec93b1710350aa13d8281ef6ac8aac52258cd63c6162b17ea",
    "rl_agent/splitfusion_gt_geometry_containment_v1/worker_entry.py": "74cc886c21023756ecc72242df5e4c2df1b4ef09decd8bb2dadeb48686237443",
}
CONTAINMENT_EVIDENCE_DIR = "experiments/splitfusion_gt_geometry_containment_v1"
# Reviewed qualification records (MANIFEST_02.json "results" / "logs").
CONTAINMENT_EVIDENCE = {"service_proof_02/RESULT.json": "7dccac50f95887a50fbab4fe169784e04fbfa5f3cf1e63b706b8c97da328d6fb",
                        "verify_engine_02/RESULT.json": "15407c98724f3c5a0d265577bbd6b3af7ff9b9f56b5b6e3065413d4756b911eb",
                        "replay_s1_01/RESULT.json": "056ea2274588026cbd2e562935eabd90eaaae88ad755e1cb905120ab7168c584",
                        "review_rerun_01/RESULT.json": "dc993f5a1845564fb7e945c19415210bab29ca18980f9d4a6b1af5b6d4a96720",
                        "tests_02.log": "9ecca07816c6e39a6290f677b74ef85ac80647ad5b3ad14f01953a538385fe9a"}
CONTAINMENT_EVIDENCE_SEALED = ("service_proof_02/RESULT.json", "verify_engine_02/RESULT.json", "replay_s1_01/RESULT.json")
V8_WORKER_ENTRY = V8.WORKER_ENTRY                   # unchanged v8 trampoline: prep tilt-aware AVO -> v7 repair
V7_WORKER_ENTRY = V7.WORKER_ENTRY
EDGE_REGISTRATION_PATH = "/partition_output/registration.json"
VERIFY_FLAG = "--m3-verify-startup"
SHIMS = {f"{PACKAGE}/runtime_shims/child_v1_m3_route_b_v19.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_m3_route_b_v19.py",
         f"{PACKAGE}/runtime_shims/edge_profiled_v1_m3_route_b_v19.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_m3_route_b_v19.py"}

# ---- 1. corrected exact GT storage accounting V3.v2 (prep only), at the reviewed final digests
#         (experiments/splitfusion_gt_storage_accounting_v3/SOURCES_SHA256_02.txt and EVIDENCE_SHA256_02.txt)
ACCOUNTING_PACKAGE = "rl_agent/splitfusion_gt_storage_accounting_v3"
ACCOUNTING_MODULE = "rl_agent.splitfusion_gt_storage_accounting_v3"
ACCOUNTING_QUALIFIED = {
    f"{ACCOUNTING_PACKAGE}/__init__.py": "871029a0f75c9115b42642403f2d23eba48a58f149d25b0e0e0b3407177f5391",
    f"{ACCOUNTING_PACKAGE}/accounting.py": "b6f4a655b65f98ac10d31227372eba78c60672fba7c25a55debe03199f391baa",
    f"{ACCOUNTING_PACKAGE}/switch.py": "3510b0f118eab1ea6ac2b94db9b9cd0cf18ce40d3d7ab17836c4ec76890eeb79"}
# V3 imports V1 for the unchanged scalar/logical-byte definition. V2 stays registered as the comparison package
# used by the reviewed qualification harnesses, but is never installed by the V11/V19 production worker.
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

# ---- Approved synthetic position-indexed field; never a time/random trace.
CONTROLLER_MODULE = MODULE + ".spatial_controller"
POSE_MODULE = MODULE + ".spatial_pose"
POSE_SNAPSHOT_RELATIVE = "run5_policy/SPATIAL_POSE.json"
POSE_MAX_SAMPLE_AGE_MS = 1000.0       # unchanged V14 route-controller freshness bound
POSE_STARTUP_GRACE_MS = 180000.0     # wait for route's first observed pose; never synthesize a target
POSE_TO_COMMAND_MAX_MS = 1100.0      # existing route-progress command-age safeguard
POSE_TERMINAL_DRAIN_MS = 20000.0     # unchanged absolute final-frame drain cap
APPROVED_DIR = "experiments/splitfusion_m3_spatial_snr_plan_v1/proposal_03"
APPROVED_FILES = {
    APPROVED_DIR + "/PROPOSED_REGION.json": "85da62c50818a4032ac0f104f8d3a2b8ff075022b418a9135cd7abcccebd91d6",
    APPROVED_DIR + "/MANIFEST.json": "d1846159b9810586be3155d389aa31f8297f19ea4b355746bc266a227e090904",
    "experiments/splitfusion_m3_spatial_snr_plan_v1/render_plan.py": "b3e552d1a6506155ea89213d0141bb1c82ac7ef785deb8363519ad45f6e37559",
    APPROVED_DIR + "/01_proposed_spatial_snr_region.png": "385998cd4cd03cde3e0eb6e37c927c8e85879f0671fc8c4a19560f32c8f4cd75",
}
SCHEDULE = dict(
    schedule_id="SPATIAL_POSITION_FIELD", trace_id="APPROVED_ELLIPSE_POSITION_FIELD_V1", seed=0,
    kind="CAUSAL_CURRENT_EGO_POSITION_LOOKUP", sample_period_ms=100, prefix_samples=0,
    center_xy_m=[-55.0, 140.0], core_semi_axes_m=[80.0,45.0], transition_outer_scale=1.25,
    high_db="23.5 - 3.5*u^3; u=0.5+0.5*sin(2*pi*x/180+0.4)*cos(2*pi*y/160-0.7)",
    low_db="13 + sin(2*pi*(x+55)/160)",
    blend="t=clip((rho-1)/0.25,0,1); w=1-3*t^2+2*t^3; target=w*low+(1-w)*high",
    high_range_db=[20.0,23.5], low_range_db=[12.0,14.0],
    pose_schema="splitfusion.m3_route_b_v19.spatial_pose.v1",
    pose_source="CAUSAL_ROUTE_TICK_OWNER_EGO_TRANSFORM",
    pose_snapshot=POSE_SNAPSHOT_RELATIVE,
    max_pose_sample_age_ms=POSE_MAX_SAMPLE_AGE_MS, max_pose_to_command_ms=POSE_TO_COMMAND_MAX_MS,
    startup_grace_ms=POSE_STARTUP_GRACE_MS, terminal_drain_ms=POSE_TERMINAL_DRAIN_MS,
    startup="retained first-capture gate; observed initial ego pose published before route ticks; no synthetic startup target",
    terminal="last observed position held ONLY during bounded terminal drain; aborted or stale active pose fails closed",
    interpolation="retained calibration inverse_interpolate, command granularity 0.25 dB",
    position_to_actor=False, zone_to_actor=False, outcome_dependent=False, time_dependent=False,
    description="Synthetic position-indexed impairment, not measured coverage or building-derived propagation")
CHANNEL_FIDELITY = dict(
    window="route-only; per-transmission PUSCH joined to latest preceding ACK by retained host clocks",
    commanded="field target and quantized calibration command verified row by row; actor uses ACKed target only",
    measured="PUSCH snrx10/10 and MCS retained separately; report by HIGH/TRANSITION/LOW_CORE",
    report=["target-minus-measured residual P05/P50/P95 and absolute residual",
            "ACK-to-sample age; per-region PUSCH and actual/prior MCS; unmatched samples"],
    limitation="ingest/observation timestamps are not packet identities; scheduler/channel response may lag commands",
    threshold="No constant-SNR median/MCS gate; no endpoint screening or outcome-triggered retry")


def approved_design(root):
    root = Path(root)
    for name, digest in APPROVED_FILES.items():
        if not digest or sha(root/name) != digest:
            raise ValueError("approved spatial design drift: " + name)
    region = json.loads((root/APPROVED_DIR/"PROPOSED_REGION.json").read_bytes())
    if (region["center_xy_m"] != SCHEDULE["center_xy_m"]
            or region["core_semi_axes_m"] != SCHEDULE["core_semi_axes_m"]
            or region["transition_outer_scale"] != SCHEDULE["transition_outer_scale"]
            or region["spatial_field"] != {k:SCHEDULE[k] for k in ("high_db","low_db","blend")}):
        raise ValueError("spatial contract does not implement the approved design")
    return region


def schedule_definition_sha256(root):
    approved_design(root)
    return canonical(dict(schedule=SCHEDULE, approved_sources=APPROVED_FILES))


def channel_registration(root):
    # V17's pinned carrier/design/mapping/anchor checks still run unchanged.
    value = V17.channel_registration(root)
    approved_design(root)
    value.update(
        schedule=dict(SCHEDULE, trace_sha256=schedule_definition_sha256(root),
                      carrier_validation_prefix_samples=value["schedule"]["prefix_samples"]),
        controller_module=CONTROLLER_MODULE,
        seam="native V19 sequence from causal route pose; retained 100-ms actuator/mapping/ACK publication",
        actor_input=dict(source="ACKed commanded target (retained controller publication)", feature_index=22,
                         value="dynamic (ACKed target_db - 5.5)/19", support_db=[5.5,24.5],
                         normalisation_changed=False, gnb_measurement_to_actor=False),
        fidelity=CHANNEL_FIDELITY, later_schedules_not_implemented=[],
        changed_from_v11=dict(approved_spatial_sources=APPROVED_FILES,
                             base="actual V17/R1 qualified containment/accounting runtime",
                             power_control_changed=False, mcs_policy_changed=False,
                             mcs_forced=False, calibration_resweep=False))
    return value


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
OFFLINE_ASSESSMENT_PACKAGE = "rl_agent/splitfusion_m3_route_b_v19_offline_v1"
OFFLINE_ASSESSMENT_MODULE = "rl_agent.splitfusion_m3_route_b_v19_offline_v1.postflight"
OFFLINE_ASSESSMENT_BINDING = "rl_agent.splitfusion_m3_route_b_v19_offline_v1.binding"
OFFLINE_ASSESSMENT_INTEGRATION = "rl_agent.splitfusion_m3_route_b_v19_offline_v1.integration"
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
# Neither the v3 nor the v11 GOOD_ALL controller is executed in a v19 run; the v19 OPERATING_14DB controller is, and it reads
# this contract.  The controller installer is likewise the native v19 one.
CONTROLLER_CONSTANTS = ("ORIGINAL_PROFILE", "SCHEDULE", "CONTROLLER_MODULE", "RETAINED_CONTROLLER_MODULE")
V3_ENTRY_HELPERS = ("install_ue_capacity", "install_map_trampoline", "_write")


def repo_root():
    return Path(__file__).resolve().parents[2]


def module_identity(function):
    """Import name of the module that defines ``function`` (a ``-m`` entry point executes as ``__main__``)."""
    spec = getattr(sys.modules.get(function.__module__), "__spec__", None)
    return spec.name if spec is not None else function.__module__


def role_constant_audit():
    """Per (module, name): the value the retained module reads equals the v19 value (functions: same object)."""
    contracts = dict(v3=V3, v4=V4)
    out = {}
    for module, bound, names in ROLE_CONSTANTS:
        for name in names:
            seen, mine = getattr(contracts[bound], name), globals()[name]
            out[f"{module}[C={bound}].{name}"] = (seen is mine) if callable(mine) else (seen == mine)
    from . import spatial_controller as GC
    for name in CONTROLLER_CONSTANTS:
        out[f"{CONTROLLER_MODULE}.{name}"] = getattr(GC.C, name) == globals()[name]
    out[f"{CONTROLLER_MODULE}.contract"] = GC.C.__name__ == __name__
    return out


def token(arm, budget=FRAME_CEILING, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    return f"M3_SPATIAL_ROUTE_B_V19_{arm.upper()}_{producer.upper()}_C{int(budget):04d}_ONCE_EXECUTE_V1"


def accounting_registration():
    """The V16 storage-accounting registration, unchanged (the containment worker installs V16's session + timer)."""
    return accounting_registration_v16()


# The reviewed containment registration as DATA (``splitfusion_gt_geometry_containment_v1.contract.containment_registration``
# at MANIFEST_02).  The slim edge container re-derives the registration through ``gc_registration`` and has no GT
# producer dependencies (scipy), so nothing here imports the containment package; equality with the reviewed
# package's own ``gc_registration`` is asserted host-side (binding regression, unit tests).
CONTAINMENT_REGISTRATION = {
    "base_stack": "rl_agent.splitfusion_m3_route_b_v16.worker_entry",
    "binding_record": "gt_gc_trace/geometry_containment_binding.json",
    "diagnostics": "gt_gc_trace/geometry_refusal_v1",
    "engine": "DeferredProducerEngine adopted as ContainedEngine (prep worker only)",
    "plan_validation": "identical to a normal frame's bind (owner bind_plan checks, identity/anchor, payload-independent admission context and live-context/source-camera pose)",
    "prep_core": "PrepCoreLifetime subclass PrepCoreContained (prep worker only)",
    "reasons": {
        "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_POSSIBLE_COMPETITOR": "GT_GEOMETRY_ENVELOPE_POSSIBLE_COMPETITOR",
        "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET": "GT_GEOMETRY_ENVELOPE_TARGET"
    },
    "statuses": [
        "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_POSSIBLE_COMPETITOR",
        "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET"
    ],
    "transport": "existing BIND refusal: prep Refused('DEFERRAL_REFUSED_' + reason) -> service outcome 'NO_GT_DEFERRAL_REFUSED_' + reason -> edge GROUND_TRUTH_UNAVAILABLE",
    "version": "splitfusion.gt_geometry_containment.v1",
}


def gc_registration(arm, budget, producer="v2"):
    """HANDOFF_02: the reviewed containment registration -- V16's (incl. the unchanged ``storage_accounting``) with
    the containment trampolines/implementations and the added ``geometry_containment`` -- and only the seam naming
    this package's native seam (its ``seam_parameters`` checks THIS registration id).  Built from V16's contract and
    the data above; never imports the containment package (edge-safe)."""
    from rl_agent.splitfusion_m3_route_b_v16 import contract as V16
    value = V16.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(value["implementations"], service=SERVICE_ENTRY, base_worker=V16_WORKER_ENTRY)
    value["geometry_containment"] = json.loads(json.dumps(CONTAINMENT_REGISTRATION))
    value["seam"] = GC_SEAM
    return value


def accounting_registration_v16():
    from rl_agent.splitfusion_m3_route_b_v16 import contract as V16
    return V16.accounting_registration()


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
    result = dict(V17.own_sources(root))
    approved_design(root)
    result.update({name: sha(root / name) for name in APPROVED_FILES})
    for package in (V7.PACKAGE, V8.PACKAGE, V11.PACKAGE, V16_PACKAGE, PACKAGE, OFFLINE_ASSESSMENT_PACKAGE):
        result.update({str(p.relative_to(root)): sha(p) for p in sorted((root / package).glob("*.py"))})
    containment = {str(p.relative_to(root)): sha(p) for p in sorted((root / CONTAINMENT_PACKAGE).glob("*.py"))}
    if containment != CONTAINMENT_QUALIFIED:
        raise ValueError("GT containment sources are not the reviewed ones")
    result.update(containment)
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
            raise ValueError("missing real M3 v19 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M3 v19 source drift: " + relative)
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
                   controller=CONTROLLER_MODULE, collector_layer=ROUTE_DRAIN_MODULE),
        edge=dict(module=EDGE, entries=ENTRIES, validate_edge_startup=ENTRIES, registration=ENTRIES,
                  registration_path=EDGE_REGISTRATION_PATH),
        gt_service=dict(seam=GC_SEAM, trampoline=SERVICE_ENTRY, implementation=SERVICE_ENTRY,
                        trampoline_contract=SERVICE_CONTRACT_MODULE, retained_service=SERVICE_MODULE,
                        registration_check="rl_agent.splitfusion_exact_gt_live_v2.service.verify_sources"),
        gt_workers=dict(trampoline=WORKER_ENTRY, v8_trampoline=V8_WORKER_ENTRY, v7_trampoline=V7_WORKER_ENTRY,
                        retained_trampoline=RETAINED_WORKER_ENTRY, base_stack=V16_WORKER_ENTRY,
                        implementation="rl_agent.splitfusion_m3_route_b_v3.worker_entry",
                        prep_core=CONTAINMENT_PREP_CORE, prep_core_base=PREP_LIFETIME_MODULE + ".PrepCoreLifetime",
                        prep_engine=CONTAINMENT_ENGINE, prep_containment_version=CONTAINMENT_VERSION,
                        prep_accounting_timer=ACCOUNTING_TIMING_MODULE + ".AccountingTimer",
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
