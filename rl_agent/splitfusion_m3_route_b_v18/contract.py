"""M3.7 v18 prospective identity (12-vs-14 dB repeatability campaign, 14-dB sessions): the V16 runtime with ONE
registered runtime change, the reviewed GT geometry-refusal containment in the GT prep worker.

Relative to V16 (M3.6, constant OPERATING_14DB, anchor 06) the channel schedule, actor, route, traffic, radio, input
processing, deadline, accounting V3.v2 (+ V16 timer), tilt-aware AVO, FREEZE lifetime repair, P3, GC policy, map
publication, drain and retention are unchanged.  The only runtime change is the GT worker composition
(``splitfusion_gt_geometry_containment_v1``, reviewed: CORRECTION_01 / HANDOFF_02, sources pinned in
``CONTAINMENT_QUALIFIED``): the registered GT service/worker trampolines are that package's own reviewed
``service_entry`` / ``worker_entry`` (prep: V16 accounting -> V8 tilt AVO -> V7 FREEZE lifetime -> containment -> V4/V3),
which convert ONLY the two qualified pedestrian ground-rule-envelope refusals into an explicit per-frame
GT-unavailable outcome (NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_*); every other error stays fatal.  This v18
package (12 dB) is the v17 package with only the V16->V15 channel-schedule patch applied.

Historical derivation text (V16, unchanged below):

Relative to V11 (``M3_1_ROUTE_B_GOOD_ALL_V11``, the completed 2026-10-09 capture, registration content
``56bd0bc1...``) only the commanded channel schedule changes, from the constant GOOD_ALL 23.5 dB (anchor 01) to the
constant OPERATING_12DB 12.0 dB (anchor 07), together with the identity/binding fields that must name this revision.
This is the same single-field derivation V12 (BAD_ALL 8.5 dB, anchor 09) used; V12 and its evidence are unchanged.
Everything else -- actor, reward, 12 modes, q support, 22 causal features and their normalisation, route, density,
seeds, permitted interventions, 170-ms inclusive deadline, corrected prep-only V3.v2 storage accounting + per-call
timer + cache bounds, GT definitions, tilt-aware AVO, registered P3, GC policy, input processing, map
publication/freshness, retention, capacity C1-C5, radio telemetry, real-ACK end-of-route drain, abort exporter,
100 MHz / 273 PRB / 4D5U radio profile, scheduler, power control, MCS policy, CPU partition and the per-role launcher
binding pattern -- is the V11 value, re-declared here under the V18 identity.

1. OPERATING_12DB = constant 12.0 dB nominal -> RFsim noise_power_dB -6.0, the existing Phase-14A 100-MHz anchor 07
   of the pinned mapping (``target_to_rfsim_mapping.csv``) whose calibration record (``anchor_summary.csv`` row 07:
   per-transmission PUSCH median 12.0 / P05 11.5 / P95 12.0 dB, selected/final MCS median 15) is hash-bound below
   and re-read at registration.  The calibration is reused unchanged: no sweep, no outcome-driven adjustment, MCS
   NOT forced.  OPERATING_12DB is a new deterministic constant schedule, reported under its own name and kept
   separate from the historical stochastic profiles.  Its state index 1 (INTERMEDIATE) is the design state whose
   mean is nearest 12.0 dB (8.35 / 15.0 / 21.65); the index labels the trace rows and prefix digest only -- the live
   loop discards it and the actor publication carries no state.  The registered schedule, its digest and the causal
   commanded-target actor input change together (native v18 controller module reading this contract); the actor
   sees (12.0 - 5.5) / 19 = 0.342105..., inside its unchanged [5.5, 24.5] dB support.  No gNB measurement reaches
   the actor; normalisation, weights, power control, MCS policy and support are unchanged.
2. Channel fidelity (evidence only, registered BEFORE launch, never a gate on the integration verdict and never a
   trigger for a retry): see ``CHANNEL_FIDELITY``.

Everything else is v11 (and through it v10/v8/v7/v4/v3).
"""
import json
from pathlib import Path
import sys

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_m3_route_b_v8 import contract as V8
from rl_agent.splitfusion_m3_route_b_v11 import contract as V11
from rl_agent.splitfusion_m3_route_b_v8.contract import *  # noqa: F401,F403

PACKAGE = "rl_agent/splitfusion_m3_route_b_v18"
MODULE = "rl_agent.splitfusion_m3_route_b_v18"
LAUNCHER_REVISION = "M3_7_ROUTE_B_STARTUP_BINDING_V18"
ID = "M3_7_ROUTE_B_OPERATING_12DB_V18"
PREVIOUS_ID = V11.ID
RUNTIME_REVISION = ID
SCIENTIFIC_RUNTIME_BASIS = V4.ID

CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v18"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v18"
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
SHIMS = {f"{PACKAGE}/runtime_shims/child_v1_m3_route_b_v18.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_m3_route_b_v18.py",
         f"{PACKAGE}/runtime_shims/edge_profiled_v1_m3_route_b_v18.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_m3_route_b_v18.py"}

# ---- 1. corrected exact GT storage accounting V3.v2 (prep only), at the reviewed final digests
#         (experiments/splitfusion_gt_storage_accounting_v3/SOURCES_SHA256_02.txt and EVIDENCE_SHA256_02.txt)
ACCOUNTING_PACKAGE = "rl_agent/splitfusion_gt_storage_accounting_v3"
ACCOUNTING_MODULE = "rl_agent.splitfusion_gt_storage_accounting_v3"
ACCOUNTING_QUALIFIED = {
    f"{ACCOUNTING_PACKAGE}/__init__.py": "871029a0f75c9115b42642403f2d23eba48a58f149d25b0e0e0b3407177f5391",
    f"{ACCOUNTING_PACKAGE}/accounting.py": "b6f4a655b65f98ac10d31227372eba78c60672fba7c25a55debe03199f391baa",
    f"{ACCOUNTING_PACKAGE}/switch.py": "3510b0f118eab1ea6ac2b94db9b9cd0cf18ce40d3d7ab17836c4ec76890eeb79"}
# V3 imports V1 for the unchanged scalar/logical-byte definition. V2 stays registered as the comparison package
# used by the reviewed qualification harnesses, but is never installed by the V11/V18 production worker.
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

# ---- 1. OPERATING_12DB constant 12.0 dB (replaces the v11 23.5-dB GOOD_ALL schedule; carrier cell unchanged)
CONTROLLER_MODULE = MODULE + ".operating_12db_controller"
V11_SCHEDULE = dict(V11.SCHEDULE)
ANCHOR_EVIDENCE = dict(
    path="experiments/splitfusion_phase14a_100mhz_calibration_v1/20260904_phase14a_live_calibration_retry3/"
         "anchor_summary.csv",
    sha256="279027290439a2fce3600675b0057df6bcc36a65cf4ae8400cdb4ed148a0829b",
    anchor_index=7, applied_noise_power_db=-6.0, achieved_median_pusch_snr_db=12.0, achieved_pusch_snr_p05_db=11.5,
    achieved_pusch_snr_p95_db=12.0, uncertainty_half_width_db=0.25, selected_mcs_median=15.0, final_mcs_median=15.0,
    pusch_sample_count=649, mcs_sample_count=649, status="ANCHOR_ACCEPTED")
SCHEDULE = dict(V3.SCHEDULE, schedule_id="OPERATING_12DB", trace_id="CONST_OPERATING_12P0DB_V1", target_snr_db=12.0,
                state_index=1, expected_rfsim_command_db=-6.0, mapping_point="PHASE14A_100MHZ_ANCHOR_07",
                qualification="experiments/splitfusion_phase14a_100mhz_calibration_v1/20260904_phase14a_live_"
                              "calibration_retry3/anchor_summary.csv anchor 07 (historical calibration, reused "
                              "unchanged): per-transmission PUSCH snrx10 median 12.0 / P05 11.5 / P95 12.0 dB, "
                              "selected/final MCS median 15 (radio only, 5 s); MCS is not forced",
                not_the_stochastic_profile="ADVERSE_STABLE and MID_VARIABLE are Gaussian-Markov traces (design state "
                                           "means 8.35 / 15.0 dB); OPERATING_12DB is a new deterministic constant "
                                           "schedule and is reported under its own name; state index 1 "
                                           "(INTERMEDIATE, nearest design mean) labels trace rows only",
                actor_observation="the retained causal SNR proxy is the ACKed COMMANDED TARGET (controller publication), "
                                  "constant 12.0 dB -> feature 22 = (12.0-5.5)/19 = 0.342105...; it is not a measured "
                                  "achieved SNR; schedule identity/step/route progress never enter the actor state")
SCHEDULE_FIELDS_CHANGED = ("schedule_id", "trace_id", "target_snr_db", "state_index", "expected_rfsim_command_db",
                           "mapping_point", "qualification", "not_the_stochastic_profile", "actor_observation")
ACTOR_SNR_FEATURE = (12.0 - 5.5) / 19.0
LATER_SCHEDULES_NOT_IMPLEMENTED = ("GOOD_TO_BAD", "BAD_TO_GOOD")
DESIGN_STATE_ORDER = ("ADVERSE", "INTERMEDIATE", "FAVORABLE")     # network_profile_design_v2 target_snr.state_order
DESIGN_STATE = "INTERMEDIATE"

# ---- 2. channel fidelity, registered before launch (evidence only; the integration verdict never reads it)
CHANNEL_FIDELITY = dict(
    window="route-only: first through last decision capture wall time (the V11 analysis definition); startup "
           "attach (clean -50 dB) and restoration are excluded",
    measured="calibrated per-transmission PUSCH snrx10/10 from m3_radio/ttracer/live_pusch_with_ingest.csv",
    checks=dict(
        F1_pusch_median_db=dict(rule="route-window median within anchor median +/- anchor half-width",
                                low=11.75, high=12.25),
        F2_pusch_p05_db=dict(rule="route-window P05 >= anchor P05 - anchor half-width", low=11.25),
        F3_pusch_p95_db=dict(rule="route-window P95 <= anchor P95 + anchor half-width", high=12.25),
        F4_commanded=dict(rule="every controller row commands target 12.0 dB -> RFsim -6.0 (also an integration check)"),
        F5_selected_mcs_median=dict(rule="route-window median selected MCS equals the anchor's 15 (an OUTCOME of the "
                                         "unchanged SINR MCS policy, never forced)", expected=15)),
    reported_alongside=["scheduler avg_snr (EMA) P50/P95", "selected/final MCS histogram", "granted PRBs split by "
                        "gNB-reported buffer", "independent pre-action RLC backlog", "ULSCH rounds / retransmissions "
                        "/ errors / DTX deltas", "MAC-log power-control SNR, separately labelled, never mixed"],
    on_mismatch="reported as CHANNEL_FIDELITY_MISMATCH with the measured values; no retry, no adjustment, the "
                "integration verdict is unaffected and the capture is still analysed under its registered name")


def schedule_prefix(count):
    """The OPERATING_12DB prefix in the retained (state, target) form, and its digest computed like prepare_sequence."""
    import hashlib
    import numpy as np
    prefix = [(int(SCHEDULE["state_index"]), float(SCHEDULE["target_snr_db"]))] * int(count)
    states = np.asarray([row[0] for row in prefix], dtype="<i4")
    targets = np.asarray([row[1] for row in prefix], dtype="<f8")
    return prefix, hashlib.sha256(states.tobytes() + targets.tobytes()).hexdigest()


def channel_registration(root):
    """v11 ``channel_registration`` reading the v18 SCHEDULE (same pinned config/design/traces/mapping checks), plus
    the hash-bound anchor-07 calibration row, the design's state order and the registered fidelity checks."""
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
    anchor = [row for row in points if float(next(iter(row.values()))) == SCHEDULE["target_snr_db"]]
    if (len(anchor) != 1 or float(list(anchor[0].values())[1]) != SCHEDULE["expected_rfsim_command_db"]
            or list(anchor[0].values())[2] != SCHEDULE["mapping_point"]):
        raise ValueError("OPERATING_12DB target is not the registered measured mapping point")
    from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as R5
    if R5.scale_snr_db(SCHEDULE["target_snr_db"]) != ACTOR_SNR_FEATURE:
        raise ValueError("OPERATING_12DB target is outside the actor's registered SNR support or scaling drifted")
    design = json.loads((root / network["design_config"]).read_bytes())
    if tuple(design["target_snr"]["state_order"]) != DESIGN_STATE_ORDER or \
            DESIGN_STATE_ORDER[SCHEDULE["state_index"]] != DESIGN_STATE:
        raise ValueError("OPERATING_12DB state index is not the design's INTERMEDIATE state")
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
            or measured["applied_noise_power_db"] != SCHEDULE["expected_rfsim_command_db"] \
            or measured["achieved_median_pusch_snr_db"] != SCHEDULE["target_snr_db"]:
        raise ValueError("anchor 07 calibration row differs from the registered OPERATING_12DB anchor evidence")
    count = int(network["prefix_samples"])
    _, digest = schedule_prefix(count)
    return dict(schedule=dict(SCHEDULE, prefix_samples=count, trace_sha256=digest,
                              continuation="CONSTANT (same value every 100-ms step after the prefix)"),
                carrier_cell=dict(action=CARRIER_ACTION, network_profile=ORIGINAL_PROFILE, trace=dict(CARRIER_TRACE),
                                  profile_substituted=False),
                controller_module=CONTROLLER_MODULE, retained_controller_module=RETAINED_CONTROLLER_MODULE,
                seam="UE child sets splitfusion_run5_full_live_v1.runtime_policy.CONTROLLER_MODULE; the controller "
                     "replaces only ue_target_snr_cell_runtime_v1.prepare_sequence and rewrites --profile-id to OPERATING_12DB",
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
                                 feature_index=22, value=ACTOR_SNR_FEATURE, support_db=[5.5, 24.5],
                                 normalisation_changed=False, gnb_measurement_to_actor=False),
                anchor_evidence=dict(ANCHOR_EVIDENCE), design_state_order=list(DESIGN_STATE_ORDER),
                fidelity=CHANNEL_FIDELITY,
                changed_from_v11=dict(fields=list(SCHEDULE_FIELDS_CHANGED),
                                      v11={key: V11_SCHEDULE[key] for key in SCHEDULE_FIELDS_CHANGED},
                                      v11_actor_feature=V11.ACTOR_SNR_FEATURE,
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
OFFLINE_ASSESSMENT_PACKAGE = "rl_agent/splitfusion_m3_route_b_v18_offline_v1"
OFFLINE_ASSESSMENT_MODULE = "rl_agent.splitfusion_m3_route_b_v18_offline_v1.postflight"
OFFLINE_ASSESSMENT_BINDING = "rl_agent.splitfusion_m3_route_b_v18_offline_v1.binding"
OFFLINE_ASSESSMENT_INTEGRATION = "rl_agent.splitfusion_m3_route_b_v18_offline_v1.integration"
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
# Neither the v3 nor the v11 GOOD_ALL controller is executed in a v18 run; the v18 OPERATING_12DB controller is, and it reads
# this contract.  The controller installer is likewise the native v18 one.
CONTROLLER_CONSTANTS = ("ORIGINAL_PROFILE", "SCHEDULE", "CONTROLLER_MODULE", "RETAINED_CONTROLLER_MODULE")
V3_ENTRY_HELPERS = ("install_ue_capacity", "install_map_trampoline", "_write")


def repo_root():
    return Path(__file__).resolve().parents[2]


def module_identity(function):
    """Import name of the module that defines ``function`` (a ``-m`` entry point executes as ``__main__``)."""
    spec = getattr(sys.modules.get(function.__module__), "__spec__", None)
    return spec.name if spec is not None else function.__module__


def role_constant_audit():
    """Per (module, name): the value the retained module reads equals the v18 value (functions: same object)."""
    contracts = dict(v3=V3, v4=V4)
    out = {}
    for module, bound, names in ROLE_CONSTANTS:
        for name in names:
            seen, mine = getattr(contracts[bound], name), globals()[name]
            out[f"{module}[C={bound}].{name}"] = (seen is mine) if callable(mine) else (seen == mine)
    from . import operating_12db_controller as GC
    for name in CONTROLLER_CONSTANTS:
        out[f"{CONTROLLER_MODULE}.{name}"] = getattr(GC.C, name) == globals()[name]
    out[f"{CONTROLLER_MODULE}.contract"] = GC.C.__name__ == __name__
    return out


def token(arm, budget=FRAME_CEILING, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    return f"M3_7_ROUTE_B_OPERATING_12DB_V18_{arm.upper()}_{producer.upper()}_C{int(budget):04d}_ONCE_EXECUTE_V1"


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
    result = dict(V4.own_sources(root))
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
            raise ValueError("missing real M3 v18 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M3 v18 source drift: " + relative)
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
