"""Static-mode v13 prospective identity: the qualified V11/V12 runtime with registered changes for ONE campaign of
13 arms (the deployed actor + 12 actor-free external fixed actions, mode 0..11 at q = 0), each 1,000 transmitted
frames on FAVORABLE_STABLE.

Relative to V12 (``M3_2_ROUTE_B_BAD_ALL_V12``, the completed 2026-10-09 capture) the registered changes are:

1. Channel: FAVORABLE_STABLE, the pinned stochastic Gauss-Markov trace ``GM_V2_FAVORABLE_STABLE_SEED_2026082101``
   executed by the RETAINED Run-5 controller (``splitfusion_run5_live_avo_v1.snr_live``), i.e. NO controller
   substitution: the carrier cell is commanded as itself.  Every arm starts a fresh controller process, so the
   identical seeded sequence restarts at step 0 (prefix 4,200 x 100 ms, digest bound below; continuation = same RNG).
   The actor (agent arm) sees the ACKed commanded target, as in V11/V12; no gNB measurement reaches the actor.
2. Budget: 1,000 transmitted frames per arm (``transmitted_budget`` = ``g1_boundary_budget`` = 1,000,
   ``exact_gt_limits.max_frames`` = 1,000, GC trace bounds derived from it as in every earlier revision).  Capacity
   C1-C5, archive cap, route registration and lifecycle bounds are UNCHANGED (sized for the 3,900-frame ceiling,
   which a 1,000-frame arm never approaches).
3. Terminal: the retained budget stop rule of ``phase6_live_child_v2.run`` (TRANSMITTED_BUDGET_REACHED at exactly the
   budget, or a closed DECISION_CYCLE_BOUNDARY within it) is kept -- the V3 route-completion substitution R1 is NOT
   installed -- while the V3 ``FinishResult`` repair (cause of the M1 post-budget TypeError) and the V12 bounded
   end-of-run drain stay installed.  The runner's G1 uses its retained budget branch (the V3 ROUTE_COMPLETED branch
   wrapper is not installed); the V3 P3 final-cycle exception is kept unchanged.
4. Paired-input retention: DISABLED uniformly (no observer, no plan file).  GT scoring, the numeric GT archive, exact
   rescore evidence, actor replay evidence, map journal, radio telemetry and the drain record are unchanged.
5. Selection: arm ``agent`` = the unchanged deployed actor; arms ``mode00``..``mode11`` = an external fixed action
   (actor never called; typed state refusals logged and the fixed action continues; no fallback substitution, no
   clamp to the actor's support) whose exact 9-field identity is frozen here and re-resolved at registration.

Everything else -- actor/reward/12 modes/q support/22 features, route, density, seeds, interventions, 170-ms
inclusive deadline, V3.v2 storage accounting + timer + bounds, GT definitions, tilt-aware AVO, P3, GC policy,
input processing, map publication, radio telemetry, abort exporter, radio profile, scheduler, power control, MCS
policy, CPU partition and the per-role launcher binding pattern -- is the V12 value re-declared under this identity.
"""
from pathlib import Path
import sys

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_m3_route_b_v8 import contract as V8
from rl_agent.splitfusion_m3_route_b_v11 import contract as V11
from rl_agent.splitfusion_m3_route_b_v12 import contract as V12
from rl_agent.splitfusion_m3_route_b_v12.contract import *  # noqa: F401,F403  (V12 values; overrides follow)

PACKAGE = "rl_agent/splitfusion_static_modes_v13"
MODULE = "rl_agent.splitfusion_static_modes_v13"
LAUNCHER_REVISION = "M1X_STATIC_MODES_STARTUP_BINDING_V13"
ID = "M1X_STATIC_MODES_FAVORABLE_V13"
PREVIOUS_ID = V12.ID
RUNTIME_REVISION = ID
SCIENTIFIC_RUNTIME_BASIS = V4.ID

CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_static_modes_v13"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_static_modes_v13"
RUNNER = MODULE + ".runner"
PARTITION = MODULE + ".partition"
HOST = MODULE + ".host"
ENTRIES = MODULE + ".entries"
GC_SEAM = MODULE + ".gc_seams"
SERVICE_ENTRY = MODULE + ".service_entry"
WORKER_ENTRY = MODULE + ".worker_entry"
FIXED_ENGINE = MODULE + ".fixed_engine"
EDGE_REGISTRATION_PATH = "/partition_output/registration.json"
VERIFY_FLAG = "--m3-verify-startup"
SHIMS = {f"{PACKAGE}/runtime_shims/child_v1_static_modes_v13.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_static_modes_v13.py",
         f"{PACKAGE}/runtime_shims/edge_profiled_v1_static_modes_v13.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_static_modes_v13.py"}
RETENTION = None                                    # paired retention disabled: no loader module exists in v13

# ---- 2. budget (capacity, archive cap, route registration and lifecycle unchanged)
ARM_BUDGET = 1000
BUDGETS = (ARM_BUDGET,)
LIFECYCLE = dict(safety_timeout_s=SAFETY_TIMEOUT_S, child_timeout_s=CHILD_TIMEOUT_S)   # = V12 registered lifecycle


def limits(budget):
    """Retained exact-GT limits with max_frames = the registered arm budget (= M1's registered limits)."""
    if int(budget) not in BUDGETS:
        raise ValueError("budget is not the registered v13 arm budget")
    return dict(V3.E.LIMITS, max_frames=int(budget))


# V12 constant-schedule names that must not be reachable from this contract.
del ANCHOR_EVIDENCE, ACTOR_SNR_FEATURE, SCHEDULE_FIELDS_CHANGED, schedule_prefix, LATER_SCHEDULES_NOT_IMPLEMENTED
del DESIGN_STATE_ORDER, V11_SCHEDULE

# ---- 1. channel: FAVORABLE_STABLE through the retained controller (no substitution)
CONTROLLER_MODULE = V3.RETAINED_CONTROLLER_MODULE
V12_SCHEDULE = dict(V12.SCHEDULE)
SCHEDULE = dict(schedule_id="FAVORABLE_STABLE", trace_id=V3.CARRIER_TRACE["trace_id"], seed=V3.CARRIER_TRACE["seed"],
                kind="STOCHASTIC_GAUSS_MARKOV_3_STATE_PINNED_SEED",
                generator="ue_target_snr_cell_runtime_v1.prepare_sequence -> DeterministicTargetSnrSequence (retained)",
                restart="every arm starts a fresh controller process: the identical seeded sequence restarts at step 0",
                actor_observation="the retained causal SNR proxy is the ACKed COMMANDED TARGET of this trace (time-"
                                  "varying, inside the actor's [5.5, 24.5] dB support); it is not a measured achieved "
                                  "SNR and no gNB measurement reaches the actor")
FAVORABLE_PREFIX_SAMPLES = 4200
FAVORABLE_TRACE_SHA256 = "8cad86e2e1ee98176c75b872132eb68f7d770f889d7330fa4a9932c45c172de6"
FAVORABLE_STEP0 = (2, 21.521707697746844)
HISTORICAL_SAME_CHANNEL = ("20261007 M1 agent/static/fixed captures (radio_trace.csv step 0 = 21.5217 dB, all "
                           "ACK_ON_TIME); M3 GOOD_ALL (constant 23.5 dB) and BAD_ALL (8.5 dB) are different channels")


def channel_registration(root):
    """FAVORABLE_STABLE, bound to the pinned config/design/traces/mapping AND to the retained generator's own output."""
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
    if int(network["prefix_samples"]) != FAVORABLE_PREFIX_SAMPLES:
        raise ValueError("FAVORABLE_STABLE prefix length differs from the registered value")
    return dict(schedule=dict(SCHEDULE, prefix_samples=FAVORABLE_PREFIX_SAMPLES, trace_sha256=FAVORABLE_TRACE_SHA256,
                              step0=list(FAVORABLE_STEP0), continuation=network["continuation"]),
                carrier_cell=dict(action=CARRIER_ACTION, network_profile=ORIGINAL_PROFILE, trace=dict(CARRIER_TRACE),
                                  profile_substituted=False),
                controller_module=CONTROLLER_MODULE, retained_controller_module=RETAINED_CONTROLLER_MODULE,
                seam="none: runtime_policy.CONTROLLER_MODULE is verified equal to the retained module and NOT replaced",
                live_config=LIVE_CONFIG, live_config_sha256=sha(root / LIVE_CONFIG),
                design_config=network["design_config"], design_config_sha256=network["design_config_sha256"],
                traces_csv=network["traces_csv"], traces_sha256=network["traces_sha256"],
                mapping_csv=network["mapping_csv"], mapping_sha256=network["mapping_sha256"],
                sample_period_ms=int(network["sample_period_ms"]), catch_up_policy=network["catch_up_policy"],
                radio_profile=network["radio_baseline"]["profile_id"],
                restore="retained runtime: noise_power_dB -50 commanded, read back and verified at teardown",
                transitions_depend_on_actor=False,
                actor_input=dict(source="ACKed commanded target (retained controller publication)",
                                 feature_index=22, value="time-varying (FAVORABLE_STABLE trace)", support_db=[5.5, 24.5],
                                 normalisation_changed=False, gnb_measurement_to_actor=False),
                historical_same_channel=HISTORICAL_SAME_CHANNEL,
                changed_from_v12=dict(v12_schedule={k: V12_SCHEDULE[k] for k in ("schedule_id", "trace_id",
                                                                                 "target_snr_db")},
                                      controller_substitution_removed=True, power_control_changed=False,
                                      mcs_policy_changed=False, mcs_forced=False, calibration_resweep=False))


def favorable_prefix_check(root):
    """Re-run the retained generator for FAVORABLE_STABLE and require the registered digest and step 0 (offline)."""
    import json
    from rl_agent import ue_target_snr_cell_runtime_v1 as base
    config = json.loads((Path(root) / LIVE_CONFIG).read_bytes())
    _sequence, prefix, frozen = base.prepare_sequence(campaign=config, profile_id=ORIGINAL_PROFILE)
    ok = (frozen.get("trace_sha256") == FAVORABLE_TRACE_SHA256 and len(prefix) == FAVORABLE_PREFIX_SAMPLES
          and tuple(prefix[0]) == FAVORABLE_STEP0 and frozen.get("trace_id") == CARRIER_TRACE["trace_id"])
    return dict(ok=ok, frozen=frozen, prefix_samples=len(prefix), step0=list(prefix[0]))


CHANNEL_FIDELITY = dict(
    window="route-only: first through last decision capture wall time",
    measured="calibrated per-transmission PUSCH snrx10/10 from m3_radio/ttracer/live_pusch_with_ingest.csv",
    commanded="radio_trace.csv: every row ACK_ON_TIME and every commanded target equal to the registered trace prefix "
              "at its step index (integration evidence)",
    reported=["commanded vs measured PUSCH time series per arm", "selected/final MCS", "RLC backlog",
              "MAC-log power-control SNR, separately labelled, never mixed"],
    on_mismatch="reported; no retry, no adjustment; the integration verdict is unaffected")

# ---- 4. retention disabled
PAIRED_RETENTION = dict(enabled=False, observer=None, plan_file=None,
                        reason="registered off uniformly for the static-mode comparison (Abiodun, 2026-10-09)",
                        unaffected=["GT scoring and the numeric GT archive", "exact rescore evidence",
                                    "actor replay evidence (agent arm)", "map journal", "radio telemetry"],
                        resource_rule_avoided="the v3/v4 per-sample wire/host caps no longer bind any arm")


def retention_registration(plan_sha256=None):
    if plan_sha256 is not None:
        raise ValueError("v13 registers paired retention as disabled; no plan digest is admissible")
    return dict(PAIRED_RETENTION)


# ---- 5. arms
AGENT_ARM = "agent"
MODE_ARMS = tuple(f"mode{m:02d}" for m in range(12))
ARM_KEYS = (AGENT_ARM,) + MODE_ARMS
RUN_ORDER = ("agent", "mode08", "mode11", "mode03", "mode02", "mode01", "mode10", "mode07", "mode00", "mode06",
             "mode04", "mode09", "mode05")
RUN_ORDER_RULE = "agent first, then numpy.default_rng(20261010).permutation(12); frozen 2026-10-09 before any result"
# Frozen from experiments/splitfusion_static_modes_v13/identities_01 (resolve_q_e4 at q_e4=0, real codec exercise).
FIXED_IDENTITIES = {
    0: ("noAE", "UINT8", 0, "split_noae_uint8_q0000", "583cbab6b68449495e779259507fdf7dcdfba56b5d31894ab9f8955da4deb10a"),
    1: ("noAE", "UINT6", 6, "split_noae_uint6_q0000", None),
    2: ("noAE", "UINT4", 12, "split_noae_uint4_q0000", None),
    3: ("AE128", "UINT8", 18, "split_ae128_uint8_q0000", None),
    4: ("AE128", "UINT6", 24, "split_ae128_uint6_q0000", None),
    5: ("AE128", "UINT4", 30, "split_ae128_uint4_q0000", None),
    6: ("AE64", "UINT8", 36, "split_ae64_uint8_q0000", None),
    7: ("AE64", "UINT6", 42, "split_ae64_uint6_q0000", None),
    8: ("AE64", "UINT4", 48, "split_ae64_uint4_q0000", None),
    9: ("AE32", "UINT8", 54, "split_ae32_uint8_q0000", "dbe6a1e5ee1b0c0e2ab8617a35ad6388f2325aff3b1761c1cf958a8a1ee976cc"),
    10: ("AE32", "UINT6", 60, "split_ae32_uint6_q0000", None),
    11: ("AE32", "UINT4", 66, "split_ae32_uint4_q0000", None)}
FIXED_BUNDLES_FILE = f"{PACKAGE}/FIXED_IDENTITIES.json"    # all 12 bundle digests, written by identities_01
SUPPORT_Q_E4_MIN = {0: 8812, 1: 8493, 2: 7491, 3: 8280, 4: 7827, 5: 5902, 6: 6603, 7: 5682, 8: 1946, 9: 3110,
                    10: 1217, 11: 0}                     # modeled_smoke_support.py:45-58 (actor support, informational)


def _bundles(root):
    import json
    path = Path(root) / FIXED_BUNDLES_FILE
    value = json.loads(path.read_bytes())
    if value.get("schema") != "splitfusion.static_modes_v13.fixed_identities.v1" or set(value["modes"]) != {
            str(m) for m in range(12)}:
        raise ValueError("frozen fixed-identity table is foreign or incomplete")
    return {int(k): v for k, v in value["modes"].items()}


def fixed_action(arm, root=None):
    """The frozen 11-field identity of a mode arm (None for the agent arm)."""
    if arm == AGENT_ARM:
        return None
    if arm not in MODE_ARMS:
        raise ValueError("unknown arm: " + repr(arm))
    mode = int(arm[4:])
    family, quantizer, action_id, profile_id, recorded = FIXED_IDENTITIES[mode]
    bundle = _bundles(root or repo_root())[mode]
    if recorded is not None and bundle != recorded:
        raise ValueError("frozen bundle differs from the previously recorded M1 bundle for mode %d" % mode)
    return dict(mode_id=mode, q_e4=0, action_id=action_id, profile_id=profile_id, family=family, quantizer=quantizer,
                keep_count=21504, drop_count=0, measurement_status="MEASURED_ANCHOR",
                execution_bundle_sha256=bundle, ranker_bypassed=True,
                support=("INSIDE_ACTOR_SUPPORT_STILL_EXTERNAL" if SUPPORT_Q_E4_MIN[mode] == 0
                         else "OUTSIDE_ACTOR_SUPPORT_EXTERNAL_FIXED"))


def resolved_fixed_action(arm):
    """Re-resolve a mode arm with the live resolver; must equal the frozen identity exactly."""
    from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import load_dynamic_execution_contract
    frozen = fixed_action(arm)
    p = load_dynamic_execution_contract().resolve_q_e4(frozen["mode_id"], 0)
    return dict(mode_id=p.mode_id, q_e4=p.q_e4, action_id=p.action_id, profile_id=p.profile_id, family=p.family,
                quantizer=p.quantizer, keep_count=p.keep_count, drop_count=p.drop_count,
                measurement_status=p.measurement_status, execution_bundle_sha256=p.execution_bundle_sha256,
                ranker_bypassed=p.execution_bundle.ranker_bypassed, support=frozen["support"])


SELECTION_SEMANTICS_FIXED = dict(
    source="EXTERNAL_FIXED_ACTION", actor_proposal_required=False, actor_called=False,
    actor_state_admission_required=False, telemetry="OBSERVATIONAL",
    typed_admission_refusals="LOG_AND_CONTINUE_FIXED", fallback_substitution=False, clamp_to_actor_support=False,
    identity_check="every decision AND hold: 9-tuple (mode, q_e4, action_id, profile_id, bundle, keep, drop, family, "
                   "quantizer) equals the registered identity, else the child raises")
SELECTION_SEMANTICS_AGENT = dict(source="DEPLOYED_ACTOR", actor_called=True, fallback="mode11_q9800_a71 (retained)")
ZERO_COMPLETION = dict(
    status="VALID_OBSERVED_OUTCOME",
    rule="zero successful edge.compute completions with CPU materialization is a registered, valid overload outcome "
         "when every infrastructure, accounting, transport-limit, terminal and cleanup check passes; it is reported as "
         "a flat zero completion curve, never as a latency distribution",
    never=["TIMEOUT/STALE timestamps as completion times", "failed edge.compute spans as completions",
           "map publication as prediction completion", "feedback receipt as prediction completion"])


def token(arm, budget=ARM_BUDGET, producer="v2"):
    if arm not in ARM_KEYS or producer not in PRODUCERS or int(budget) not in BUDGETS:
        raise ValueError("unknown arm/producer/budget")
    return f"M1X_STATIC_MODES_V13_{arm.upper()}_{producer.upper()}_B{int(budget):04d}_ONCE_EXECUTE_V1"


# ---- 3. terminal rule
TERMINAL = dict(
    rule="RETAINED_BUDGET_STOP",
    child=("retained phase6_live_child_v2.run budget rule: TRANSMITTED_BUDGET_REACHED with sent == budget, or a closed "
           "DECISION_CYCLE_BOUNDARY with 1 <= sent <= budget; collector cleanup_ok and no collector failures"),
    route_completion_substitution_installed=False,
    finish_repair="splitfusion_m3_route_b_v3.route_completion.M3CollectorLayer (FinishResult) -- removes the M1 "
                  "post-budget TypeError at its cause",
    drain="this package's route_drain (copy of the V12 bounded drain) directly above the V3 layer: stop new "
          "submissions, wait for the in-flight token, wait for each obligation's own ACK horizon, close the rest "
          "through record_expired on the real clock",
    runner_g1="retained runner_v1 G1 budget branch (TRANSMITTED_BUDGET_REACHED / verified cycle boundary); the V3 "
              "ROUTE_COMPLETED branch wrapper is not installed",
    p3_final_cycle_exception="kept (V3, unchanged)",
    route_summary="reported; a budget stop is not a route completion and is never labelled as one")

# ---- identity, sources and binding
ROUTE_DRAIN_MODULE = MODULE + ".route_drain"
POSTFLIGHT_P3_MODULE = MODULE + ".postflight_p3"
RADIO_TELEMETRY_MODULE = MODULE + ".radio_telemetry"
OFFLINE_ASSESSMENT_PACKAGE = "rl_agent/splitfusion_static_modes_v13_offline_v1"
OFFLINE_ASSESSMENT_MODULE = "rl_agent.splitfusion_static_modes_v13_offline_v1.assessment"
OFFLINE_ASSESSMENT_BINDING = "rl_agent.splitfusion_static_modes_v13_offline_v1.binding"
OFFLINE_ASSESSMENT_INTEGRATION = "rl_agent.splitfusion_static_modes_v13_offline_v1.assessment"
CONTROLLER_CONSTANTS = ()


def radio_telemetry_registration(root):
    value = V12.radio_telemetry_registration(root)
    value["module"] = RADIO_TELEMETRY_MODULE
    return value


def assessment_registration():
    return dict(module=OFFLINE_ASSESSMENT_MODULE, binding=OFFLINE_ASSESSMENT_BINDING,
                package=OFFLINE_ASSESSMENT_PACKAGE,
                derives="arm identity, selection semantics, budget, channel, terminal rule and retention status from "
                        "the verified registration",
                classes=["VALID_MEASUREMENT (completion statistics, including zero)", "INFRASTRUCTURE_FAILURE",
                         "INVALID_IDENTITY"],
                completion="successful live edge.compute span (ok, reward_requested) with edge:prediction_materialized; "
                           "cross-checked one-to-one",
                numeric_host_exit_status="CAPTURED at launch; for actor-free arms a non-zero exit caused ONLY by "
                                         "actor-only gates is recorded, never re-labelled")


def bookkeeping_registration():
    from . import postflight_p3 as P3
    from . import route_drain as RD
    return dict(end_of_route_drain=RD.drain_registration(), offline_p3=P3.postflight_registration(),
                registered_assessment=assessment_registration(), terminal=dict(TERMINAL))


def repo_root():
    return Path(__file__).resolve().parents[2]


def module_identity(function):
    spec = getattr(sys.modules.get(function.__module__), "__spec__", None)
    return spec.name if spec is not None else function.__module__


def role_constant_audit():
    contracts = dict(v3=V3, v4=V4)
    out = {}
    for module, bound, names in ROLE_CONSTANTS:
        for name in names:
            seen, mine = getattr(contracts[bound], name), globals()[name]
            out[f"{module}[C={bound}].{name}"] = (seen is mine) if callable(mine) else (seen == mine)
    return out


def accounting_registration():
    """V12's registration with the timer module named as this package's (the copy the v13 worker installs)."""
    value = V12.accounting_registration()
    value["timing"] = dict(value["timing"], module=MODULE + ".accounting_timing")
    return value


def gc_registration(arm, budget, producer="v2"):
    value = V12.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(value["implementations"], service=SERVICE_ENTRY)
    value["storage_accounting"] = accounting_registration()
    value["seam"] = GC_SEAM
    return value


def own_sources(root):
    """v12 sources (which include v4/v7/v8/v11/v12 and the qualified repairs) plus every file of this package and
    of its offline assessment package and the frozen identity table."""
    root = Path(root).resolve(strict=True)
    result = dict(V12.own_sources(root))
    for package in (PACKAGE, OFFLINE_ASSESSMENT_PACKAGE):
        result.update({str(p.relative_to(root)): sha(p) for p in sorted((root / package).glob("*.py"))})
    result[FIXED_BUNDLES_FILE] = sha(root / FIXED_BUNDLES_FILE)
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
            raise ValueError("missing real v13 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("v13 source drift: " + relative)
    for label, qualified in (("prep lifetime repair", PREP_LIFETIME_QUALIFIED), ("tilt-aware AVO", AVO_TILT_QUALIFIED),
                             ("storage accounting", ACCOUNTING_QUALIFIED),
                             ("storage accounting v1 base", ACCOUNTING_V1_QUALIFIED),
                             ("storage accounting v2 comparison", ACCOUNTING_V2_QUALIFIED)):
        for relative, digest in qualified.items():
            if result[relative] != digest:
                raise ValueError(f"{label} differs from the qualified source: " + relative)
    return dict(sorted(result.items()))


def launcher_binding():
    return dict(
        launcher_revision=LAUNCHER_REVISION, runtime_revision=RUNTIME_REVISION,
        scientific_runtime_basis=SCIENTIFIC_RUNTIME_BASIS, contract_module=MODULE + ".contract",
        host=dict(verify_prospective=HOST, verify_prerequisite=HOST, check_prerequisites=HOST,
                  verify_startup_proof=HOST),
        runner=dict(module=RUNNER, validate_parent=RUNNER, verify_prospective=HOST, verify_prerequisite=HOST,
                    radio_telemetry=RADIO_TELEMETRY_MODULE, mac_log_observer=MAC_LOG_OBSERVER),
        child=dict(module=CHILD, entries=ENTRIES, validate_child_startup=ENTRIES, verify_prospective=HOST,
                   registration=ENTRIES, load_observer=None, seam_parameters=GC_SEAM, controller=CONTROLLER_MODULE,
                   collector_layer=ROUTE_DRAIN_MODULE, fixed_engine=FIXED_ENGINE),
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
                        assessment_binding=OFFLINE_ASSESSMENT_BINDING),
        partition=PARTITION)


def lineage(root, actor_boundary, support_sha256, budget, arm, producer="v2", gc_arm="candidate"):
    if arm not in ARM_KEYS or producer not in PRODUCERS or int(budget) not in BUDGETS:
        raise ValueError("unknown arm/producer/budget")
    return canonical(dict(
        launcher_revision=LAUNCHER_REVISION, runtime_revision=RUNTIME_REVISION,
        scientific_runtime_basis=SCIENTIFIC_RUNTIME_BASIS, launcher_binding=launcher_binding(),
        paired_retention=retention_registration(), channel=channel_registration(root),
        route=route_registration(root), capacity=capacity_registration(), evidence_additions=evidence_registration(),
        pose_repair=pose_registration(), map_class=map_class_registration(), avo_geometry=avo_registration(),
        storage_accounting=accounting_registration(), radio_telemetry=radio_telemetry_registration(root),
        bookkeeping=bookkeeping_registration(), actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=None, budget=int(budget),
        arm=arm, external_fixed_action=fixed_action(arm, root),
        selection_semantics=SELECTION_SEMANTICS_AGENT if arm == AGENT_ARM else SELECTION_SEMANTICS_FIXED,
        zero_completion=ZERO_COMPLETION, gt_gc_arm=gc_arm, gc_policy=GC_POLICY, gc_safety=SAFETY,
        numeric_producer=producer, producer_substitutions=list(SUBSTITUTIONS) if producer == "v2" else []))
