"""Prospective publisher, verifiers and guarded launch for static-mode v13 (one prospective root per arm).

* ``prepare`` derives one create-only registration per arm from the sealed V4 prospective (revalidated under the V4
  contract), exactly as V6-V12 did, then applies the registered v13 changes and compares the result with the ACTUAL
  V12 registration (``bad_all_prospective_02``, the completed 2026-10-09 capture):
    - top-level keys may differ only inside ``V12_DIFFERENCES_ALLOWED`` (identity/binding + the registered changes);
    - every preserved scientific setting in ``V12_IDENTICAL_FIELDS`` is byte-identical to V12;
    - ``exact_gt_limits`` differs from V12 ONLY in ``max_frames`` (3,900 -> 1,000);
    - ``gt_gc`` equals V12's after the package-name mapping, except ``trace_limits`` = ``trace_limits(1000)``;
    - ``radio_telemetry`` equals V12's after the package-name mapping.
* Prerequisites bind: the sealed V4 base and its proofs, the v13 binding regression and real-service proof (v13
  sources), the sealed v8 tilt / v7 FREEZE / AVO / storage-accounting qualifications (unchanged sources), the V12
  registered PASS assessment of the actual V12 registration, the v13 identity qualification (identities_01) and the
  v13 outcome-aware assessment qualification (M1 reproduction).
* ``launch`` refuses unless a sealed fresh-process startup verification of THIS prospective passed, then runs the
  retained host lifecycle with the v13 runner/partition injected (no older host module is mutated).
The V12 host's pure helpers are imported (read-only use; nothing is rebound).
"""
import argparse
import inspect
import json
import os
from pathlib import Path

from . import contract as C
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_m3_route_b_v8 import contract as V8
from rl_agent.splitfusion_m3_route_b_v12 import host as V12H

SCHEMA = "splitfusion.static_modes_v13.prospective.v1"
PREREQUISITE_SCHEMA = "splitfusion.static_modes_v13.prerequisite.v1"
STARTUP_SCHEMA = "splitfusion.static_modes_v13.startup_verification.v1"
REGRESSION_SCHEMA = "splitfusion.static_modes_v13.binding_regression.v1"
SERVICE_PROOF_SCHEMA = "splitfusion.static_modes_v13.service_proof.v1"
IDENTITIES_SCHEMA = "splitfusion.static_modes_v13.identities.v1"
ASSESSMENT_QUALIFICATION_SCHEMA = "splitfusion.static_modes_v13_offline_v1.assessment_qualification.v1"
V12_ROOT_RELATIVE = "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v12/bad_all_prospective_02"
V12_REGISTRATION_FILE_SHA256 = "8cfbd1004f63effe4134bbf81e0692519da9610068f5fe6c2dd849e83adc3428"
V12_REGISTRATION_CONTENT_SHA256 = "19cdc8212ab19f73f9b5d13e503b5019f50a4254a80988cc84f8593b4f4cf6f3"
V12_ASSESSMENT_SCHEMA = "splitfusion.m3_route_b_v12_offline_v1.registered_assessment.v1"
_sealed_descriptor, _digest_only = V12H._sealed_descriptor, V12H._digest_only
compose_text, write = V12H.compose_text, V12H.write

V12_DIFFERENCES_ALLOWED = frozenset((
    "id", "child_module", "edge_module", "exact_gt_sources", "controller_lineage_sha256", "numeric_run_binding",
    "gt_gc", "paired_retention", "prerequisite_binding", "launcher_binding", "launcher_revision", "runtime_revision",
    "claim_scope", "arms", "experiment_binds", "registration_sha256", "channel", "radio_telemetry", "bookkeeping",
    # registered v13 changes
    "transmitted_budget", "g1_boundary_budget", "exact_gt_limits", "static_arm", "external_fixed_action",
    "selection_semantics", "zero_completion_semantics"))
V12_IDENTICAL_FIELDS = (
    ("channel", ("carrier_cell",)), ("channel", ("live_config",)), ("channel", ("design_config",)),
    ("channel", ("traces_csv",)), ("channel", ("mapping_csv",)), ("channel", ("sample_period_ms",)),
    ("channel", ("catch_up_policy",)), ("channel", ("retained_controller_module",)),
    ("channel", ("transitions_depend_on_actor",)), ("channel", ("live_config_sha256",)),
    ("channel", ("design_config_sha256",)), ("channel", ("traces_sha256",)), ("channel", ("mapping_sha256",)),
    ("channel", ("radio_profile",)), ("channel", ("restore",)),
    ("radio_telemetry", ("database_sha256",)), ("radio_telemetry", ("events",)),
    ("radio_telemetry", ("quantities",)), ("radio_telemetry", ("bounds",)),
    ("bookkeeping", ("end_of_route_drain", "bounds")), ("bookkeeping", ("offline_p3", "rule")),
    ("route", ()), ("capacity", ()), ("pose_repair", ()), ("map_class", ()),
    ("evidence_additions", ()), ("lifecycle", ()), ("support_sha256", ()),
    ("network_profile", ()), ("live_metric_contract", ()), ("policy", ()), ("actor_audit", ()),
    ("semantics", ()), ("selection", ()), ("selection_scope", ()), ("frozen_deadline_ms", ()),
    ("quality_evaluation_inside_deadline", ()), ("online_learning", ()), ("no_training", ()),
    ("numeric_prediction_contract", ()), ("numeric_producer_binding", ()),
    ("ue_input_candidate", ()), ("mask_lookup_candidate", ()), ("archive_capacity_repair", ()),
    ("exact_gt_socket_name", ()), ("exact_gt_capability_name", ()), ("scientific_runtime_basis", ()),
    ("profiler", ()))
V12_NAME_MAP = (("splitfusion_static_modes_v13_offline_v1", "splitfusion_m3_route_b_v12_offline_v1"),
                ("splitfusion_static_modes_v13", "splitfusion_m3_route_b_v12"),
                ("static_modes_v13", "m3_route_b_v12"))
STARTUP_ROLES = ("host", "runner_parent", "runner_cli_preflight", "ue_child", "edge", "gt_service_workers")
LIFECYCLE = {C.ARM_BUDGET: C.LIFECYCLE}


def _verify_record(output, lifecycle=LIFECYCLE):
    output = Path(output).resolve(strict=True)
    record = json.loads((output / "PROSPECTIVE.json").read_bytes())
    if (record.get("schema") != SCHEMA or record.get("budget") not in C.BUDGETS
            or record.get("gc_arm") not in C.ARMS or record.get("producer") not in C.PRODUCERS
            or record.get("static_arm") not in C.ARM_KEYS):
        raise ValueError("foreign prospective registration")
    root = Path(record["worktree"]).resolve(strict=True)
    if record.get("runtime_sources") != C.source_inventory(root):
        raise ValueError("complete runtime source inventory drift")
    for name, digest in record["sources"].items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("prospective dependency changed: " + name)
    for name, field in (("registration.json", "registration_file_sha256"), ("arm.json", "arm_file_sha256"),
                        ("compose_partition.yaml", "compose_file_sha256"),
                        ("PREREQUISITE.json", "prerequisite_file_sha256"),
                        ("ACCEPTANCE.json", "acceptance_file_sha256")):
        if C.sha(output / name) != record[field]:
            raise ValueError("prospective file drift: " + name)
    if (output / C.RETENTION_PLAN_FILE).exists():
        raise ValueError("a retention plan file is present although retention is registered disabled")
    reg = json.loads((output / "registration.json").read_bytes())
    arm = record["static_arm"]
    if (reg.get("id") != C.ID or reg.get("transmitted_budget") != record["budget"]
            or reg.get("g1_boundary_budget") != record["budget"]
            or reg.get("child_module") != C.CHILD or reg.get("edge_module") != C.EDGE
            or reg.get("exact_gt_sources") != C.exact_sources(root)
            or reg.get("live_metric_contract") != C.metrics()
            or reg.get("registration_sha256") != C.canonical(
                {key: value for key, value in reg.items() if key != "registration_sha256"})):
        raise ValueError("foreign/tampered static-mode v13 registration")
    if reg.get("network_profile") != C.NETWORK_PROFILE or reg.get("channel") != C.channel_registration(root):
        raise ValueError("registration schedule differs from the registered FAVORABLE_STABLE trace / carrier cell")
    if reg.get("pose_repair") != C.pose_registration() or reg.get("map_class") != C.map_class_registration():
        raise ValueError("registration pose/map-class repairs differ from the sealed contract")
    if (reg.get("route") != C.route_registration(root) or reg.get("capacity") != C.capacity_registration()
            or reg.get("evidence_additions") != C.evidence_registration()
            or reg.get("lifecycle") != lifecycle[record["budget"]]):
        raise ValueError("registration route/capacity/evidence/lifecycle differ from the contract")
    if "forced_action" in reg or reg.get("exact_gt_limits") != C.limits(record["budget"]):
        raise ValueError("registration carries a forced action or foreign limits")
    if (reg.get("radio_telemetry") != C.radio_telemetry_registration(root)
            or reg.get("bookkeeping") != C.bookkeeping_registration()):
        raise ValueError("registration radio telemetry / bookkeeping sections differ from the v13 contract")
    if (reg.get("gt_gc") != C.gc_registration(record["gc_arm"], record["budget"], record["producer"])
            or reg["arms"]["C2"]["execute_token"] != C.token(arm, record["budget"], record["producer"])):
        raise ValueError("registration GC arm, limits or token differ from the prospective record")
    semantics = C.SELECTION_SEMANTICS_AGENT if arm == C.AGENT_ARM else C.SELECTION_SEMANTICS_FIXED
    if (reg.get("static_arm") != arm or reg.get("external_fixed_action") != C.fixed_action(arm, root)
            or reg.get("selection_semantics") != semantics
            or reg.get("zero_completion_semantics") != C.ZERO_COMPLETION
            or reg.get("paired_retention") != C.retention_registration()):
        raise ValueError("registration arm identity / selection / retention differ from the v13 contract")
    return record, reg


def check_v12_qualification(descriptor, root):
    value = _sealed_descriptor(descriptor)
    binding = value.get("binding") or {}
    exit_status = value.get("exit_status") or {}
    registration = Path(root) / V12_ROOT_RELATIVE / "registration.json"
    if (value.get("schema") != V12_ASSESSMENT_SCHEMA or value.get("verdict") != "PASS_M3_FULL_ROUTE_INTEGRATION"
            or exit_status.get("numeric_host_exit_status") != 0 or exit_status.get("captured_at_launch") is not True
            or binding.get("registration_id") != C.V12.ID or binding.get("runtime_revision") != C.V12.RUNTIME_REVISION
            or binding.get("registration_file_sha256") != V12_REGISTRATION_FILE_SHA256
            or binding.get("registration_content_sha256") != V12_REGISTRATION_CONTENT_SHA256
            or C.sha(registration) != V12_REGISTRATION_FILE_SHA256):
        raise ValueError("V12 qualification is not the registered PASS of the actual V12 registration")
    return value


def check_prerequisites(root, proof):
    root = Path(root).resolve(strict=True)
    if proof.get("schema") != PREREQUISITE_SCHEMA:
        raise ValueError("foreign v13 prerequisite record")
    sealed = proof.get("sealed_v4") or {}
    V12H._verify_sealed_v4(sealed.get("path"), sealed.get("sha256"))
    regression = _sealed_descriptor(proof["binding_regression"])
    if (regression.get("schema") != REGRESSION_SCHEMA or regression.get("status") != "PASS"
            or regression.get("live_services_started") is not False
            or regression.get("sources") != C.exact_sources(root)
            or regression.get("launcher_binding") != C.launcher_binding()
            or not all(regression.get("checks", {}).values()) or len(regression.get("checks", {})) < 8):
        raise ValueError("v13 binding regression does not bind the prospective sources")
    service = _sealed_descriptor(proof["service_proof"])
    if (service.get("schema") != SERVICE_PROOF_SCHEMA or service.get("status") != "PASS"
            or service.get("live_services_started") is not False
            or service.get("sources") != C.exact_sources(root)
            or service.get("launcher_binding") != C.launcher_binding()
            or not service.get("checks") or not all(service["checks"].values())):
        raise ValueError("v13 real-service proof does not bind the prospective sources")
    tilt = _sealed_descriptor(proof["v8_tilt_service_proof"])
    if (tilt.get("schema") != V12H.V8_TILT_PROOF_SCHEMA or tilt.get("status") != "PASS"
            or tilt.get("sources") != V8.exact_sources(root) or tilt.get("launcher_binding") != V8.launcher_binding()
            or not all((tilt.get("checks") or {"": False}).values())):
        raise ValueError("sealed v8 real-service tilt proof does not bind the unchanged v8 sources")
    freeze = _sealed_descriptor(proof["v7_freeze_lifetime_proof"])
    if (freeze.get("schema") != V12H.V7_FREEZE_PROOF_SCHEMA or freeze.get("status") != "PASS"
            or freeze.get("sources") != V7.exact_sources(root)
            or freeze.get("launcher_binding") != V7.launcher_binding()
            or not all((freeze.get("checks") or {"": False}).values())):
        raise ValueError("sealed v7 FREEZE lifetime proof does not bind the unchanged v7 sources")
    qualification = proof.get("avo_tilt_qualification") or {}
    if set(qualification) != set(C.AVO_TILT_EVIDENCE):
        raise ValueError("tilt-aware AVO qualification evidence set differs from the contract")
    for name, descriptor in qualification.items():
        value = _sealed_descriptor(descriptor)
        if (descriptor["sha256"] != C.AVO_TILT_EVIDENCE[name] or value.get("status") != "PASS"
                or value.get("schema") != C.AVO_TILT_EVIDENCE_SCHEMAS[name]
                or not all((value.get("checks") or {"": False}).values())):
            raise ValueError("tilt-aware AVO qualification record is not the qualified PASS: " + name)
    V12H.check_accounting_qualification(proof.get("storage_accounting_qualification") or {})
    check_v12_qualification(proof.get("v12_qualification") or {}, root)
    identities = _sealed_descriptor(proof["identities"])
    table = json.loads((root / C.FIXED_BUNDLES_FILE).read_bytes())
    if (identities.get("schema") != IDENTITIES_SCHEMA or identities.get("status") != "PASS"
            or table.get("source_sha256") != proof["identities"]["sha256"]
            or {str(r["mode_id"]): r["execution_bundle_sha256"] for r in identities["identities"]} != table["modes"]):
        raise ValueError("frozen fixed identities are not the qualified identities_01 result")
    assessment = _sealed_descriptor(proof["assessment_qualification"])
    if (assessment.get("schema") != ASSESSMENT_QUALIFICATION_SCHEMA or assessment.get("status") != "PASS"
            or {k: v for k, v in (assessment.get("sources") or {}).items() if k.startswith(C.OFFLINE_ASSESSMENT_PACKAGE)}
            != {k: v for k, v in C.exact_sources(root).items() if k.startswith(C.OFFLINE_ASSESSMENT_PACKAGE)}):
        raise ValueError("outcome-aware assessment qualification does not bind the registered assessment sources")
    return proof


def verify_prerequisite(output, record):
    path = Path(output) / "PREREQUISITE.json"
    if C.sha(path) != record["prerequisite_file_sha256"]:
        raise ValueError("prerequisite file drift")
    return check_prerequisites(Path(record["worktree"]), json.loads(path.read_bytes()))


def verify_prospective(output):
    record, registration = _verify_record(output)
    output = Path(output).resolve(strict=True)
    if (output / "compose_partition.yaml").read_text() != compose_text(output, record["retained_code"]):
        raise ValueError("edge partition mount does not bind this prospective root")
    if (record.get("launcher_revision") != C.LAUNCHER_REVISION or record.get("runtime_revision") != C.RUNTIME_REVISION
            or registration.get("launcher_revision") != C.LAUNCHER_REVISION
            or registration.get("runtime_revision") != C.RUNTIME_REVISION
            or registration.get("launcher_binding") != C.launcher_binding()):
        raise ValueError("foreign or incoherent v13 launcher binding")
    return record


def verify_startup_proof(output, proof_path):
    output = Path(output).resolve(strict=True)
    proof_path = Path(proof_path).resolve(strict=True)
    value = _sealed_descriptor(dict(path=str(proof_path), sha256=C.sha(proof_path),
                                    manifest_sha256=C.sha(proof_path.parent / "OUTPUT_SHA256.json")))
    record = json.loads((output / "PROSPECTIVE.json").read_bytes())
    roles = value.get("roles") or {}
    if (value.get("schema") != STARTUP_SCHEMA or value.get("status") != "PASS"
            or value.get("prospective") != str(output)
            or value.get("prospective_file_sha256") != C.sha(output / "PROSPECTIVE.json")
            or value.get("registration_file_sha256") != C.sha(output / "registration.json")
            or value.get("sources") != C.source_inventory(Path(record["worktree"]))
            or value.get("launcher_binding") != C.launcher_binding()
            or set(roles) != set(STARTUP_ROLES) or any(roles[name].get("status") != "PASS" for name in roles)
            or value.get("all_negative_controls_rejected") is not True
            or value.get("live_services_started") is not False):
        raise ValueError("startup verification does not cover this prospective, its sources and every role")
    return value


def _v12_registration(root):
    path = Path(root) / V12_ROOT_RELATIVE / "registration.json"
    if C.sha(path) != V12_REGISTRATION_FILE_SHA256:
        raise ValueError("the actual V12 registration drifted: " + str(path))
    return json.loads(path.read_bytes())


def _normalise(value):
    if isinstance(value, dict):
        return {_normalise(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, str):
        for mine, theirs in V12_NAME_MAP:
            value = value.replace(mine, theirs)
    return value


def v12_identity(registration, root):
    """Every V12 difference is registered; every preserved setting is byte-identical; budget-derived values differ
    only where registered."""
    v12 = _v12_registration(root)
    differences = sorted(k for k in set(v12) | set(registration) if v12.get(k, KeyError) != registration.get(k, KeyError))
    if not set(differences) <= V12_DIFFERENCES_ALLOWED:
        raise ValueError("v13 registration differs from V12 outside the registered changes: "
                         + ", ".join(sorted(set(differences) - V12_DIFFERENCES_ALLOWED)))
    differing, checked = [], {}
    for key, path in V12_IDENTICAL_FIELDS:
        mine, theirs = registration.get(key, KeyError), v12.get(key, KeyError)
        for step in path:
            mine = mine.get(step, KeyError) if isinstance(mine, dict) else KeyError
            theirs = theirs.get(step, KeyError) if isinstance(theirs, dict) else KeyError
        name = ":".join((key, *path))
        if mine is KeyError or theirs is KeyError or mine != theirs:
            differing.append(name)
        checked[name] = True
    limits_mine = dict(registration["exact_gt_limits"])
    limits_v12 = dict(v12["exact_gt_limits"])
    if {k: v for k, v in limits_mine.items() if k != "max_frames"} != {k: v for k, v in limits_v12.items()
                                                                       if k != "max_frames"} \
            or limits_mine["max_frames"] != C.ARM_BUDGET:
        differing.append("exact_gt_limits (beyond max_frames)")
    gc_mine = _normalise(dict(registration["gt_gc"], trace_limits=None))
    if gc_mine != dict(v12["gt_gc"], trace_limits=None) or \
            registration["gt_gc"]["trace_limits"] != C.trace_limits(C.ARM_BUDGET):
        differing.append("gt_gc (beyond package names and budget-derived trace_limits)")
    if _normalise(registration["radio_telemetry"]) != v12["radio_telemetry"]:
        differing.append("radio_telemetry (beyond package names)")
    if differing:
        raise ValueError("v13 registration changes settings that must equal V12: " + ", ".join(differing))
    return dict(differs_from_v12=differences, identical_to_v12=sorted(checked),
                v12_registration_file_sha256=V12_REGISTRATION_FILE_SHA256)


def build_proof(root, *, base_prospective, regression, service_proof, v8_tilt_proof, v7_freeze_proof,
                avo_evidence_root, accounting_evidence_root, v12_assessment, identities, assessment_qualification):
    base_path = Path(base_prospective).resolve(strict=True)

    def descriptor(path):
        path = Path(path).resolve(strict=True)
        return dict(path=str(path), sha256=C.sha(path), manifest_sha256=C.sha(path.parent / "OUTPUT_SHA256.json"))

    def digest_descriptor(path):
        path = Path(path).resolve(strict=True)
        return dict(path=str(path), sha256=C.sha(path))
    accounting_root = Path(accounting_evidence_root).resolve(strict=True)
    avo_root = Path(avo_evidence_root).resolve(strict=True)
    return dict(
        schema=PREREQUISITE_SCHEMA,
        sealed_v4=dict(path=str(base_path), sha256=C.sha(base_path),
                       registration_sha256=C.sha(base_path.parent / "registration.json"),
                       prerequisite_sha256=C.sha(base_path.parent / "PREREQUISITE.json")),
        binding_regression=descriptor(regression), service_proof=descriptor(service_proof),
        v8_tilt_service_proof=descriptor(v8_tilt_proof), v7_freeze_lifetime_proof=descriptor(v7_freeze_proof),
        avo_tilt_qualification={name: descriptor(avo_root / name / "RESULT.json") for name in C.AVO_TILT_EVIDENCE},
        storage_accounting_qualification={
            name: (descriptor if (accounting_root / name / "OUTPUT_SHA256.json").is_file() else digest_descriptor)(
                accounting_root / name / filename)
            for name, filename in C.ACCOUNTING_EVIDENCE_FILES.items()},
        v12_qualification=descriptor(v12_assessment), identities=descriptor(identities),
        assessment_qualification=descriptor(assessment_qualification),
        reuse=dict(v12_runtime="experiments/splitfusion_m3_route_b_v12/registered_assessment_01 (PASS 24/24, rc 0) "
                               "of the actual V12 registration this prospective is compared against field by field",
                   unchanged_components="sealed V4 base proofs, v8 tilt, v7 FREEZE, tilt-aware AVO and V3.v2 "
                                        "storage-accounting qualifications are reused unchanged (their sources are "
                                        "byte-identical and bound)"),
        live_services_started=False)


def prepare(output, *, arm, **proofs):
    """Derive one create-only v13 registration (one arm) from the sealed V4 prospective."""
    from rl_agent.splitfusion_run5_full_live_v1 import actor as PA
    from rl_agent.splitfusion_numeric_gt_runtime_v1.deploy_live import write_bytes
    if arm not in C.ARM_KEYS:
        raise ValueError("unknown static arm: " + repr(arm))
    root = Path(__file__).resolve().parents[2]
    output = Path(output).absolute()
    if output.exists() or not output.resolve().is_relative_to(root):
        raise ValueError("new owned output directory required")
    if os.getuid() == 0:
        raise ValueError("host publisher must be the unprivileged user")
    base_path = Path(proofs["base_prospective"]).resolve(strict=True)
    base_record, base_reg, _ = V12H._verify_sealed_v4(base_path, C.sha(base_path))
    if Path(base_record["worktree"]) != root:
        raise ValueError("sealed v4 prospective must belong to this deployment worktree")
    proof = build_proof(root, **proofs)
    check_prerequisites(root, proof)
    inventory, exact = C.source_inventory(root), C.exact_sources(root)
    actor = PA.load_export()
    gc_arm, producer, budget = base_record["arm"], base_record["producer"], C.ARM_BUDGET
    sources = dict(base_record["sources"])
    sources.update({str(root / relative): digest for relative, digest in inventory.items()})
    if any(C.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("retained experiment binding drift")
    fixed = C.fixed_action(arm, root)
    if fixed is not None and C.resolved_fixed_action(arm) != fixed:
        raise ValueError("live resolver differs from the frozen identity for " + arm)
    registration = dict(base_reg)
    registration.update(
        id=C.ID, child_module=C.CHILD, edge_module=C.EDGE, exact_gt_sources=exact,
        transmitted_budget=budget, g1_boundary_budget=budget, exact_gt_limits=C.limits(budget),
        lifecycle=C.LIFECYCLE,
        controller_lineage_sha256=C.lineage(root, actor.boundary_sha256, registration["support_sha256"], budget,
                                            arm, producer, gc_arm),
        numeric_run_binding=C.canonical(dict(
            sources=inventory, actor=actor.boundary_sha256, metrics=C.metrics(), budget=budget, output=str(output),
            support=PA.SUPPORT_SHA256, gt_gc_arm=gc_arm, numeric_producer=producer, static_arm=arm,
            paired_retention="DISABLED", launcher_revision=C.LAUNCHER_REVISION, runtime_revision=C.RUNTIME_REVISION)),
        gt_gc=C.gc_registration(gc_arm, budget, producer), paired_retention=C.retention_registration(),
        prerequisite_binding=proof, launcher_binding=C.launcher_binding(),
        launcher_revision=C.LAUNCHER_REVISION, runtime_revision=C.RUNTIME_REVISION,
        scientific_runtime_basis=C.SCIENTIFIC_RUNTIME_BASIS,
        channel=C.channel_registration(root), radio_telemetry=C.radio_telemetry_registration(root),
        bookkeeping=C.bookkeeping_registration(), static_arm=arm, external_fixed_action=fixed,
        selection_semantics=C.SELECTION_SEMANTICS_AGENT if arm == C.AGENT_ARM else C.SELECTION_SEMANTICS_FIXED,
        zero_completion_semantics=dict(C.ZERO_COMPLETION),
        claim_scope="M1X_STATIC_MODE_COMPARISON__V12_RUNTIME__FAVORABLE_STABLE__1000_FRAME_ARM__"
                    + ("DEPLOYED_ACTOR_REFERENCE" if arm == C.AGENT_ARM else
                       f"EXTERNAL_FIXED_MODE_{fixed['mode_id']:02d}_Q0_NO_SPATIAL_DROPPING__NOT_BEST_ACHIEVABLE"),
        publisher_uid=os.getuid(), publisher_gid=os.getgid())
    registration["arms"] = json.loads(json.dumps(base_reg["arms"]))
    registration["arms"]["C2"]["execute_token"] = C.token(arm, budget, producer)
    registration["arms"]["C2"]["label"] = (
        f"v13 static-mode arm {arm}: V12 runtime, FAVORABLE_STABLE (retained controller, step 0), 1,000 transmitted "
        "frames, retained budget stop + V3 finish repair + bounded drain, paired retention disabled; "
        + ("deployed actor" if arm == C.AGENT_ARM else f"external fixed mode {fixed['mode_id']} at q=0 (actor-free)"))
    registration["experiment_binds"] = dict(base_reg["experiment_binds"])
    registration["experiment_binds"].update({
        "v13:" + str(index): dict(path=str(root / relative), sha256=digest)
        for index, (relative, digest) in enumerate(sorted(inventory.items()))})
    registration.pop("registration_sha256", None)
    registration["registration_sha256"] = C.canonical(registration)
    identity = v12_identity(registration, root)

    output.mkdir(parents=True, exist_ok=False)
    for name in ("ACCEPTANCE.json", "arm.json"):
        write_bytes(output / name, (base_path.parent / name).read_bytes())
    write_bytes(output / "compose_partition.yaml", compose_text(output.resolve(), base_record["retained_code"]).encode())
    write(output / "PREREQUISITE.json", proof)
    write(output / "registration.json", registration)
    record = dict(
        schema=SCHEMA, launcher_revision=C.LAUNCHER_REVISION, runtime_revision=C.RUNTIME_REVISION,
        scientific_runtime_basis=C.SCIENTIFIC_RUNTIME_BASIS, budget=budget, gc_arm=gc_arm, producer=producer,
        static_arm=arm, worktree=str(root), retained_code=base_record["retained_code"], sources=sources,
        runtime_sources=inventory, registration_file_sha256=C.sha(output / "registration.json"),
        compose_file_sha256=C.sha(output / "compose_partition.yaml"), arm_file_sha256=C.sha(output / "arm.json"),
        prerequisite_file_sha256=C.sha(output / "PREREQUISITE.json"),
        acceptance_file_sha256=C.sha(output / "ACCEPTANCE.json"),
        source_acceptance_sha256=base_record["source_acceptance_sha256"], base_v4_prospective_sha256=C.sha(base_path),
        run_order_position=C.RUN_ORDER.index(arm), run_order=list(C.RUN_ORDER), run_order_rule=C.RUN_ORDER_RULE,
        prepared_before_any_frame=True, no_retry=True, live_performance_not_yet_known=True)
    write(output / "PROSPECTIVE.json", record)
    verified = verify_prospective(output)
    verify_prerequisite(output, verified)
    return dict(status="PREPARED_NOT_RUN", output=str(output), arm=arm, token=C.token(arm, budget, producer),
                registration_sha256=C.sha(output / "registration.json"), v12_identity=identity,
                source_inventory_sha256=C.canonical(inventory), source_files=len(inventory))


def verify(output, startup_proof=None):
    record = verify_prospective(output)
    verify_prerequisite(output, record)
    proof = None if startup_proof is None else verify_startup_proof(output, startup_proof)
    identity = C.module_identity
    return dict(role="host", status="PASS", resolved=dict(
        verify_prospective=identity(verify_prospective), verify_prerequisite=identity(verify_prerequisite),
        check_prerequisites=identity(check_prerequisites), verify_startup_proof=identity(verify_startup_proof),
        contract=C.__name__), record_schema=record["schema"], static_arm=record["static_arm"],
        startup_proof_verified=proof is not None)


def launch(output, *, execute_token, startup_proof):
    verify_startup_proof(output, startup_proof)
    from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as retained
    source = inspect.getsource(retained.launch)
    for before, after in (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
                          ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION),
                          ("deadline = time.monotonic() + 1000",
                           f"deadline = time.monotonic() + {int(C.HOST_ROUTE_LIFECYCLE_S)}")):
        if source.count(before) != 1:
            raise ValueError("retained lifecycle seam drift: " + before)
        source = source.replace(before, after)
    namespace = dict(vars(retained), _verify_prospective=verify_prospective, _verify_prerequisite=verify_prerequisite)
    exec(compile(source, "<static-modes-v13-retained-host-lifecycle>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


PROOF_ARGS = ("base_prospective", "regression", "service_proof", "v8_tilt_proof", "v7_freeze_proof",
              "avo_evidence_root", "accounting_evidence_root", "v12_assessment", "identities",
              "assessment_qualification")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=C.ARM_KEYS)
    for name in PROOF_ARGS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--startup-proof", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        missing = [name for name in PROOF_ARGS if getattr(args, name) is None]
        if missing or args.arm is None:
            parser.error("prepare requires --arm and " + ", ".join("--" + m.replace("_", "-") for m in PROOF_ARGS))
        print(json.dumps(prepare(args.output, arm=args.arm, **{n: getattr(args, n) for n in PROOF_ARGS}),
                         sort_keys=True))
        return 0
    if args.operation == "verify":
        print("STATIC_V13_STARTUP_VERIFICATION=" + json.dumps(verify(args.output, args.startup_proof), sort_keys=True))
        return 0
    if args.startup_proof is None:
        parser.error("launch requires --startup-proof")
    return 0 if launch(args.output, execute_token=args.execute, startup_proof=args.startup_proof) else 1


if __name__ == "__main__":
    raise SystemExit(main())
