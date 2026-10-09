"""Prospective publisher, verifiers and guarded launch for M3.1 v8 (copy of the v7 host).

* ``verify_prospective`` / ``verify_prerequisite`` / ``check_prerequisites`` are native to this package and read only
  the v8 contract; the UE child calls this same ``verify_prospective`` (``entries.validate_child_startup``).
* ``prepare`` derives one create-only registration from the sealed v4 prospective (revalidated under the v4
  contract), exactly as v6/v7 did, with the v8 identity: id/token, child/edge trampolines, the v8 GT service/worker
  trampolines (prep: tilt-aware AVO, then the v7 FREEZE lifetime repair) and the per-role ``launcher_binding``.  It
  records which registration fields differ from the actual V7 registration and refuses any difference outside
  identity/binding fields.
* The prerequisite binds the v8 binding regression and the v8 real-service tilt proof (both against the v8 sources),
  the sealed v7 FREEZE lifetime proof (against the unchanged v7 sources) and the candidate's sealed qualification
  records (saved-failure replay, upright parity, real-service proof; qualified AVO digests).
* ``launch`` refuses unless a sealed fresh-process startup verification of THIS prospective (every process role,
  ``startup_verification``) passed; it then runs the retained host lifecycle with the v8 runner/partition and
  verifiers injected (the v3/v5/v6/v7 pattern; no older host module is mutated).
* ``verify`` is the host's verification-only mode: the exact functions ``launch`` calls first, then stop.
"""
import argparse
import inspect
import json
import os
from pathlib import Path

from . import contract as C
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7

SCHEMA = "splitfusion.m3_route_b_v8.prospective.v1"
PREREQUISITE_SCHEMA = "splitfusion.m3_route_b_v8.prerequisite.v1"
STARTUP_SCHEMA = "splitfusion.m3_route_b_v8.startup_verification.v1"
REGRESSION_SCHEMA = "splitfusion.m3_route_b_v8.binding_regression.v1"
V4_PROSPECTIVE_SCHEMA = "splitfusion.m3_route_b_v3.prospective.v1"     # the sealed v4 root used the v3 publisher
V4_PREREQUISITE_SCHEMA = "splitfusion.m3_route_b_v4.prerequisite.v1"
V7_FREEZE_PROOF_SCHEMA = "splitfusion.m3_route_b_v7.freeze_lifetime_service_proof.v1"
TILT_PROOF_SCHEMA = "splitfusion.m3_route_b_v8.tilt_service_proof.v1"
V7_ROOT_RELATIVE = "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261008_m3_route_b_v7/good_all_prospective_01"
# Registration fields that may differ from the actual V7 registration: identity and binding only.
V7_DIFFERENCES_ALLOWED = frozenset((
    "id", "child_module", "edge_module", "exact_gt_sources", "controller_lineage_sha256", "numeric_run_binding",
    "gt_gc", "paired_retention", "prerequisite_binding", "launcher_binding", "launcher_revision", "runtime_revision",
    "claim_scope", "arms", "experiment_binds", "registration_sha256"))
STARTUP_ROLES = ("host", "runner_parent", "runner_cli_preflight", "ue_child", "edge", "gt_service_workers")
LIFECYCLE = {C.FRAME_CEILING: dict(safety_timeout_s=C.SAFETY_TIMEOUT_S, child_timeout_s=C.CHILD_TIMEOUT_S)}


def compose_text(output, retained_code):
    """The edge's partition override: /partition_output MUST be this prospective root (v5 copied the v4 root's file,
    so its edge would have read the v4 registration and published READY into the v4 root)."""
    return ("services:\n  oai-perception-rx:\n    volumes:\n"
            f"      - {output}:/partition_output:rw\n      - {retained_code}:/cpu_partition_code:ro\n")


def write(path, value):
    from rl_agent.splitfusion_numeric_gt_runtime_v1.deploy_live import write_bytes
    write_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def _sealed_descriptor(descriptor):
    """Load a sealed evidence JSON after checking its digest and every file of its OUTPUT_SHA256 manifest."""
    path = Path(descriptor["path"]).resolve(strict=True)
    manifest = path.parent / "OUTPUT_SHA256.json"
    if C.sha(path) != descriptor["sha256"] or C.sha(manifest) != descriptor["manifest_sha256"]:
        raise ValueError("sealed evidence descriptor drift: " + str(path))
    for name, digest in json.loads(manifest.read_bytes()).items():
        candidate = path.parent / name
        if not candidate.is_file() or candidate.is_symlink() or C.sha(candidate) != digest:
            raise ValueError("sealed evidence drift: " + str(candidate))
    return json.loads(path.read_bytes())


def _verify_record(output, contract, schema, lifecycle):
    output = Path(output).resolve(strict=True)
    record = json.loads((output / "PROSPECTIVE.json").read_bytes())
    if (record.get("schema") != schema or record.get("budget") not in contract.BUDGETS
            or record.get("arm") not in contract.ARMS or record.get("producer") not in contract.PRODUCERS):
        raise ValueError("foreign prospective registration")
    root = Path(record["worktree"]).resolve(strict=True)
    if record.get("runtime_sources") != contract.source_inventory(root):
        raise ValueError("complete runtime source inventory drift")
    for name, digest in record["sources"].items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("prospective dependency changed: " + name)
    for name, field in ((contract.RETENTION_PLAN_FILE, "retention_plan_file_sha256"),
                        ("registration.json", "registration_file_sha256"), ("arm.json", "arm_file_sha256"),
                        ("compose_partition.yaml", "compose_file_sha256"),
                        ("PREREQUISITE.json", "prerequisite_file_sha256"),
                        ("ACCEPTANCE.json", "acceptance_file_sha256")):
        if C.sha(output / name) != record[field]:
            raise ValueError("prospective file drift: " + name)
    reg = json.loads((output / "registration.json").read_bytes())
    if (reg.get("id") != contract.ID or reg.get("transmitted_budget") != record["budget"]
            or reg.get("child_module") != contract.CHILD or reg.get("edge_module") != contract.EDGE
            or reg.get("exact_gt_sources") != contract.exact_sources(root)
            or reg.get("live_metric_contract") != contract.metrics()
            or reg.get("registration_sha256") != contract.canonical(
                {key: value for key, value in reg.items() if key != "registration_sha256"})):
        raise ValueError("foreign/tampered full Run-5 live registration")
    if reg.get("network_profile") != contract.NETWORK_PROFILE or reg.get("channel") != contract.channel_registration(root):
        raise ValueError("registration schedule differs from the registered GOOD_ALL schedule / carrier cell")
    if reg.get("pose_repair") != contract.pose_registration() or reg.get("map_class") != contract.map_class_registration():
        raise ValueError("registration pose/map-class repairs differ from the sealed contract")
    if (reg.get("route") != contract.route_registration(root) or reg.get("capacity") != contract.capacity_registration()
            or reg.get("evidence_additions") != contract.evidence_registration()
            or reg.get("lifecycle") != lifecycle[record["budget"]]):
        raise ValueError("registration route/capacity/evidence/lifecycle differ from the M3 contract")
    if "forced_action" in reg or reg.get("exact_gt_limits") != contract.limits(record["budget"]):
        raise ValueError("registration carries a forced action or foreign limits")
    if (reg.get("gt_gc") != contract.gc_registration(record["arm"], record["budget"], record["producer"])
            or reg["arms"]["C2"]["execute_token"] != contract.token(record["arm"], record["budget"], record["producer"])):
        raise ValueError("registration GC arm, limits or token differ from the prospective record")
    from rl_agent.splitfusion_paired_retention_v2 import plan as RP
    plan = RP.load_plan(output / contract.RETENTION_PLAN_FILE, record["retention_plan_file_sha256"])
    if (RP.canonical(plan) != RP.canonical(contract.retention_plan(record["budget"]))
            or reg.get("paired_retention") != contract.retention_registration(record["retention_plan_file_sha256"])):
        raise ValueError("retention plan/registration differ from the contract")
    return record, reg


def _verify_sealed_v4(path, expected_sha256):
    """The sealed v4 prospective and its prerequisite set, revalidated under the v4 contract (v5 function, copied)."""
    prospective = Path(path).resolve(strict=True)
    if C.sha(prospective) != expected_sha256:
        raise ValueError("sealed v4 prospective record drift")
    record, registration = _verify_record(prospective.parent, V4, V4_PROSPECTIVE_SCHEMA, LIFECYCLE)
    prerequisite_path = prospective.parent / "PREREQUISITE.json"
    if C.sha(prerequisite_path) != record["prerequisite_file_sha256"]:
        raise ValueError("sealed v4 prerequisite drift")
    prerequisite = json.loads(prerequisite_path.read_bytes())
    if prerequisite.get("schema") != V4_PREREQUISITE_SCHEMA:
        raise ValueError("foreign sealed v4 prerequisite")
    descriptors = [value for value in prerequisite.values()
                   if isinstance(value, dict) and {"path", "sha256", "manifest_sha256"} <= set(value)]
    if len(descriptors) < 11:
        raise ValueError("sealed v4 prerequisite descriptor set is incomplete")
    for descriptor in descriptors:
        _sealed_descriptor(descriptor)
    worker = _sealed_descriptor(prerequisite["v4_worker_rehearsal"])
    deployment = _sealed_descriptor(prerequisite["v4_deployment_preflight"])
    worktree = Path(record["worktree"])
    if (worker.get("status") != "PASS" or worker.get("live_services_started") is not False
            or worker.get("sources") != V4.exact_sources(worktree)
            or not all(worker.get("fixture_checks", {}).values())):
        raise ValueError("sealed v4 worker/pose proof is not reusable")
    if (deployment.get("status") != "PASS" or deployment.get("live_services_started") is not False
            or deployment.get("sources") != V4.source_inventory(worktree)
            or deployment.get("test_returncode") != 0):
        raise ValueError("sealed v4 deployment/abort proof is not reusable")
    return record, registration, prerequisite


def check_prerequisites(root, proof):
    root = Path(root).resolve(strict=True)
    if proof.get("schema") != PREREQUISITE_SCHEMA:
        raise ValueError("foreign v8 prerequisite record")
    sealed = proof.get("sealed_v4") or {}
    _verify_sealed_v4(sealed.get("path"), sealed.get("sha256"))
    regression = _sealed_descriptor(proof["binding_regression"])
    if (regression.get("schema") != REGRESSION_SCHEMA or regression.get("status") != "PASS"
            or regression.get("live_services_started") is not False
            or regression.get("sources") != C.exact_sources(root)
            or regression.get("launcher_binding") != C.launcher_binding()
            or not all(regression.get("checks", {}).values()) or len(regression.get("checks", {})) < 8):
        raise ValueError("v8 binding regression does not bind the prospective sources")
    tilt = _sealed_descriptor(proof["tilt_service_proof"])
    if (tilt.get("schema") != TILT_PROOF_SCHEMA or tilt.get("status") != "PASS"
            or tilt.get("live_services_started") is not False
            or tilt.get("sources") != C.exact_sources(root)
            or tilt.get("launcher_binding") != C.launcher_binding()
            or not tilt.get("checks") or not all(tilt["checks"].values())):
        raise ValueError("v8 real-service tilt proof does not bind the prospective sources")
    freeze = _sealed_descriptor(proof["v7_freeze_lifetime_proof"])          # reused: v7 sources are unchanged
    if (freeze.get("schema") != V7_FREEZE_PROOF_SCHEMA or freeze.get("status") != "PASS"
            or freeze.get("live_services_started") is not False
            or freeze.get("sources") != V7.exact_sources(root)
            or freeze.get("launcher_binding") != V7.launcher_binding()
            or not freeze.get("checks") or not all(freeze["checks"].values())):
        raise ValueError("sealed v7 FREEZE lifetime proof does not bind the unchanged v7 sources")
    qualification = proof.get("avo_tilt_qualification") or {}
    if set(qualification) != set(C.AVO_TILT_EVIDENCE):
        raise ValueError("tilt-aware AVO qualification evidence set differs from the contract")
    for name, descriptor in qualification.items():
        value = _sealed_descriptor(descriptor)
        if (descriptor["sha256"] != C.AVO_TILT_EVIDENCE[name] or value.get("status") != "PASS"
                or value.get("schema") != C.AVO_TILT_EVIDENCE_SCHEMAS[name]
                or value.get("live_services_started") is not False
                or not value.get("checks") or not all(value["checks"].values())):
            raise ValueError("tilt-aware AVO qualification record is not the qualified PASS: " + name)
    qualified = _sealed_descriptor(qualification["service_proof_02"])
    if {k: v for k, v in qualified["sources"].items() if k in C.AVO_TILT_QUALIFIED} != C.AVO_TILT_QUALIFIED:
        raise ValueError("registered tilt-aware AVO sources are not the qualified ones")
    return proof


def verify_prerequisite(output, record):
    path = Path(output) / "PREREQUISITE.json"
    if C.sha(path) != record["prerequisite_file_sha256"]:
        raise ValueError("prerequisite file drift")
    return check_prerequisites(Path(record["worktree"]), json.loads(path.read_bytes()))


def verify_prospective(output):
    record, registration = _verify_record(output, C, SCHEMA, LIFECYCLE)
    output = Path(output).resolve(strict=True)
    if (output / "compose_partition.yaml").read_text() != compose_text(output, record["retained_code"]):
        raise ValueError("edge partition mount does not bind this prospective root")
    if (record.get("launcher_revision") != C.LAUNCHER_REVISION or record.get("runtime_revision") != C.RUNTIME_REVISION
            or registration.get("launcher_revision") != C.LAUNCHER_REVISION
            or registration.get("runtime_revision") != C.RUNTIME_REVISION
            or registration.get("launcher_binding") != C.launcher_binding()):
        raise ValueError("foreign or incoherent v8 launcher binding")
    return record


def verify_startup_proof(output, proof_path):
    """The sealed fresh-process startup verification must cover every role for THIS prospective and these sources."""
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
            or value.get("live_services_started") is not False):
        raise ValueError("startup verification does not cover this prospective, its sources and every role")
    return value


def v7_differences(registration, root):
    """Top-level registration keys whose values differ from the actual V7 registration (and keys only in one)."""
    v7 = json.loads((Path(root) / V7_ROOT_RELATIVE / "registration.json").read_bytes())
    return sorted(key for key in set(v7) | set(registration) if v7.get(key, KeyError) != registration.get(key, KeyError))


def prepare(output, *, base_prospective, regression, tilt_proof, v7_freeze_proof, avo_evidence_root):
    """Derive one create-only v8 registration from the sealed v4 prospective."""
    from rl_agent.splitfusion_run5_full_live_v1 import actor as PA
    from rl_agent.splitfusion_paired_retention_v2 import plan as RP
    from rl_agent.splitfusion_numeric_gt_runtime_v1.deploy_live import write_bytes

    root = Path(__file__).resolve().parents[2]
    output = Path(output).absolute()
    if output.exists() or not output.resolve().is_relative_to(root):
        raise ValueError("new owned output directory required")
    if os.getuid() == 0:
        raise ValueError("host publisher must be the unprivileged user")
    base_path = Path(base_prospective).resolve(strict=True)
    base_record, base_reg, _ = _verify_sealed_v4(base_path, C.sha(base_path))
    if Path(base_record["worktree"]) != root:
        raise ValueError("sealed v4 prospective must belong to this deployment worktree")
    regression_path = Path(regression).resolve(strict=True)
    tilt_path = Path(tilt_proof).resolve(strict=True)
    freeze_path = Path(v7_freeze_proof).resolve(strict=True)
    avo_root = Path(avo_evidence_root).resolve(strict=True)

    def descriptor(path):
        path = Path(path).resolve(strict=True)
        return dict(path=str(path), sha256=C.sha(path), manifest_sha256=C.sha(path.parent / "OUTPUT_SHA256.json"))
    proof = dict(
        schema=PREREQUISITE_SCHEMA,
        sealed_v4=dict(path=str(base_path), sha256=C.sha(base_path),
                       registration_sha256=C.sha(base_path.parent / "registration.json"),
                       prerequisite_sha256=C.sha(base_path.parent / "PREREQUISITE.json")),
        binding_regression=descriptor(regression_path),
        tilt_service_proof=descriptor(tilt_path),
        v7_freeze_lifetime_proof=descriptor(freeze_path),
        avo_tilt_qualification={name: descriptor(avo_root / name / "RESULT.json") for name in C.AVO_TILT_EVIDENCE},
        reuse=dict(worker_pose="sealed v4 worker_rehearsal (real GT service + both workers through the v4 trampolines, "
                               "high-coordinate P3); v8 runs the same worker implementation under the same v4 worker "
                               "trampoline (via the v7 trampoline); its service-path changes are covered by "
                               "tilt_service_proof and v7_freeze_lifetime_proof",
                   prep_freeze_lifetime="experiments/splitfusion_gt_freeze_lifetime_v1/qualification_02 (main checkout):"
                                        " 49-frame exact control/candidate parity and 600-request mapping/FD soak of "
                                        "the same repair sources (contract PREP_LIFETIME_QUALIFIED); v7 real-service "
                                        "FREEZE proof (v7_freeze_lifetime_proof) on the unchanged v7 trampoline",
                   avo_tilt="experiments/splitfusion_avo_tilt_v1 (main checkout): replay_3502_02 (saved V7 failure, "
                            "geometry-step replay), upright_parity_02 (49 frames, 2,450 rows score-identical), "
                            "service_proof_02 (real service 19/19 incl. envelope fail-closed); 16 focused tests",
                   abort_export="sealed v4 deployment_preflight (abort-evidence exporter); v8 wraps the child with the "
                                "same unchanged exporter",
                   retained_pipeline="sealed v4 prerequisite set (pipeline, capacity, mask lookup, input prep, "
                                     "producer v2, retention offline), revalidated under the v4 contract"),
        live_services_started=False)
    check_prerequisites(root, proof)
    inventory = C.source_inventory(root)
    exact = C.exact_sources(root)
    actor = PA.load_export()
    plan_sha = base_record["retention_plan_file_sha256"]
    arm, producer, budget = base_record["arm"], base_record["producer"], base_record["budget"]
    sources = dict(base_record["sources"])
    sources.update({str(root / relative): digest for relative, digest in inventory.items()})
    if any(C.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("retained experiment binding drift")
    registration = dict(base_reg)
    registration.update(
        id=C.ID, child_module=C.CHILD, edge_module=C.EDGE, exact_gt_sources=exact,
        controller_lineage_sha256=C.lineage(root, actor.boundary_sha256, registration["support_sha256"], budget,
                                            arm, producer, plan_sha),
        numeric_run_binding=C.canonical(dict(
            sources=inventory, actor=actor.boundary_sha256, metrics=C.metrics(), budget=budget, output=str(output),
            support=PA.SUPPORT_SHA256, gt_gc_arm=arm, numeric_producer=producer, retention_plan_sha256=plan_sha,
            launcher_revision=C.LAUNCHER_REVISION, runtime_revision=C.RUNTIME_REVISION)),
        gt_gc=C.gc_registration(arm, budget, producer), paired_retention=C.retention_registration(plan_sha),
        prerequisite_binding=proof, launcher_binding=C.launcher_binding(),
        launcher_revision=C.LAUNCHER_REVISION, runtime_revision=C.RUNTIME_REVISION,
        scientific_runtime_basis=C.SCIENTIFIC_RUNTIME_BASIS,
        claim_scope=base_reg["claim_scope"] + "__V8_COHERENT_BINDING_PREP_FREEZE_LIFETIME_REPAIR_TILT_AWARE_AVO",
        publisher_uid=os.getuid(), publisher_gid=os.getgid())
    registration["arms"] = json.loads(json.dumps(base_reg["arms"]))
    registration["arms"]["C2"]["execute_token"] = C.token(arm, budget, producer)
    registration["arms"]["C2"]["label"] = base_reg["arms"]["C2"]["label"] + " [v8 coherent startup binding + prep FREEZE lifetime repair + tilt-aware pedestrian AVO]"
    registration["experiment_binds"] = dict(base_reg["experiment_binds"])
    registration["experiment_binds"].update({
        "m3_v8:" + str(index): dict(path=str(root / relative), sha256=digest)
        for index, (relative, digest) in enumerate(sorted(inventory.items()))})
    registration.pop("registration_sha256", None)
    registration["registration_sha256"] = C.canonical(registration)
    differences = v7_differences(registration, root)
    if not set(differences) <= V7_DIFFERENCES_ALLOWED:
        raise ValueError("v8 registration differs from V7 outside identity/binding: "
                         + ", ".join(sorted(set(differences) - V7_DIFFERENCES_ALLOWED)))

    output.mkdir(parents=True, exist_ok=False)
    for name in (C.RETENTION_PLAN_FILE, "ACCEPTANCE.json", "arm.json"):
        write_bytes(output / name, (base_path.parent / name).read_bytes())
    write_bytes(output / "compose_partition.yaml", compose_text(output.resolve(), base_record["retained_code"]).encode())
    write(output / "PREREQUISITE.json", proof)
    write(output / "registration.json", registration)
    record = dict(
        schema=SCHEMA, launcher_revision=C.LAUNCHER_REVISION, runtime_revision=C.RUNTIME_REVISION,
        scientific_runtime_basis=C.SCIENTIFIC_RUNTIME_BASIS, budget=budget, arm=arm, producer=producer,
        worktree=str(root), retained_code=base_record["retained_code"], sources=sources, runtime_sources=inventory,
        registration_file_sha256=C.sha(output / "registration.json"),
        compose_file_sha256=C.sha(output / "compose_partition.yaml"), arm_file_sha256=C.sha(output / "arm.json"),
        prerequisite_file_sha256=C.sha(output / "PREREQUISITE.json"),
        acceptance_file_sha256=C.sha(output / "ACCEPTANCE.json"),
        source_acceptance_sha256=base_record["source_acceptance_sha256"],
        retention_plan_file_sha256=C.sha(output / C.RETENTION_PLAN_FILE),
        base_v4_prospective_sha256=C.sha(base_path),
        prepared_before_any_frame=True, no_retry=True, live_performance_not_yet_known=True)
    if record["retention_plan_file_sha256"] != RP.plan_sha256(C.retention_plan(budget)):
        raise ValueError("copied retention plan differs from the unchanged v4 contract")
    write(output / "PROSPECTIVE.json", record)
    verified = verify_prospective(output)
    verify_prerequisite(output, verified)
    return dict(status="PREPARED_NOT_RUN", output=str(output), launcher_revision=C.LAUNCHER_REVISION,
                runtime_revision=C.RUNTIME_REVISION, token=C.token(arm, budget, producer), differs_from_v7=differences,
                registration_sha256=C.sha(output / "registration.json"),
                source_inventory_sha256=C.canonical(inventory), source_files=len(inventory))


def verify(output, startup_proof=None):
    """Host verification-only mode: the functions ``launch`` runs before anything else, then stop."""
    record = verify_prospective(output)
    verify_prerequisite(output, record)
    proof = None if startup_proof is None else verify_startup_proof(output, startup_proof)
    identity = C.module_identity
    return dict(role="host", status="PASS", resolved=dict(
        verify_prospective=identity(verify_prospective), verify_prerequisite=identity(verify_prerequisite),
        check_prerequisites=identity(check_prerequisites), verify_startup_proof=identity(verify_startup_proof),
        contract=C.__name__), record_schema=record["schema"], startup_proof_verified=proof is not None)


def launch(output, *, execute_token, startup_proof):
    """Retained host lifecycle with the v8 runner/partition/verifiers injected; requires the startup proof."""
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
    exec(compile(source, "<m3-route-b-v8-retained-host-lifecycle>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-prospective", type=Path)
    parser.add_argument("--regression", type=Path)
    parser.add_argument("--tilt-proof", type=Path)
    parser.add_argument("--v7-freeze-proof", type=Path)
    parser.add_argument("--avo-evidence-root", type=Path)
    parser.add_argument("--startup-proof", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if None in (args.base_prospective, args.regression, args.tilt_proof, args.v7_freeze_proof,
                    args.avo_evidence_root):
            parser.error("prepare requires --base-prospective, --regression, --tilt-proof, --v7-freeze-proof and "
                         "--avo-evidence-root")
        print(json.dumps(prepare(args.output, base_prospective=args.base_prospective, regression=args.regression,
                                 tilt_proof=args.tilt_proof, v7_freeze_proof=args.v7_freeze_proof,
                                 avo_evidence_root=args.avo_evidence_root), sort_keys=True))
        return 0
    if args.operation == "verify":
        print("M3_V8_STARTUP_VERIFICATION=" + json.dumps(verify(args.output, args.startup_proof), sort_keys=True))
        return 0
    if args.startup_proof is None:
        parser.error("launch requires --startup-proof")
    return 0 if launch(args.output, execute_token=args.execute, startup_proof=args.startup_proof) else 1


if __name__ == "__main__":
    raise SystemExit(main())
