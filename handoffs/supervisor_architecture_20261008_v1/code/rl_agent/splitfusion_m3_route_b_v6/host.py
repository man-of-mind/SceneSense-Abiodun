"""Prospective publisher, verifiers and guarded launch for M3.1 v6.

* ``verify_prospective`` / ``verify_prerequisite`` / ``check_prerequisites`` are native to this package and read only
  the v6 contract; the UE child calls this same ``verify_prospective`` (``entries.validate_child_startup``).
* ``prepare`` derives one create-only registration from the sealed v4 prospective (revalidated under the v4
  contract), exactly as v5 did, but with the v6 identity: id/token, child/edge trampolines, registered GT
  trampolines and retention plan id that match what runs, and the per-role ``launcher_binding``.
* ``launch`` refuses unless a sealed fresh-process startup verification of THIS prospective (every process role,
  ``startup_verification``) passed; it then runs the retained host lifecycle with the v6 runner/partition and
  verifiers injected (the v3/v5 pattern; no older host module is mutated).
* ``verify`` is the host's verification-only mode: the exact functions ``launch`` calls first, then stop.
"""
import argparse
import inspect
import json
import os
from pathlib import Path

from . import contract as C
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4

SCHEMA = "splitfusion.m3_route_b_v6.prospective.v1"
PREREQUISITE_SCHEMA = "splitfusion.m3_route_b_v6.prerequisite.v1"
STARTUP_SCHEMA = "splitfusion.m3_route_b_v6.startup_verification.v1"
REGRESSION_SCHEMA = "splitfusion.m3_route_b_v6.binding_regression.v1"
V4_PROSPECTIVE_SCHEMA = "splitfusion.m3_route_b_v3.prospective.v1"     # the sealed v4 root used the v3 publisher
V4_PREREQUISITE_SCHEMA = "splitfusion.m3_route_b_v4.prerequisite.v1"
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
        raise ValueError("foreign v6 prerequisite record")
    sealed = proof.get("sealed_v4") or {}
    _verify_sealed_v4(sealed.get("path"), sealed.get("sha256"))
    regression = _sealed_descriptor(proof["binding_regression"])
    if (regression.get("schema") != REGRESSION_SCHEMA or regression.get("status") != "PASS"
            or regression.get("live_services_started") is not False
            or regression.get("sources") != C.exact_sources(root)
            or regression.get("launcher_binding") != C.launcher_binding()
            or not all(regression.get("checks", {}).values()) or len(regression.get("checks", {})) < 8):
        raise ValueError("v6 binding regression does not bind the prospective sources")
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
        raise ValueError("foreign or incoherent v6 launcher binding")
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


def prepare(output, *, base_prospective, regression):
    """Derive one create-only v6 registration from the sealed v4 prospective."""
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
    proof = dict(
        schema=PREREQUISITE_SCHEMA,
        sealed_v4=dict(path=str(base_path), sha256=C.sha(base_path),
                       registration_sha256=C.sha(base_path.parent / "registration.json"),
                       prerequisite_sha256=C.sha(base_path.parent / "PREREQUISITE.json")),
        binding_regression=dict(path=str(regression_path), sha256=C.sha(regression_path),
                                manifest_sha256=C.sha(regression_path.parent / "OUTPUT_SHA256.json")),
        reuse=dict(worker_pose="sealed v4 worker_rehearsal (real GT service + both workers through the v4 trampolines, "
                               "high-coordinate P3); v6 runs the same trampolines and implementations",
                   abort_export="sealed v4 deployment_preflight (abort-evidence exporter); v6 wraps the child with the "
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
        claim_scope=base_reg["claim_scope"] + "__V6_COHERENT_CROSS_PROCESS_REGISTRATION_BINDING",
        publisher_uid=os.getuid(), publisher_gid=os.getgid())
    registration["arms"] = json.loads(json.dumps(base_reg["arms"]))
    registration["arms"]["C2"]["execute_token"] = C.token(arm, budget, producer)
    registration["arms"]["C2"]["label"] = base_reg["arms"]["C2"]["label"] + " [v6 coherent startup binding]"
    registration["experiment_binds"] = dict(base_reg["experiment_binds"])
    registration["experiment_binds"].update({
        "m3_v6:" + str(index): dict(path=str(root / relative), sha256=digest)
        for index, (relative, digest) in enumerate(sorted(inventory.items()))})
    registration.pop("registration_sha256", None)
    registration["registration_sha256"] = C.canonical(registration)

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
                runtime_revision=C.RUNTIME_REVISION, token=C.token(arm, budget, producer),
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
    """Retained host lifecycle with the v6 runner/partition/verifiers injected; requires the startup proof."""
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
    exec(compile(source, "<m3-route-b-v6-retained-host-lifecycle>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-prospective", type=Path)
    parser.add_argument("--regression", type=Path)
    parser.add_argument("--startup-proof", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if args.base_prospective is None or args.regression is None:
            parser.error("prepare requires --base-prospective and --regression")
        print(json.dumps(prepare(args.output, base_prospective=args.base_prospective, regression=args.regression),
                         sort_keys=True))
        return 0
    if args.operation == "verify":
        print("M3_V6_STARTUP_VERIFICATION=" + json.dumps(verify(args.output, args.startup_proof), sort_keys=True))
        return 0
    if args.startup_proof is None:
        parser.error("launch requires --startup-proof")
    return 0 if launch(args.output, execute_token=args.execute, startup_proof=args.startup_proof) else 1


if __name__ == "__main__":
    raise SystemExit(main())
