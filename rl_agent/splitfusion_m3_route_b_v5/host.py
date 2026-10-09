"""Prospective publisher and strict verifiers for the M3.1 v5 launcher repair.

Unlike v4, this module never rebinds globals in an older host module.  The
prospective and prerequisite verifiers are native functions in this version.
The sealed v4 prospective is a prerequisite and is revalidated under the v4
contract before the v5 launcher-only registration is derived from it.
"""
import argparse
import inspect
import json
import os
from pathlib import Path

from . import contract as C
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4

SCHEMA = "splitfusion.m3_route_b_v5.prospective.v1"
PREREQUISITE_SCHEMA = "splitfusion.m3_route_b_v5.prerequisite.v1"
HOST_MODULE = "rl_agent.splitfusion_m3_route_b_v5.host"
LIFECYCLE = {C.FRAME_CEILING: dict(safety_timeout_s=C.SAFETY_TIMEOUT_S,
                                    child_timeout_s=C.CHILD_TIMEOUT_S)}


def write(path, value):
    from rl_agent.splitfusion_numeric_gt_runtime_v1.deploy_live import write_bytes
    write_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def expected_binding():
    return dict(
        launcher_revision=C.LAUNCHER_REVISION,
        runtime_revision=C.RUNTIME_REVISION,
        runner_module="rl_agent.splitfusion_m3_route_b_v5.runner",
        contract_module="rl_agent.splitfusion_m3_route_b_v5.contract",
        verify_prospective_module=HOST_MODULE,
        verify_prerequisite_module=HOST_MODULE,
        prerequisite_checker_module=HOST_MODULE,
        child_module=C.CHILD,
        edge_module=C.EDGE,
    )


def verify_source_inventory(expected, root, *, label="registered source"):
    """Verify an arbitrary relative-path manifest; used by the real verifier and isolated drift controls."""
    root = Path(root).resolve(strict=True)
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError(f"{label} drift: {relative}")
    return dict(expected)


def _verify_manifest_descriptor(descriptor):
    path = Path(descriptor["path"]).resolve(strict=True)
    manifest = path.parent / "OUTPUT_SHA256.json"
    if C.sha(path) != descriptor["sha256"] or C.sha(manifest) != descriptor["manifest_sha256"]:
        raise ValueError("prerequisite descriptor drift: " + str(path))
    for name, digest in json.loads(manifest.read_bytes()).items():
        candidate = path.parent / name
        if not candidate.is_file() or candidate.is_symlink() or C.sha(candidate) != digest:
            raise ValueError("prerequisite evidence drift: " + str(candidate))
    return json.loads(path.read_bytes())


def _verify_record(output, contract, schema):
    output = Path(output).resolve(strict=True)
    record = json.loads((output / "PROSPECTIVE.json").read_bytes())
    if (record.get("schema") != schema or record.get("budget") not in contract.BUDGETS
            or record.get("arm") not in contract.ARMS or record.get("producer") not in contract.PRODUCERS):
        raise ValueError("foreign prospective registration")
    root = Path(record["worktree"]).resolve(strict=True)
    current = contract.source_inventory(root)
    if record.get("runtime_sources") != current:
        raise ValueError("complete runtime source inventory drift")
    verify_source_inventory(record["runtime_sources"], root, label="runtime source")
    for name, digest in record["sources"].items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("prospective dependency changed: " + name)
    for name, field in ((contract.RETENTION_PLAN_FILE, "retention_plan_file_sha256"),
                        ("registration.json", "registration_file_sha256"),
                        ("arm.json", "arm_file_sha256"),
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
            or reg.get("lifecycle") != LIFECYCLE[record["budget"]]):
        raise ValueError("registration route/capacity/evidence/lifecycle differ from the M3 contract")
    if "forced_action" in reg or reg.get("exact_gt_limits") != contract.limits(record["budget"]):
        raise ValueError("registration carries a forced action or foreign limits")
    if (reg.get("gt_gc") != contract.gc_registration(record["arm"], record["budget"], record["producer"])
            or reg["arms"]["C2"]["execute_token"] != contract.token(
                record["arm"], record["budget"], record["producer"])):
        raise ValueError("registration GC arm, limits or token differ from the prospective record")
    from rl_agent.splitfusion_paired_retention_v2 import plan as RP
    plan = RP.load_plan(output / contract.RETENTION_PLAN_FILE, record["retention_plan_file_sha256"])
    if (RP.canonical(plan) != RP.canonical(contract.retention_plan(record["budget"]))
            or reg.get("paired_retention") != contract.retention_registration(record["retention_plan_file_sha256"])):
        raise ValueError("retention plan/registration differ from the contract")
    return record, reg


def _verify_sealed_v4(path, expected_sha256):
    prospective = Path(path).resolve(strict=True)
    if C.sha(prospective) != expected_sha256:
        raise ValueError("sealed v4 prospective record drift")
    record, registration = _verify_record(
        prospective.parent, V4, "splitfusion.m3_route_b_v3.prospective.v1")
    prerequisite_path = prospective.parent / "PREREQUISITE.json"
    if C.sha(prerequisite_path) != record["prerequisite_file_sha256"]:
        raise ValueError("sealed v4 prerequisite drift")
    prerequisite = json.loads(prerequisite_path.read_bytes())
    if prerequisite.get("schema") != "splitfusion.m3_route_b_v4.prerequisite.v1":
        raise ValueError("foreign sealed v4 prerequisite")
    descriptors = [value for value in prerequisite.values()
                   if isinstance(value, dict) and {"path", "sha256", "manifest_sha256"} <= set(value)]
    if len(descriptors) < 11:
        raise ValueError("sealed v4 prerequisite descriptor set is incomplete")
    for descriptor in descriptors:
        _verify_manifest_descriptor(descriptor)
    worker = _verify_manifest_descriptor(prerequisite["v4_worker_rehearsal"])
    deployment = _verify_manifest_descriptor(prerequisite["v4_deployment_preflight"])
    if (worker.get("status") != "PASS" or worker.get("live_services_started") is not False
            or worker.get("sources") != V4.exact_sources(Path(record["worktree"]))
            or not all(worker.get("fixture_checks", {}).values())):
        raise ValueError("sealed v4 worker/pose proof is not reusable")
    if (deployment.get("status") != "PASS" or deployment.get("live_services_started") is not False
            or deployment.get("sources") != V4.source_inventory(Path(record["worktree"]))
            or deployment.get("test_returncode") != 0):
        raise ValueError("sealed v4 deployment/abort proof is not reusable")
    return record, registration, prerequisite


def check_prerequisites(root, proof):
    root = Path(root).resolve(strict=True)
    if proof.get("schema") != PREREQUISITE_SCHEMA:
        raise ValueError("foreign v5 prerequisite record")
    sealed = proof.get("sealed_v4") or {}
    _verify_sealed_v4(sealed.get("path"), sealed.get("sha256"))
    regression = _verify_manifest_descriptor(proof["runner_binding_regression"])
    expected_old = dict(runner_module="rl_agent.splitfusion_m3_route_b_v3.runner",
                        verifier_module="rl_agent.splitfusion_m3_route_b_v3.host")
    if (regression.get("status") != "PASS" or regression.get("live_services_started") is not False
            or regression.get("old_wrong_resolution") != expected_old
            or regression.get("repaired_binding") != expected_binding()
            or regression.get("sources") != C.exact_sources(root)):
        raise ValueError("v5 runner-binding regression does not bind the prospective sources")
    return proof


def verify_prerequisite(output, record):
    path = Path(output) / "PREREQUISITE.json"
    if C.sha(path) != record["prerequisite_file_sha256"]:
        raise ValueError("prerequisite file drift")
    proof = json.loads(path.read_bytes())
    return check_prerequisites(Path(record["worktree"]), proof)


def verify_prospective(output):
    record, registration = _verify_record(output, C, SCHEMA)
    if (record.get("launcher_revision") != C.LAUNCHER_REVISION
            or record.get("runtime_revision") != C.RUNTIME_REVISION
            or registration.get("launcher_binding") != expected_binding()):
        raise ValueError("foreign or incoherent v5 launcher binding")
    return record


def prepare(output, *, base_prospective, regression):
    """Derive one create-only launcher registration from the sealed v4 prospective."""
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
    regression_manifest = regression_path.parent / "OUTPUT_SHA256.json"
    proof = dict(
        schema=PREREQUISITE_SCHEMA,
        sealed_v4=dict(path=str(base_path), sha256=C.sha(base_path),
                       registration_sha256=C.sha(base_path.parent / "registration.json"),
                       prerequisite_sha256=C.sha(base_path.parent / "PREREQUISITE.json")),
        runner_binding_regression=dict(path=str(regression_path), sha256=C.sha(regression_path),
                                       manifest_sha256=C.sha(regression_manifest)),
        reuse=dict(worker_pose="sealed v4 worker_rehearsal_09; dependencies unchanged",
                   abort_export="sealed v4 deployment_preflight_03; dependencies unchanged"),
        live_services_started=False,
    )
    check_prerequisites(root, proof)
    inventory = C.source_inventory(root)
    exact = C.exact_sources(root)
    actor = PA.load_export()
    plan_sha = base_record["retention_plan_file_sha256"]
    registration = dict(base_reg)
    sources = dict(base_record["sources"])
    sources.update({str(root / relative): digest for relative, digest in inventory.items()})
    if any(C.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("retained experiment binding drift")
    run_binding = C.canonical(dict(
        sources=inventory, actor=actor.boundary_sha256, metrics=C.metrics(), budget=base_record["budget"],
        output=str(output), support=PA.SUPPORT_SHA256, gt_gc_arm=base_record["arm"],
        numeric_producer=base_record["producer"], retention_plan_sha256=plan_sha,
        launcher_revision=C.LAUNCHER_REVISION, runtime_revision=C.RUNTIME_REVISION))
    registration.update(
        exact_gt_sources=exact,
        controller_lineage_sha256=C.lineage(
            root, actor.boundary_sha256, registration["support_sha256"], base_record["budget"],
            base_record["arm"], base_record["producer"], plan_sha),
        numeric_run_binding=run_binding,
        prerequisite_binding=proof,
        launcher_binding=expected_binding(),
        launcher_revision=C.LAUNCHER_REVISION,
        runtime_revision=C.RUNTIME_REVISION,
        claim_scope=registration["claim_scope"] + "__EXPLICIT_V5_RUNNER_VERIFIER_BINDING",
        publisher_uid=os.getuid(), publisher_gid=os.getgid(),
    )
    registration["experiment_binds"].update({
        "m3_launcher_v5:" + str(index): dict(path=str(root / relative), sha256=digest)
        for index, (relative, digest) in enumerate(sorted(C.own_sources(root).items()))
    })
    registration.pop("registration_sha256", None)
    registration["registration_sha256"] = C.canonical(registration)

    output.mkdir(parents=True, exist_ok=False)
    for name in (C.RETENTION_PLAN_FILE, "ACCEPTANCE.json", "arm.json", "compose_partition.yaml"):
        write_bytes(output / name, (base_path.parent / name).read_bytes())
    write(output / "PREREQUISITE.json", proof)
    write(output / "registration.json", registration)
    record = dict(
        schema=SCHEMA, launcher_revision=C.LAUNCHER_REVISION, runtime_revision=C.RUNTIME_REVISION,
        budget=base_record["budget"], arm=base_record["arm"], producer=base_record["producer"],
        worktree=str(root), retained_code=base_record["retained_code"], sources=sources,
        runtime_sources=inventory, registration_file_sha256=C.sha(output / "registration.json"),
        compose_file_sha256=C.sha(output / "compose_partition.yaml"),
        arm_file_sha256=C.sha(output / "arm.json"),
        prerequisite_file_sha256=C.sha(output / "PREREQUISITE.json"),
        acceptance_file_sha256=C.sha(output / "ACCEPTANCE.json"),
        source_acceptance_sha256=base_record["source_acceptance_sha256"],
        retention_plan_file_sha256=C.sha(output / C.RETENTION_PLAN_FILE),
        prepared_before_any_frame=True, no_retry=True, live_performance_not_yet_known=True)
    if record["retention_plan_file_sha256"] != RP.plan_sha256(C.retention_plan(record["budget"])):
        raise ValueError("copied retention plan differs from the unchanged v4 contract")
    write(output / "PROSPECTIVE.json", record)
    verified = verify_prospective(output)
    verify_prerequisite(output, verified)
    return dict(status="PREPARED_NOT_RUN", output=str(output), launcher_revision=C.LAUNCHER_REVISION,
                runtime_revision=C.RUNTIME_REVISION, registration_sha256=C.sha(output / "registration.json"),
                source_inventory_sha256=C.canonical(inventory))


def launch(output, *, execute_token):
    """Use the retained live lifecycle with explicit v5 entry-point/verifier injection."""
    from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as retained
    source = inspect.getsource(retained.launch)
    substitutions = (
        ("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
        ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION),
        ("deadline = time.monotonic() + 1000",
         f"deadline = time.monotonic() + {int(C.HOST_ROUTE_LIFECYCLE_S)}"),
    )
    for before, after in substitutions:
        if source.count(before) != 1:
            raise ValueError("retained lifecycle seam drift: " + before)
        source = source.replace(before, after)
    namespace = dict(vars(retained), _verify_prospective=verify_prospective,
                     _verify_prerequisite=verify_prerequisite)
    exec(compile(source, "<m3-route-b-v5-retained-host-lifecycle>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-prospective", type=Path)
    parser.add_argument("--regression", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if args.base_prospective is None or args.regression is None:
            parser.error("prepare requires --base-prospective and --regression")
        print(json.dumps(prepare(args.output, base_prospective=args.base_prospective,
                                 regression=args.regression), sort_keys=True))
        return 0
    return 0 if launch(args.output, execute_token=args.execute) else 1


if __name__ == "__main__":
    raise SystemExit(main())
