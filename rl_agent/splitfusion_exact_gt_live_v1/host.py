"""One source-bound early-GT engineering check using the retained Run-5 host lifecycle.

prepare: writes PREREQUISITE.json, ACCEPTANCE.json (registered before any frame),
registration.json, arm.json, compose_partition.yaml and PROSPECTIVE.json, create-only.
launch:  the retained ``host_capture.launch`` with only the runner/partition modules and the
prospective/prerequisite verifiers substituted (same pattern as the numeric-GT runtime).
"""
import argparse
import inspect
import json
import os
from pathlib import Path

from . import contract as C


def write(path, value):
    from rl_agent.splitfusion_numeric_gt_runtime_v1.deploy_live import write_bytes
    write_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def _proof(path, required):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_bytes())
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"prerequisite {path.name}: {key}={value.get(key)!r} expected {expected!r}")
    manifest = path.parent / "OUTPUT_SHA256.json"
    for name, digest in json.loads(manifest.read_bytes()).items():
        candidate = path.parent / name
        if candidate.is_symlink() or C.sha(candidate) != digest:
            raise ValueError("prerequisite evidence drift: " + str(candidate))
    return dict(path=str(path), sha256=C.sha(path), manifest_sha256=C.sha(manifest)), value


def check_prerequisites(root, offline, rehearsal, startup, container):
    sources = C.exact_sources(root)
    off, off_value = _proof(offline, dict(status="PASS", failures=0, errors=0, live_services_started=False))
    if off_value.get("sources") != sources or off_value.get("tests", 0) < 44:
        raise ValueError("offline proof must bind the exact current sources and the full test set")
    reh, reh_value = _proof(rehearsal, dict(status="PASS", all_children_exit_zero=True, live_services_started=False,
                                            deadline_claim=False))
    if reh_value.get("sources") != sources or len(reh_value.get("scenarios", {})) < 6:
        raise ValueError("integration rehearsal must bind current sources and cover all registered scenarios")
    start, start_value = _proof(startup, dict(passed=True, live_services_started=False, cuda_initialized=False))
    if start_value.get("sources") != C.source_inventory(root):
        raise ValueError("startup proof differs from runtime sources")
    cont, cont_value = _proof(container, dict(status="PASS", live_services_started=False))
    if cont_value.get("sources") != sources:
        raise ValueError("container handoff proof differs from current sources")
    return dict(schema="splitfusion.exact_gt_live.prerequisite.v1", offline=off, rehearsal=reh, startup=start,
                container=cont)


def verify_prerequisite(output, record):
    path = Path(output) / "PREREQUISITE.json"
    if C.sha(path) != record["prerequisite_file_sha256"]:
        raise ValueError("prerequisite file drift")
    proof = json.loads(path.read_bytes())
    root = Path(record["worktree"])
    fresh = check_prerequisites(root, proof["offline"]["path"], proof["rehearsal"]["path"], proof["startup"]["path"],
                                proof["container"]["path"])
    if fresh != proof:
        raise ValueError("prerequisite evidence changed")
    return proof


def verify_prospective(output):
    output = Path(output).resolve(strict=True)
    record = json.loads((output / "PROSPECTIVE.json").read_bytes())
    if record.get("schema") != "splitfusion.exact_gt_live.prospective.v1" or record.get("budget") != C.BUDGET:
        raise ValueError("foreign early-GT registration")
    root = Path(record["worktree"])
    if record.get("runtime_sources") != C.source_inventory(root):
        raise ValueError("complete runtime source inventory drift")
    for name, digest in record["sources"].items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("prospective dependency changed: " + name)
    for name, field in (("registration.json", "registration_file_sha256"), ("arm.json", "arm_file_sha256"),
                        ("compose_partition.yaml", "compose_file_sha256"), ("PREREQUISITE.json", "prerequisite_file_sha256"),
                        ("ACCEPTANCE.json", "acceptance_file_sha256")):
        if C.sha(output / name) != record[field]:
            raise ValueError("prospective file drift: " + name)
    reg = json.loads((output / "registration.json").read_bytes())
    if (reg.get("id") != C.ID or reg.get("transmitted_budget") != C.BUDGET or reg.get("child_module") != C.CHILD
            or reg.get("edge_module") != C.EDGE or reg.get("exact_gt_sources") != C.exact_sources(root)
            or reg.get("live_metric_contract") != C.metrics()
            or reg.get("registration_sha256") != C.canonical({k: v for k, v in reg.items() if k != "registration_sha256"})):
        raise ValueError("foreign/tampered early-GT registration")
    return record


def prepare(output, *, retained_code, offline, rehearsal, startup, container, acceptance):
    from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A, prepare_binding as PB
    from rl_agent.splitfusion_run5_live_avo_v1.audit_actor import load_registered_actor
    from rl_agent.splitfusion_numeric_gt_producer_v1.originals import startup_binding
    from rl_agent.splitfusion_live_parity_repair_v2.prediction_archive import native_prediction_contract
    root = Path(__file__).resolve().parents[2]
    output, retained = Path(output).absolute(), Path(retained_code).resolve(strict=True)
    if output.exists() or not output.resolve().is_relative_to(root):
        raise ValueError("new owned output directory required")
    if os.getuid() == 0:
        raise ValueError("host publisher must be the unprivileged user")
    inventory = C.source_inventory(root)
    proof = check_prerequisites(root, offline, rehearsal, startup, container)
    acceptance_value = json.loads(Path(acceptance).read_bytes())
    if acceptance_value.get("schema") != "splitfusion.exact_gt_live.acceptance.v1":
        raise ValueError("registered acceptance criteria required")
    PB.verify_packaged_export(root / "rl_agent/splitfusion_run5_live_avo_v1/actor_export", root)
    actor = load_registered_actor()
    template = root / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                       "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
    reg = json.loads(template.read_bytes())
    sources = {str(root / name): digest for name, digest in inventory.items()}
    sources.update({str(retained / name): C.sha(retained / name) for name in ("host_partition.py", "cpu_partition.py", "edge_gate.py")})
    sources.update({entry["path"]: entry["sha256"] for entry in reg["experiment_binds"].values()})
    if any(C.sha(name) != digest for name, digest in sources.items()):
        raise ValueError("retained experiment binding drift")
    run_binding = C.canonical(dict(sources=inventory, actor=actor.boundary_sha256, metrics=C.metrics(), budget=C.BUDGET,
                                   output=str(output)))
    reg.update(id=C.ID, child_module=C.CHILD, edge_module=C.EDGE, transmitted_budget=C.BUDGET, g1_boundary_budget=C.BUDGET,
        controller_lineage_sha256=C.lineage(root, actor.boundary_sha256),
        policy=dict(audit_module="rl_agent.splitfusion_run5_live_avo_v1.audit_actor",
            binding_path="rl_agent/splitfusion_run5_live_avo_v1/actor_export/ACTOR_BINDING.json",
            weights_path="rl_agent/splitfusion_run5_live_avo_v1/actor_export/actor_state_dict.pt",
            expected_weights_sha256=A.WEIGHTS_SHA256),
        live_metric_contract=C.metrics(), exact_gt_sources=C.exact_sources(root),
        numeric_run_binding=run_binding, numeric_producer_binding=startup_binding()["binding_sha256"],
        numeric_prediction_contract=native_prediction_contract().document(), exact_gt_limits=dict(C.LIMITS),
        exact_gt_socket_name=C.SOCKET_NAME, exact_gt_capability_name=C.CAPABILITY_NAME,
        publisher_uid=os.getuid(), publisher_gid=os.getgid(),
        lifecycle=dict(safety_timeout_s=120, child_timeout_s=1200), frozen_deadline_ms=170,
        quality_evaluation_inside_deadline=True, network_profile="FAVORABLE_STABLE", no_retry=True, no_training=True,
        claim_scope="EARLY_GT_LIVE_ENGINEERING_ONLY", policy_performance_claim=False,
        historical_quality_not_comparable=True, prerequisite_binding=proof, acceptance_sha256=C.sha(acceptance))
    reg["semantics"]["actor_loader"] = "rl_agent.splitfusion_run5_live_avo_v1.audit_actor:load_registered_actor"
    reg["profiler"]["minimum_free_disk_bytes"] = 8_000_000_000
    reg["arms"]["C2"]["execute_token"] = C.TOKEN
    reg["arms"]["C2"]["label"] = "Original Run5, early-GT host service, bit-exact BGR2RGB, retained CPU partition"
    reg["experiment_binds"].update({"exact_gt_live:" + str(i): dict(path=name, sha256=digest)
                                    for i, (name, digest) in enumerate(sorted(sources.items()))})
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = C.canonical(reg)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "PREREQUISITE.json", proof)
    write(output / "ACCEPTANCE.json", acceptance_value)
    write(output / "registration.json", reg)
    write(output / "arm.json", dict(arm="treatment", original_cpus=list(range(24)),
        radio_cpus=list(range(4)) + list(range(8, 20)), edge_cpus=list(range(4, 8)) + list(range(20, 24))))
    from rl_agent.splitfusion_numeric_gt_runtime_v1.deploy_live import write_bytes
    write_bytes(output / "compose_partition.yaml", ("services:\n  oai-perception-rx:\n    volumes:\n"
        f"      - {output}:/partition_output:rw\n      - {retained}:/cpu_partition_code:ro\n").encode())
    record = dict(schema="splitfusion.exact_gt_live.prospective.v1", budget=C.BUDGET, worktree=str(root),
        retained_code=str(retained), sources=sources, runtime_sources=inventory,
        registration_file_sha256=C.sha(output / "registration.json"),
        compose_file_sha256=C.sha(output / "compose_partition.yaml"), arm_file_sha256=C.sha(output / "arm.json"),
        prerequisite_file_sha256=C.sha(output / "PREREQUISITE.json"),
        acceptance_file_sha256=C.sha(output / "ACCEPTANCE.json"),
        prepared_before_any_frame=True, no_retry=True, live_performance_not_yet_known=True)
    write(output / "PROSPECTIVE.json", record)
    verify_prospective(output)
    return dict(status="PREPARED_NOT_RUN", output=str(output), budget=C.BUDGET)


def launch(output, *, execute_token):
    from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H
    source = inspect.getsource(H.launch)
    for before, after in (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
                          ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION)):
        if source.count(before) != 1:
            raise ValueError("retained lifecycle seam drift")
        source = source.replace(before, after)
    namespace = dict(vars(H), _verify_prospective=verify_prospective, _verify_prerequisite=verify_prerequisite)
    exec(compile(source, "<early-gt-retained-host-lifecycle>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    for name in ("retained-code", "offline", "rehearsal", "startup", "container", "acceptance"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args()
    if args.operation == "prepare":
        print(json.dumps(prepare(args.output, retained_code=args.retained_code, offline=args.offline,
                                 rehearsal=args.rehearsal, startup=args.startup, container=args.container,
                                 acceptance=args.acceptance)))
        return 0
    return 0 if launch(args.output, execute_token=args.execute) else 1


if __name__ == "__main__":
    raise SystemExit(main())
