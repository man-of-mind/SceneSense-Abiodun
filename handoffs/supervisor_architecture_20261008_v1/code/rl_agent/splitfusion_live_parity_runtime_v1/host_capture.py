"""One 100-frame engineering qualification, never an automatic campaign."""
import argparse
import inspect
import json
from pathlib import Path
from . import contract as C
from .entries import write_x
from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H


def verify_offline(path, worktree):
    from .deploy import CPU_FIXTURE_FILES
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if (value.get("schema") != "splitfusion.parity_runtime.offline.v1"
            or value.get("passed") is not True or value.get("live_started") is not False
            or value.get("sources") != C.source_inventory(worktree)):
        raise ValueError("fresh full runtime offline source-bound verification required")
    for name, digest in value.get("logs", {}).items():
        candidate = path.parent / name
        if Path(name).name != name or candidate.is_symlink() or C.sha(candidate) != digest:
            raise ValueError("offline verification log drift")
    if not value.get("logs"):
        raise ValueError("offline verification logs missing")
    fixtures = {name: digest for name, (_size, digest) in CPU_FIXTURE_FILES.items()}
    if value.get("fixtures") != fixtures or any(C.sha(Path(worktree)/name) != digest for name, digest in fixtures.items()):
        raise ValueError("offline verification finite-fixture drift")
    return {"path": str(path), "sha256": C.sha(path), "passed": True}


def verify_prerequisite(output, record):
    proof = json.loads((Path(output)/"PREREQUISITE.json").read_text())
    if (proof.get("schema") != "splitfusion.parity_runtime.prerequisite.v1"
            or verify_offline(proof["offline"]["path"], record["worktree"]) != proof["offline"]
            or H._probe_prerequisite(proof["sync_probe"]["path"], record["worktree"]) != proof["sync_probe"]):
        raise ValueError("runtime prerequisite changed")
    return proof


def verify_prospective(output):
    record = H._verify_prospective(output)
    if record.get("schema") != "splitfusion.parity_runtime.prospective.v1" or record.get("budget") != C.BUDGET:
        raise ValueError("foreign parity registration")
    if record.get("runtime_sources") != C.source_inventory(record["worktree"]):
        raise ValueError("complete runtime source set drift")
    registration = json.loads((Path(output)/"registration.json").read_text())
    if (registration.get("id") != C.ID or registration.get("child_module") != C.CHILD
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("foreign/tampered runtime registration")
    return record


def prepare(output, worktree, retained_code, *, offline, sync_probe):
    from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A, prepare_binding as P
    from rl_agent.splitfusion_run5_live_avo_v1.audit_actor import load_registered_actor
    output, worktree, retained = (Path(p).resolve() for p in (output, worktree, retained_code))
    if output.exists():
        raise FileExistsError(output)
    if worktree != Path(__file__).resolve().parents[2]:
        raise ValueError("prepare must run from the actual deployed runtime")
    proof = dict(schema="splitfusion.parity_runtime.prerequisite.v1",
        offline=verify_offline(offline, worktree), sync_probe=H._probe_prerequisite(sync_probe, worktree),
        note="Prior exact sensor sync only; repaired geometry still requires this new live qualification")
    P.verify_packaged_export(worktree/"rl_agent/splitfusion_run5_live_avo_v1/actor_export", worktree)
    template = worktree / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
        "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
    reg = json.loads(template.read_text())
    inventory = C.source_inventory(worktree)
    sources = {str(worktree/name): digest for name, digest in inventory.items()}
    sources.update({str(retained/name): C.sha(retained/name)
                    for name in ("host_partition.py", "cpu_partition.py", "edge_gate.py")})
    # Retain every original partition/guard dependency, not only new sources.
    sources.update({item["path"]: item["sha256"] for item in reg["experiment_binds"].values()})
    if any(C.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("retained partition/source drift")
    actor = load_registered_actor()
    reg.update(id=C.ID, child_module=C.CHILD, transmitted_budget=C.BUDGET, g1_boundary_budget=C.BUDGET,
        claim_scope="COMPLETE_GT_RUNTIME_ENGINEERING_QUALIFICATION_ONLY",
        controller_lineage_sha256=C.lineage(worktree, actor.boundary_sha256),
        policy={"audit_module": "rl_agent.splitfusion_run5_live_avo_v1.audit_actor",
            "binding_path": "rl_agent/splitfusion_run5_live_avo_v1/actor_export/ACTOR_BINDING.json",
            "weights_path": "rl_agent/splitfusion_run5_live_avo_v1/actor_export/actor_state_dict.pt",
            "expected_weights_sha256": A.WEIGHTS_SHA256},
        live_metric_contract=C.metrics(), prerequisite_binding=proof,
        frozen_deadline_ms=170, quality_evaluation_inside_deadline=True,
        lifecycle={"safety_timeout_s": 120, "child_timeout_s": 1200},
        network_profile="FAVORABLE_STABLE", no_retry=True, no_training=True,
        historical_quality_not_comparable=True,
        extra_acceptance=["complete target/source/prediction/pose population verified", "actual evaluator drain complete",
            "zero repair/GT/archive faults", "exact original actor/state/reward replay", "retained E1-E5 and restored partition"],
        timing="report inclusive 170ms feedback and GT readiness without subtracting profiler or GT cost")
    reg["semantics"]["actor_loader"] = "rl_agent.splitfusion_run5_live_avo_v1.audit_actor:load_registered_actor"
    reg["profiler"]["minimum_free_disk_bytes"] = 8_000_000_000
    reg["arms"]["C2"]["execute_token"] = C.TOKEN
    reg["arms"]["C2"]["label"] = "Original Run5 + retained cap/partition + complete GT and owned pose repair"
    reg["experiment_binds"].update({f"parity:{i}": {"path": name, "sha256": digest}
        for i, (name, digest) in enumerate(sorted(sources.items()))})
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = C.canonical(reg)
    output.mkdir(parents=False, exist_ok=False)
    write_x(output/"PREREQUISITE.json", proof)
    write_x(output/"registration.json", reg)
    write_x(output/"arm.json", {"arm": "treatment", "original_cpus": list(range(24)),
        "radio_cpus": list(range(4))+list(range(8,20)), "edge_cpus": list(range(4,8))+list(range(20,24))})
    with (output/"compose_partition.yaml").open("x") as stream:
        stream.write("services:\n  oai-perception-rx:\n    volumes:\n"
            f"      - {output}:/partition_output:rw\n      - {retained}:/cpu_partition_code:ro\n")
    record = dict(schema="splitfusion.parity_runtime.prospective.v1", budget=C.BUDGET,
        worktree=str(worktree), retained_code=str(retained), sources=sources, runtime_sources=inventory,
        registration_file_sha256=C.sha(output/"registration.json"),
        compose_file_sha256=C.sha(output/"compose_partition.yaml"), arm_file_sha256=C.sha(output/"arm.json"),
        prerequisite_file_sha256=C.sha(output/"PREREQUISITE.json"), prepared_before_any_frame=True,
        no_retraining=True, no_retry=True, no_heavy_nsight=True, live_performance_not_yet_known=True)
    write_x(output/"PROSPECTIVE.json", record)
    verify_prospective(output)
    return {"status": "RUNTIME_PREPARED_NOT_RUN", "output": str(output), "budget": C.BUDGET}


def launch(output, *, execute_token):
    """Reuse the exact retained lifecycle, substituting only two module names.

    Explicit occurrence checks and regression tests reject accidental broader
    substitutions; no new process/teardown logic or failure retry is introduced.
    """
    source = inspect.getsource(H.launch)
    pairs = (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
             ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION))
    for before, after in pairs:
        if source.count(before) != 1:
            raise ValueError("retained launch seam changed")
        source = source.replace(before, after)
    namespace = dict(H.__dict__, _verify_prospective=verify_prospective,
                     _verify_prerequisite=verify_prerequisite)
    exec(compile(source, "<source-bound-retained-parity-launch>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def postflight(output):
    from .postflight import postflight as repaired, verify_avo_capture
    from rl_agent.splitfusion_run5_live_avo_v1 import verify_live_gt_capture as V
    verify_prospective(output)
    original_verify_avo = V.verify_capture
    old = H._verify_prospective
    H._verify_prospective = lambda path: verify_record_without_recursion(path, old)
    V.verify_capture = verify_avo_capture
    try:
        H.postflight(output)
    finally:
        H._verify_prospective = old
        V.verify_capture = original_verify_avo
    return repaired(output)


def verify_record_without_recursion(path, old):
    record = old(path)
    if record.get("runtime_sources") != C.source_inventory(record["worktree"]):
        raise ValueError("runtime source drift at postflight")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch", "postflight"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--retained-code", type=Path)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--sync-probe", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if any(x is None for x in (args.worktree, args.retained_code, args.offline, args.sync_probe)):
            parser.error("prepare requires worktree, retained-code, offline and sync-probe")
        result = prepare(args.output, args.worktree, args.retained_code, offline=args.offline, sync_probe=args.sync_probe)
    elif args.operation == "launch":
        # Postflight still runs if retained gates return non-PASS. Setup exceptions
        # preserve lifecycle failure evidence and stop; they never imply success.
        live_passed = launch(args.output, execute_token=args.execute)
        result = postflight(args.output)
        result = {**result, "retained_launch_passed": live_passed}
    else:
        result = postflight(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", args.operation == "prepare") else 1


if __name__ == "__main__":
    raise SystemExit(main())
