"""One source-bound numeric-GT engineering trial using the retained lifecycle."""
import argparse
import inspect
import json
import os
from pathlib import Path
from . import live_contract as C
from .deploy_live import write_bytes


def write(path, value):
    write_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def offline_proof(path):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_bytes())
    plan = json.loads((path.parent / "PLAN.json").read_bytes())
    if (value.get("status") != "PASS" or value.get("tests", 0) < 159
            or value.get("failures") != 0 or value.get("errors") != 0
            or value.get("skipped") != 0 or value.get("source_files_unchanged") is not True
            or value.get("live_services_started") is not False
            or plan.get("source_sha256") != C.worker_sources()):
        raise ValueError("fresh complete source-bound numerical runtime tests required")
    manifest_path = path.parent / "OUTPUT_SHA256.json"
    manifest = json.loads(manifest_path.read_bytes())
    for name, digest in manifest.items():
        candidate = path.parent / name
        if Path(name).name != name or candidate.is_symlink() or C.sha(candidate) != digest:
            raise ValueError("offline proof file drift")
    return dict(path=str(path), sha256=C.sha(path), manifest_sha256=C.sha(manifest_path),
                tests=value["tests"], sources=plan["source_sha256"])


def verify_prerequisite(output, record):
    from .live_clock import ASYNC_HANDOFF_ID
    path = Path(output) / "PREREQUISITE.json"
    proof = json.loads(path.read_bytes())
    if C.sha(path) != record["prerequisite_file_sha256"]:
        raise ValueError("prerequisite file drift")
    if offline_proof(proof["offline"]["path"]) != proof["offline"]:
        raise ValueError("offline prerequisite changed")
    rehearsal = Path(proof["rehearsal"]["path"])
    result = json.loads(rehearsal.read_bytes())
    if (C.sha(rehearsal) != proof["rehearsal"]["sha256"]
            or result.get("all_children_exit_zero") is not True
            or result.get("sources_unchanged") is not True
            or result.get("live_services_started") is not False
            or result.get("actual_ue_handoff") is not True
            or result.get("handoff_id") != ASYNC_HANDOFF_ID
            or result.get("deadline_claim") is not False
            or len(result.get("frames", [])) < 4):
        raise ValueError("real scored/unscored multi-frame rehearsal proof changed")
    startup = Path(proof["startup"]["path"])
    value = json.loads(startup.read_bytes())
    if (C.sha(startup) != proof["startup"]["sha256"] or value.get("passed") is not True
            or value.get("sources") != record["runtime_sources"]
            or value.get("live_services_started") is not False
            or value.get("cuda_initialized") is not False):
        raise ValueError("new runtime startup proof missing/changed")
    return proof


def verify_prospective(output):
    output = Path(output).resolve(strict=True)
    record = json.loads((output / "PROSPECTIVE.json").read_bytes())
    if record.get("schema") != "splitfusion.numeric_gt.prospective.v1" or record.get("budget") != C.BUDGET:
        raise ValueError("foreign numeric-GT registration")
    root = Path(record["worktree"])
    if record.get("runtime_sources") != C.source_inventory(root):
        raise ValueError("complete runtime source inventory drift")
    for name, digest in record["sources"].items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("prospective dependency changed: " + name)
    for name, field in (("registration.json", "registration_file_sha256"),
                        ("arm.json", "arm_file_sha256"),
                        ("compose_partition.yaml", "compose_file_sha256"),
                        ("PREREQUISITE.json", "prerequisite_file_sha256")):
        if C.sha(output / name) != record[field]:
            raise ValueError("prospective file drift: " + name)
    registration = json.loads((output / "registration.json").read_bytes())
    if (registration.get("id") != C.ID or registration.get("transmitted_budget") != C.BUDGET
            or registration.get("child_module") != C.CHILD or registration.get("edge_module") != C.EDGE
            or registration.get("numeric_worker_sources") != C.worker_sources()
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("foreign/tampered numerical live registration")
    return record


def prepare(output, *, retained_code, offline, startup, rehearsal):
    from .live_clock import ASYNC_HANDOFF_ID
    from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A, prepare_binding as P
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
    offline = offline_proof(offline)
    startup = Path(startup).resolve(strict=True)
    startup_value = json.loads(startup.read_bytes())
    if (startup_value.get("passed") is not True or startup_value.get("sources") != inventory
            or startup_value.get("live_services_started") is not False
            or startup_value.get("cuda_initialized") is not False):
        raise ValueError("complete new runtime startup proof required")
    rehearsal = Path(rehearsal).resolve(strict=True)
    replay = json.loads(rehearsal.read_bytes())
    replay_plan = json.loads((rehearsal.parent / "PLAN.json").read_bytes())
    if (replay.get("all_children_exit_zero") is not True or replay.get("sources_unchanged") is not True
            or replay.get("live_services_started") is not False or len(replay.get("frames", [])) < 4
            or replay.get("actual_ue_handoff") is not True
            or replay.get("handoff_id") != ASYNC_HANDOFF_ID
            or replay.get("deadline_claim") is not False
            or replay_plan.get("source_sha256") != C.worker_sources()):
        raise ValueError("current-source real scored/unscored multi-frame rehearsal required")
    proof = dict(schema="splitfusion.numeric_gt.prerequisite.v1", offline=offline,
                 startup=dict(path=str(startup), sha256=C.sha(startup)),
                 rehearsal=dict(path=str(rehearsal), sha256=C.sha(rehearsal)))
    P.verify_packaged_export(root / "rl_agent/splitfusion_run5_live_avo_v1/actor_export", root)
    actor = load_registered_actor()
    template = root / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
        "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
    reg = json.loads(template.read_bytes())
    sources = {str(root / name): digest for name, digest in inventory.items()}
    sources.update({str(retained / name): C.sha(retained / name)
                    for name in ("host_partition.py", "cpu_partition.py", "edge_gate.py")})
    sources.update({entry["path"]: entry["sha256"] for entry in reg["experiment_binds"].values()})
    if any(C.sha(name) != digest for name, digest in sources.items()):
        raise ValueError("retained experiment binding drift")
    numeric_binding = C.canonical(dict(sources=inventory, actor=actor.boundary_sha256,
        metrics=C.metrics(), budget=C.BUDGET, output=str(output)))
    reg.update(id=C.ID, child_module=C.CHILD, edge_module=C.EDGE,
        transmitted_budget=C.BUDGET, g1_boundary_budget=C.BUDGET,
        controller_lineage_sha256=C.lineage(root, actor.boundary_sha256),
        policy=dict(audit_module="rl_agent.splitfusion_run5_live_avo_v1.audit_actor",
            binding_path="rl_agent/splitfusion_run5_live_avo_v1/actor_export/ACTOR_BINDING.json",
            weights_path="rl_agent/splitfusion_run5_live_avo_v1/actor_export/actor_state_dict.pt",
            expected_weights_sha256=A.WEIGHTS_SHA256),
        live_metric_contract=C.metrics(), numeric_worker_sources=C.worker_sources(),
        numeric_run_binding=numeric_binding, numeric_producer_binding=startup_binding()["binding_sha256"],
        numeric_prediction_contract=native_prediction_contract().document(),
        numeric_limits=dict(max_frames=C.BUDGET, retained_jobs=2, pending_reward_jobs=2,
                            retained_bytes=128 * 1024 * 1024,
                            producer_cpu=4, scorer_cpu=5, process_timeout_s=30),
        publisher_uid=os.getuid(), publisher_gid=os.getgid(), socket_name=C.SOCKET_NAME,
        lifecycle=dict(safety_timeout_s=120, child_timeout_s=1200),
        frozen_deadline_ms=170, quality_evaluation_inside_deadline=True,
        network_profile="FAVORABLE_STABLE", no_retry=True, no_training=True,
        claim_scope="NUMERIC_GT_LIVE_ENGINEERING_ONLY", policy_performance_claim=False,
        historical_quality_not_comparable=True, prerequisite_binding=proof,
        numeric_terminal_semantics="ACTUAL_UE_INCLUSIVE_170MS__ONE_TERMINAL__LATE_REPLY_ORPHAN",
        numeric_archive_semantics="SCORED_OR_EXPLICIT_UNSCORED_TERMINAL__DURABLE_BEFORE_RELEASE",
        additional_acceptance=["numeric identity/context/pose parity", "no producer/scorer/archive leaks",
            "no fabricated score or feedback", "registered CPU-only worker process scope",
            "original actor/state/reward replay exact", "retained disjoint CPU partition restored"])
    reg["semantics"]["actor_loader"] = "rl_agent.splitfusion_run5_live_avo_v1.audit_actor:load_registered_actor"
    reg["profiler"]["minimum_free_disk_bytes"] = 8_000_000_000
    reg["arms"]["C2"]["execute_token"] = C.TOKEN
    reg["arms"]["C2"]["label"] = "Original Run5, numeric GT/scorer, deferred archive, retained CPU partition"
    reg["experiment_binds"].update({"numeric_live:" + str(i): dict(path=name, sha256=digest)
        for i, (name, digest) in enumerate(sorted(sources.items()))})
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = C.canonical(reg)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "PREREQUISITE.json", proof)
    write(output / "registration.json", reg)
    write(output / "arm.json", dict(arm="treatment", original_cpus=list(range(24)),
        radio_cpus=list(range(4)) + list(range(8, 20)), edge_cpus=list(range(4, 8)) + list(range(20, 24))))
    write_bytes(output / "compose_partition.yaml", (
        "services:\n  oai-perception-rx:\n    volumes:\n"
        f"      - {output}:/partition_output:rw\n      - {retained}:/cpu_partition_code:ro\n").encode())
    record = dict(schema="splitfusion.numeric_gt.prospective.v1", budget=C.BUDGET,
        worktree=str(root), retained_code=str(retained), sources=sources, runtime_sources=inventory,
        registration_file_sha256=C.sha(output / "registration.json"),
        compose_file_sha256=C.sha(output / "compose_partition.yaml"),
        arm_file_sha256=C.sha(output / "arm.json"),
        prerequisite_file_sha256=C.sha(output / "PREREQUISITE.json"),
        prepared_before_any_frame=True, no_retry=True, live_performance_not_yet_known=True)
    write(output / "PROSPECTIVE.json", record)
    verify_prospective(output)
    return dict(status="PREPARED_NOT_RUN", output=str(output), budget=C.BUDGET)


def launch(output, *, execute_token):
    # Retain process ownership, cold-host checks, restoration and teardown.
    from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H
    source = inspect.getsource(H.launch)
    for before, after in (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
                          ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION)):
        if source.count(before) != 1:
            raise ValueError("retained lifecycle seam drift")
        source = source.replace(before, after)
    namespace = dict(vars(H), _verify_prospective=verify_prospective,
                     _verify_prerequisite=verify_prerequisite)
    exec(compile(source, "<numeric-gt-retained-host-lifecycle>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-code", type=Path)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--startup", type=Path)
    parser.add_argument("--rehearsal", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args()
    if args.operation == "prepare":
        if None in (args.retained_code, args.offline, args.startup, args.rehearsal):
            parser.error("prepare requires retained-code, offline, startup and rehearsal")
        print(json.dumps(prepare(args.output, retained_code=args.retained_code,
                                 offline=args.offline, startup=args.startup, rehearsal=args.rehearsal)))
        return 0
    return 0 if launch(args.output, execute_token=args.execute) else 1


if __name__ == "__main__":
    raise SystemExit(main())
