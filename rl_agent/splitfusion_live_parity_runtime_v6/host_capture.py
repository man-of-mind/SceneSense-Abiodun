"""One newly authorized 100-frame attempt; fresh offline and image proof gates."""
import argparse
import inspect
import json
from pathlib import Path
from types import FunctionType
from . import contract as C
from rl_agent.splitfusion_live_parity_runtime_v1.entries import write_x
from rl_agent.splitfusion_live_parity_runtime_v1.deploy import CPU_FIXTURE_FILES
from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H

OFFLINE_SUITES = frozenset(("integration_v6", "paired_replay_v1", "paired_replay_v2", "integration_v5", "optimization_v2", "integration_v4", "optimization", "integration_v3", "integration_v2", "integration_v1", "repair_v2", "repair", "existing_run5", "frozen_semantics"))
ALLOWED_SKIP = dict(id="rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_prewarm_gt_priority_v2."
                   "AddendumTest.test_addendum_and_manifest_bind_prior_attempts_and_are_not_executed",
                   reason="prior evidence not present on this host")
STARTUP_CHECKS = frozenset(("entry_imported", "registration_checked", "edge_pose_scope",
    "close_evidence_scope", "repaired_evaluation_scope", "retained_dispatch_reached",
    "forbidden_ue_dependencies_absent", "no_cuda_initialized", "no_services_started",
    "target_roundtrip_scored", "actual_close_archives_flushed"))


def _proof(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("proof must be a regular file")
    path = path.resolve(strict=True)
    return path, json.loads(path.read_text())


def _verify_logs(path, value):
    logs = value.get("logs", {})
    if not isinstance(logs, dict) or not logs:
        raise ValueError("verification logs missing")
    for name, digest in logs.items():
        candidate = path.parent / name
        if Path(name).name != name or candidate.is_symlink() or not candidate.is_file() or C.sha(candidate) != digest:
            raise ValueError("verification log drift")


def verify_offline(path, worktree):
    path, value = _proof(path)
    if (value.get("schema") != "splitfusion.parity_runtime.offline.v6"
            or value.get("passed") is not True or value.get("live_started") is not False
            or value.get("training_started") is not False
            or value.get("sources") != C.source_inventory(worktree)):
        raise ValueError("fresh full v5 offline source-bound verification required")
    suites = value.get("suites", {})
    if set(suites) != OFFLINE_SUITES:
        raise ValueError("full eleven-suite offline verification required")
    for name, result in suites.items():
        skips = result.get("skips")
        if (result.get("returncode") != 0 or type(result.get("tests")) is not int or result["tests"] < 1
                or not isinstance(skips, list) or result.get("skipped") != len(skips)
                or (skips and (name != "frozen_semantics" or skips != [ALLOWED_SKIP]))):
            raise ValueError("offline failure, missing tests, or unapproved skip")
    if value.get("total_tests") != sum(result["tests"] for result in suites.values()):
        raise ValueError("offline test count mismatch")
    _verify_logs(path, value)
    if set(value["logs"]) != {name + ".log" for name in OFFLINE_SUITES}:
        raise ValueError("all eleven suite logs required")
    fixtures = {name: digest for name, (_size, digest) in CPU_FIXTURE_FILES.items()}
    if value.get("fixtures") != fixtures or any((Path(worktree)/name).is_symlink()
            or C.sha(Path(worktree)/name) != digest for name, digest in fixtures.items()):
        raise ValueError("offline verification finite-fixture drift")
    return {"path": str(path), "sha256": C.sha(path), "passed": True}


def verify_startup_scope(path, worktree):
    path, value = _proof(path)
    checks = value.get("checks", {})
    if (value.get("schema") != "splitfusion.parity_runtime.edge_startup_scope.v6"
            or value.get("passed") is not True or value.get("live_started") is not False
            or value.get("cuda_initialized") is not False or value.get("image_id") != C.IMAGE_ID
            or value.get("edge_module") != C.EDGE or value.get("sources") != C.source_inventory(worktree)
            or set(checks) != STARTUP_CHECKS or any(flag is not True for flag in checks.values())):
        raise ValueError("fresh pinned-image source-bound edge startup-scope proof required")
    _verify_logs(path, value)
    return {"path": str(path), "sha256": C.sha(path), "passed": True,
            "image_id": C.IMAGE_ID, "edge_module": C.EDGE}


def verify_prerequisite(output, record):
    proof = json.loads((Path(output)/"PREREQUISITE.json").read_text())
    if (proof.get("schema") != "splitfusion.parity_runtime.prerequisite.v6"
            or verify_offline(proof["offline"]["path"], record["worktree"]) != proof["offline"]
            or verify_startup_scope(proof["startup_scope"]["path"], record["worktree"]) != proof["startup_scope"]
            or H._probe_prerequisite(proof["sync_probe"]["path"], record["worktree"]) != proof["sync_probe"]):
        raise ValueError("runtime prerequisite changed")
    return proof


def verify_prospective(output):
    from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import source_binding_hashes
    record = H._verify_prospective(output)
    if record.get("schema") != "splitfusion.parity_runtime.prospective.v6" or record.get("budget") != C.BUDGET:
        raise ValueError("foreign parity registration")
    if record.get("runtime_sources") != C.source_inventory(record["worktree"]):
        raise ValueError("complete runtime source set drift")
    if record.get("optimization_sources") != C.optimization_inventory(record["worktree"]):
        raise ValueError("optimization source binding drift")
    registration = json.loads((Path(output)/"registration.json").read_text())
    if (registration.get("id") != C.ID or registration.get("child_module") != C.CHILD
            or registration.get("edge_module") != C.EDGE or registration.get("transmitted_budget") != C.BUDGET
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("paired_input") != C.INPUT_SPEC
            or registration.get("model_replay_sources") != source_binding_hashes(Path(record["worktree"]))
            or registration.get("optimization_sources") != record["optimization_sources"]
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("foreign/tampered runtime registration")
    return record


def prepare(output, worktree, retained_code, *, offline, sync_probe, startup_scope):
    from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A, prepare_binding as P
    from rl_agent.splitfusion_run5_live_avo_v1.audit_actor import load_registered_actor
    from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import source_binding_hashes
    output, worktree, retained = (Path(p).resolve() for p in (output, worktree, retained_code))
    if output.exists():
        raise FileExistsError(output)
    if worktree != Path(__file__).resolve().parents[2]:
        raise ValueError("prepare must run from the actual deployed runtime")
    proof = dict(schema="splitfusion.parity_runtime.prerequisite.v6",
        offline=verify_offline(offline, worktree), startup_scope=verify_startup_scope(startup_scope, worktree),
        sync_probe=H._probe_prerequisite(sync_probe, worktree),
        note="Prior exact sensor sync only; new startup proof is CPU-only, not live qualification")
    P.verify_packaged_export(worktree/"rl_agent/splitfusion_run5_live_avo_v1/actor_export", worktree)
    template = worktree / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
        "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
    reg = json.loads(template.read_text())
    inventory = C.source_inventory(worktree)
    optimization = C.optimization_inventory(worktree)
    sources = {str(worktree/name): digest for name, digest in inventory.items()}
    sources.update({str(retained/name): C.sha(retained/name)
                    for name in ("host_partition.py", "cpu_partition.py", "edge_gate.py")})
    sources.update({item["path"]: item["sha256"] for item in reg["experiment_binds"].values()})
    if any(C.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("retained partition/source drift")
    actor = load_registered_actor()
    reg.update(id=C.ID, child_module=C.CHILD, edge_module=C.EDGE,
        transmitted_budget=C.BUDGET, g1_boundary_budget=C.BUDGET,
        claim_scope="EXACT_INPUT_PAIRED_QUALITY_DIAGNOSTIC_ONLY",
        controller_lineage_sha256=C.lineage(worktree, actor.boundary_sha256),
        policy={"audit_module": "rl_agent.splitfusion_run5_live_avo_v1.audit_actor",
            "binding_path": "rl_agent/splitfusion_run5_live_avo_v1/actor_export/ACTOR_BINDING.json",
            "weights_path": "rl_agent/splitfusion_run5_live_avo_v1/actor_export/actor_state_dict.pt",
            "expected_weights_sha256": A.WEIGHTS_SHA256},
        live_metric_contract=C.metrics(), prerequisite_binding=proof, optimization_sources=optimization,
        frozen_deadline_ms=170, quality_evaluation_inside_deadline=True,
        lifecycle={"safety_timeout_s": 120, "child_timeout_s": 1200},
        network_profile="FAVORABLE_STABLE", no_retry=True, no_training=True,
        historical_quality_not_comparable=True,
        paired_input=dict(C.INPUT_SPEC),
        model_replay_sources=source_binding_hashes(worktree),
        extra_acceptance=["complete target/source/prediction/pose population verified", "actual evaluator drain complete",
            "zero repair/GT/archive faults", "exact original actor/state/reward replay", "retained E1-E5 and restored partition"],
        timing="report inclusive 170ms feedback and GT readiness without subtracting profiler or GT cost")
    reg["semantics"]["actor_loader"] = "rl_agent.splitfusion_run5_live_avo_v1.audit_actor:load_registered_actor"
    reg["profiler"]["minimum_free_disk_bytes"] = 8_000_000_000
    reg["arms"]["C2"]["execute_token"] = C.TOKEN
    reg["arms"]["C2"]["label"] = "Original Run5 + retained optimized GT + exact CPU-input diagnostic capture"
    reg["experiment_binds"].update({f"parity_v6:{i}": {"path": name, "sha256": digest}
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
    record = dict(schema="splitfusion.parity_runtime.prospective.v6", budget=C.BUDGET,
        worktree=str(worktree), retained_code=str(retained), sources=sources, runtime_sources=inventory,
        optimization_sources=optimization,
        registration_file_sha256=C.sha(output/"registration.json"),
        compose_file_sha256=C.sha(output/"compose_partition.yaml"), arm_file_sha256=C.sha(output/"arm.json"),
        prerequisite_file_sha256=C.sha(output/"PREREQUISITE.json"), prepared_before_any_frame=True,
        no_retraining=True, no_retry=True, no_heavy_nsight=True, live_performance_not_yet_known=True)
    write_x(output/"PROSPECTIVE.json", record)
    verify_prospective(output)
    return {"status": "RUNTIME_PREPARED_NOT_RUN", "output": str(output), "budget": C.BUDGET}


def launch(output, *, execute_token):
    """The exact retained lifecycle with exactly its two command modules changed."""
    source = inspect.getsource(H.launch)
    pairs = (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
             ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION))
    for before, after in pairs:
        if source.count(before) != 1:
            raise ValueError("retained launch seam changed")
        source = source.replace(before, after)
    namespace = dict(H.__dict__, _verify_prospective=verify_prospective,
                     _verify_prerequisite=verify_prerequisite)
    exec(compile(source, "<source-bound-retained-parity-v5-launch>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def postflight(output):
    from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as repaired
    from rl_agent.splitfusion_run5_live_avo_v1 import verify_live_gt_capture as V
    from rl_agent.splitfusion_live_parity_repair_v2.runtime import install_postflight_reader
    verify_prospective(output)
    # Rebind the exact retained function, never H or v1 contract module globals.
    original = H.postflight
    retained = FunctionType(original.__code__, dict(original.__globals__, _verify_prospective=verify_prospective),
                            original.__name__, original.__defaults__, original.__closure__)
    old_verify = V.verify_capture
    V.verify_capture = repaired.verify_avo_capture
    try:
        retained(output)
    finally:
        V.verify_capture = old_verify
    with install_postflight_reader():
        result = repaired.postflight(output)
    from .postflight_checks import verify_optimization_scope
    record = verify_prospective(output)
    try:
        cells = sorted((Path(output)/"capture/cells").iterdir())
        if len(cells) != 1:
            raise ValueError("one qualification cell required")
        optimization = verify_optimization_scope(cells[0], expected_sources=record["optimization_sources"],
            expected_prospective_sha256=C.sha(Path(output)/"PROSPECTIVE.json"))
    except Exception as exc:
        optimization = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    summary = {**result, "optimization":optimization,
               "passed": result.get("passed") is True and optimization.get("passed") is True}
    from .postflight_checks import verify_input_scope
    try:
        inputs = verify_input_scope(cells[0], expected_prospective_sha256=C.sha(Path(output)/"PROSPECTIVE.json"))
    except Exception as exc:
        inputs = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    summary["paired_input"] = inputs
    summary["passed"] = summary["passed"] and inputs.get("passed") is True
    summary["retained_parity_status"] = result.get("status")
    summary["status"] = ("EXACT_INPUT_CAPTURE_ENGINEERING_PASS__PAIRED_REPLAY_PENDING" if summary["passed"]
                         else "STOP_EXACT_INPUT_CAPTURE_ENGINEERING_FAILED")
    write_x(Path(output)/"OPTIMIZED_POSTFLIGHT.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch", "postflight"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--retained-code", type=Path)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--sync-probe", type=Path)
    parser.add_argument("--startup-scope", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if any(x is None for x in (args.worktree, args.retained_code, args.offline, args.sync_probe, args.startup_scope)):
            parser.error("prepare requires worktree, retained-code, offline, sync-probe and startup-scope")
        result = prepare(args.output, args.worktree, args.retained_code,
                         offline=args.offline, sync_probe=args.sync_probe, startup_scope=args.startup_scope)
    elif args.operation == "launch":
        live_passed = launch(args.output, execute_token=args.execute)
        result = {**postflight(args.output), "retained_launch_passed": live_passed}
    else:
        result = postflight(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", args.operation == "prepare") and result.get("retained_launch_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
