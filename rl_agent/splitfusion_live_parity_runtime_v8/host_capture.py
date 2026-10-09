"""One newly authorized 100-frame attempt; fresh offline and image proof gates."""
import argparse
import inspect
import json
import os
from pathlib import Path
from types import FunctionType
from . import contract as C
from rl_agent.splitfusion_live_parity_runtime_v1.entries import write_x
from rl_agent.splitfusion_live_parity_runtime_v1.deploy import CPU_FIXTURE_FILES
from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H

OFFLINE_SUITES = frozenset(("integration_v8", "integration_v7", "handoff_v1", "optimization_v5", "optimization_v6", "optimization_v3", "optimization_v4", "integration_v6", "paired_replay_v1", "paired_replay_v2", "integration_v5", "optimization_v2", "integration_v4", "optimization", "integration_v3", "integration_v2", "integration_v1", "repair_v2", "repair", "existing_run5", "frozen_semantics"))
ALLOWED_SKIP = dict(id="rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_prewarm_gt_priority_v2."
                   "AddendumTest.test_addendum_and_manifest_bind_prior_attempts_and_are_not_executed",
                   reason="prior evidence not present on this host")
from .container_check import CHECKS as _STARTUP_CHECKS
STARTUP_CHECKS = frozenset(_STARTUP_CHECKS)
WORKER_CLIENT_FACTORY = "rl_agent.splitfusion_live_parity_runtime_v8.worker_observer.ObservedClient"
HANDOFF_MANIFEST_SHA256 = "77a3f9c810f792af4e7d0b28258c05b97f4d5e37a3262093c12c4c7d4577055d"


def publisher_binding():
    """Seal the real host identity, never a container-inferred default."""
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise ValueError("registered publisher must be the unprivileged host user")
    return dict(publisher_uid=uid, publisher_gid=gid, socket_mode="0600",
                publisher_authentication="SO_PEERCRED_BEFORE_APPLICATION_READ")



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
    if (value.get("schema") != "splitfusion.parity_runtime.offline.v8"
            or value.get("passed") is not True or value.get("live_started") is not False
            or value.get("training_started") is not False
            or value.get("sources") != C.source_inventory(worktree)):
        raise ValueError("fresh full v8 offline source-bound verification required")
    suites = value.get("suites", {})
    if set(suites) != OFFLINE_SUITES:
        raise ValueError("full registered-suite offline verification required")
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
        raise ValueError("all registered suite logs required")
    fixtures = {name: digest for name, (_size, digest) in CPU_FIXTURE_FILES.items()}
    if value.get("fixtures") != fixtures or any((Path(worktree)/name).is_symlink()
            or C.sha(Path(worktree)/name) != digest for name, digest in fixtures.items()):
        raise ValueError("offline verification finite-fixture drift")
    return {"path": str(path), "sha256": C.sha(path), "passed": True}


def verify_startup_scope(path, worktree):
    path, value = _proof(path)
    checks = value.get("checks", {})
    if (value.get("schema") != "splitfusion.parity_runtime.edge_startup_scope.v8"
            or value.get("passed") is not True or value.get("live_started") is not False
            or value.get("cuda_initialized") is not False or value.get("image_id") != C.IMAGE_ID
            or value.get("edge_module") != C.EDGE or value.get("sources") != C.source_inventory(worktree)
            or value.get("publisher") != publisher_binding() or value.get("errors") != []
            or value.get("sources_unchanged") is not True or value.get("ready_exists") is not False
            or set(checks) != STARTUP_CHECKS or any(flag is not True for flag in checks.values())):
        raise ValueError("fresh pinned-image source-bound edge startup-scope proof required")
    _verify_logs(path, value)
    return {"path": str(path), "sha256": C.sha(path), "passed": True,
            "image_id": C.IMAGE_ID, "edge_module": C.EDGE, "publisher": publisher_binding()}


def verify_worker_proof(path, worktree):
    # The old 288-replay proof still binds the exact retained worker implementation.
    from rl_agent.splitfusion_live_parity_runtime_v7.host_capture import verify_worker_proof as original
    return original(path, worktree)


def verify_handoff_proof(path, worktree):
    """Bind the independently sealed final handoff proof without editing it."""
    path, report = _proof(path)
    manifest_path, document = _proof(path.parent / "OUTPUT_SHA256.json")
    manifest = document.get("files") if isinstance(document, dict) else None
    if (C.sha(manifest_path) != HANDOFF_MANIFEST_SHA256
            or not isinstance(manifest, dict) or not manifest
            or report.get("schema") != "splitfusion.gt_scoring_before_archive.result.v1"
            or report.get("status") != "OFFLINE_STRUCTURAL_HANDOFF_EXACT_PASS__NOT_LIVE_QUALIFIED"
            or report.get("exact_full_replays") != 192 or report.get("unique_frames") != 48
            or report.get("total_tests") != 325 or report.get("live_runs") != 0
            or report.get("cuda_initialized") is not False or report.get("actor_retrained") is not False
            or report.get("deadline_success_improvement_established") is not False
            or report.get("source_binding") != C.handoff_binding(worktree)):
        raise ValueError("fixed final-code handoff proof required")
    proof_root = path.parents[3]
    if path.parents[2].name != "experiments" or str(path.relative_to(proof_root)) not in manifest:
        raise ValueError("handoff proof repository layout differs")
    for name, digest in manifest.items():
        relative = Path(name)
        candidate = proof_root / relative
        if (relative.is_absolute() or ".." in relative.parts or candidate.is_symlink()
                or not candidate.is_file() or not candidate.resolve().is_relative_to(proof_root)
                or any(parent.is_symlink() for parent in candidate.parents if parent.is_relative_to(proof_root))
                or C.sha(candidate) != digest):
            raise ValueError("handoff proof evidence/source hash drift: " + str(name))
    design_path = path.parent / "full48_final/DESIGN.json"
    if str(design_path.relative_to(proof_root)) not in manifest:
        raise ValueError("final handoff replay design is not sealed")
    design = json.loads(design_path.read_bytes())
    if (design.get("sources", {}).get("candidate") != C.bootstrap_inventory(worktree)
            or design.get("sources", {}).get("body") != C.worker_inventory(worktree)
            or design.get("binding") != C.handoff_binding(worktree)):
        raise ValueError("handoff executable sources differ from exact replay")
    return dict(path=str(path), sha256=C.sha(path), output_manifest_sha256=C.sha(manifest_path),
        checked_files=len(manifest), exact_replays=192, total_tests=325,
        source_binding=report["source_binding"], passed=True, live_qualified=False)


def verify_prerequisite(output, record):
    proof = json.loads((Path(output)/"PREREQUISITE.json").read_text())
    if (proof.get("schema") != "splitfusion.parity_runtime.prerequisite.v8"
            or verify_offline(proof["offline"]["path"], record["worktree"]) != proof["offline"]
            or verify_startup_scope(proof["startup_scope"]["path"], record["worktree"]) != proof["startup_scope"]
            or H._probe_prerequisite(proof["sync_probe"]["path"], record["worktree"]) != proof["sync_probe"]
            or verify_worker_proof(proof["worker_proof"]["path"], record["worktree"]) != proof["worker_proof"]
            or verify_handoff_proof(proof["handoff_proof"]["path"], record["worktree"]) != proof["handoff_proof"]):
        raise ValueError("runtime prerequisite changed")
    return proof


def verify_prospective(output):
    from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import source_binding_hashes
    record = H._verify_prospective(output)
    if record.get("schema") != "splitfusion.parity_runtime.prospective.v8" or record.get("budget") != C.BUDGET:
        raise ValueError("foreign parity registration")
    if record.get("runtime_sources") != C.source_inventory(record["worktree"]):
        raise ValueError("complete runtime source set drift")
    if record.get("optimization_sources") != C.optimization_inventory(record["worktree"]):
        raise ValueError("optimization source binding drift")
    if (record.get("worker_sources") != C.worker_inventory(record["worktree"])
            or record.get("worker_timeout_s") != C.WORKER_TIMEOUT_S
            or record.get("worker_client_factory") != WORKER_CLIENT_FACTORY):
        raise ValueError("CPU worker source or timeout binding drift")
    if (record.get("bootstrap_sources") != C.bootstrap_inventory(record["worktree"])
            or record.get("handoff_binding") != C.handoff_binding(record["worktree"])
            or record.get("publisher") != publisher_binding()
            or any(record.get(key) != value for key, value in publisher_binding().items())):
        raise ValueError("handoff bootstrap, source binding or host publisher drift")
    registration = json.loads((Path(output)/"registration.json").read_text())
    if (registration.get("id") != C.ID or registration.get("child_module") != C.CHILD
            or registration.get("edge_module") != C.EDGE or registration.get("transmitted_budget") != C.BUDGET
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("paired_input") != C.INPUT_SPEC
            or registration.get("model_replay_sources") != source_binding_hashes(Path(record["worktree"]))
            or registration.get("optimization_sources") != record["optimization_sources"]
            or registration.get("worker_sources") != record["worker_sources"]
            or registration.get("worker_timeout_s") != record["worker_timeout_s"]
            or registration.get("worker_client_factory") != WORKER_CLIENT_FACTORY
            or registration.get("bootstrap_sources") != record["bootstrap_sources"]
            or registration.get("handoff_binding") != record["handoff_binding"]
            or any(registration.get(key) != value for key, value in record["publisher"].items())
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("foreign/tampered runtime registration")
    return record


def prepare(output, worktree, retained_code, *, offline, sync_probe, startup_scope, worker_proof, handoff_proof):
    from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A, prepare_binding as P
    from rl_agent.splitfusion_run5_live_avo_v1.audit_actor import load_registered_actor
    from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import source_binding_hashes
    output, worktree, retained = (Path(p).resolve() for p in (output, worktree, retained_code))
    if output.exists():
        raise FileExistsError(output)
    if worktree != Path(__file__).resolve().parents[2]:
        raise ValueError("prepare must run from the actual deployed runtime")
    proof = dict(schema="splitfusion.parity_runtime.prerequisite.v8",
        offline=verify_offline(offline, worktree), startup_scope=verify_startup_scope(startup_scope, worktree),
        sync_probe=H._probe_prerequisite(sync_probe, worktree),
        worker_proof=verify_worker_proof(worker_proof, worktree),
        handoff_proof=verify_handoff_proof(handoff_proof, worktree),
        note="Prior exact sensor sync only; new startup proof is CPU-only, not live qualification")
    P.verify_packaged_export(worktree/"rl_agent/splitfusion_run5_live_avo_v1/actor_export", worktree)
    template = worktree / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
        "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
    reg = json.loads(template.read_text())
    inventory = C.source_inventory(worktree)
    optimization = C.optimization_inventory(worktree)
    worker_sources = C.worker_inventory(worktree)
    bootstrap_sources = C.bootstrap_inventory(worktree)
    handoff_binding = C.handoff_binding(worktree)
    publisher = publisher_binding()
    sources = {str(worktree/name): digest for name, digest in inventory.items()}
    sources.update({str(retained/name): C.sha(retained/name)
                    for name in ("host_partition.py", "cpu_partition.py", "edge_gate.py")})
    sources.update({item["path"]: item["sha256"] for item in reg["experiment_binds"].values()})
    if any(C.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("retained partition/source drift")
    actor = load_registered_actor()
    reg.update(id=C.ID, child_module=C.CHILD, edge_module=C.EDGE,
        transmitted_budget=C.BUDGET, g1_boundary_budget=C.BUDGET,
        claim_scope="GT_SCORING_HANDOFF_ENGINEERING_AND_EXACT_INPUT_DIAGNOSTIC_ONLY",
        controller_lineage_sha256=C.lineage(worktree, actor.boundary_sha256),
        policy={"audit_module": "rl_agent.splitfusion_run5_live_avo_v1.audit_actor",
            "binding_path": "rl_agent/splitfusion_run5_live_avo_v1/actor_export/ACTOR_BINDING.json",
            "weights_path": "rl_agent/splitfusion_run5_live_avo_v1/actor_export/actor_state_dict.pt",
            "expected_weights_sha256": A.WEIGHTS_SHA256},
        live_metric_contract=C.metrics(), prerequisite_binding=proof, optimization_sources=optimization,
        worker_sources=worker_sources, worker_timeout_s=C.WORKER_TIMEOUT_S,
        worker_client_factory=WORKER_CLIENT_FACTORY,
        bootstrap_sources=bootstrap_sources, handoff_binding=handoff_binding,
        **publisher,
        gt_transport_semantics="IMMUTABLE_SCORING_PACKET__DURABILITY_DEFERRED__EXACT_NUMERICS_V1",
        target_reference_kind="SCORING_PACKET_SHA256_NOT_DURABLE_READY_SHA256",
        scoring_ready_is_not_archive_durable=True,
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
    reg["arms"]["C2"]["label"] = "Original Run5 + exact scoring packet handoff + deferred archive + retained optimizations"
    reg["experiment_binds"].update({f"parity_v8:{i}": {"path": name, "sha256": digest}
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
    record = dict(schema="splitfusion.parity_runtime.prospective.v8", budget=C.BUDGET,
        worktree=str(worktree), retained_code=str(retained), sources=sources, runtime_sources=inventory,
        optimization_sources=optimization,
        worker_sources=worker_sources, worker_timeout_s=C.WORKER_TIMEOUT_S,
        worker_client_factory=WORKER_CLIENT_FACTORY,
        bootstrap_sources=bootstrap_sources, handoff_binding=handoff_binding, publisher=publisher, **publisher,
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
    exec(compile(source, "<source-bound-retained-parity-v8-launch>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def postflight(output):
    from .postflight_checks import postflight as verify
    return verify(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch", "postflight"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--retained-code", type=Path)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--sync-probe", type=Path)
    parser.add_argument("--startup-scope", type=Path)
    parser.add_argument("--worker-proof", type=Path)
    parser.add_argument("--handoff-proof", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if any(x is None for x in (args.worktree, args.retained_code, args.offline, args.sync_probe, args.startup_scope, args.worker_proof, args.handoff_proof)):
            parser.error("prepare requires worktree, retained-code, offline, sync-probe, startup-scope, worker-proof and handoff-proof")
        result = prepare(args.output, args.worktree, args.retained_code,
                         offline=args.offline, sync_probe=args.sync_probe, startup_scope=args.startup_scope,
                         worker_proof=args.worker_proof, handoff_proof=args.handoff_proof)
    elif args.operation == "launch":
        live_passed = launch(args.output, execute_token=args.execute)
        result = {**postflight(args.output), "retained_launch_passed": live_passed}
    else:
        result = postflight(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", args.operation == "prepare") and result.get("retained_launch_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
