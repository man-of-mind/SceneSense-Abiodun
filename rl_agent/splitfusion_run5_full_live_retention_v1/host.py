"""Preparation and launch of ONE guarded full Run-5 validation with paired-input retention (retention launcher v1:
the gc_v2 host plus the create-only pre-launch retention plan, its digest in the registration/lineage/prospective record,
and a required retention offline proof). Original gc_v2 header:

Preparation and launch of ONE guarded full Run-5 validation with the adopted GC policy and numeric producer v2
(``producer`` = v2; v1 is the registered rollback through the same launcher). Versioned copy of the GC-qualification
host; everything below is otherwise unchanged. Original header:

Per-arm preparation and launch of the paired GT GC live qualification (versioned copy of
``splitfusion_run5_full_live_v1.host``; retained Run-5 host lifecycle, proofs and checks unchanged). The registration
additionally binds the arm (``gt_gc``) and its lineage; both arm registrations are prepared and frozen before the
control arm starts. Budget 1000 only.

prepare: one fresh attempt root per registered run (budget 100 or 1000), create-only: PREREQUISITE.json,
         ACCEPTANCE.json, registration.json, arm.json, compose_partition.yaml and PROSPECTIVE.json.
launch:  the retained ``host_capture.launch`` with only runner/partition modules and the prospective/prerequisite
         verifiers substituted (same pattern as the v2/v3 screens). No forced-action layer.
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
            raise ValueError(f"prerequisite {path.parent.name}: {key}={value.get(key)!r} expected {expected!r}")
    manifest = path.parent / "OUTPUT_SHA256.json"
    for name, digest in json.loads(manifest.read_bytes()).items():
        candidate = path.parent / name
        if candidate.is_symlink() or C.sha(candidate) != digest:
            raise ValueError("prerequisite evidence drift: " + str(candidate))
    return dict(path=str(path), sha256=C.sha(path), manifest_sha256=C.sha(manifest)), value


PROOFS = ("pipeline_offline", "pipeline_rehearsal", "pipeline_container", "archive_capacity", "mask_lookup",
          "input_prep_v2", "full_offline", "full_startup", "gc_service_rehearsal", "producer_v2_parity",
          "retention_offline")
LIFECYCLE = {100: dict(safety_timeout_s=120, child_timeout_s=1200), 1000: dict(safety_timeout_s=600.0, child_timeout_s=1800.0)}


def check_prerequisites(root, proofs):
    """v2 pipeline proofs (offline suites + parity incl. anchor proof, composed rehearsal, container) and the full Run-5
    proofs (focused adapter/state/action-parity suite, startup), each bound to the exact current sources.
    Archive capacity repair v1: the three retained v2 pipeline proofs are reused when they differ from the current
    sources ONLY by the registered original->repaired archive.py/test_archive.py substitution, and then the
    capacity proof (bound to the exact current sources) is required."""
    from rl_agent.splitfusion_exact_gt_live_v2 import contract as E
    from rl_agent.splitfusion_gt_archive_capacity_v1 import contract as AC
    from rl_agent.splitfusion_gt_mask_lookup_v1 import contract as MK
    delta = C.pipeline_delta(root)
    pipeline = E.exact_sources(root)
    out = dict(schema="splitfusion.run5_full_live_retention_v1.prerequisite.v1", pipeline_delta_from_v1=delta)
    reused = []

    def pipeline_bound(value):
        if value.get("sources") == pipeline:
            return True
        if MK.substitution_ok(value.get("sources"), pipeline, root):    # archive repair + mask-lookup transitions
            reused.append(True)
            return True
        return False
    out["pipeline_offline"], v = _proof(proofs["pipeline_offline"], dict(status="PASS", failures=0, errors=0,
                                                                         live_services_started=False))
    if not pipeline_bound(v) or v.get("tests", 0) < 53 or not all(
            p.get("status", "").endswith("EXACT") for p in (v.get("parity") or {}).values()) or set(v.get("parity") or {}) != {
            "step1", "step2", "step3", "anchor"}:
        raise ValueError("v2 offline proof must bind the exact pipeline, all suites and all four exact parity runs")
    out["pipeline_rehearsal"], v = _proof(proofs["pipeline_rehearsal"], dict(status="PASS", all_children_exit_zero=True,
                                                                             live_services_started=False))
    if not pipeline_bound(v):
        raise ValueError("v2 composed rehearsal does not bind the exact pipeline")
    out["pipeline_container"], v = _proof(proofs["pipeline_container"], dict(status="PASS", live_services_started=False))
    if not pipeline_bound(v):
        raise ValueError("v2 container proof does not bind the exact pipeline")
    out["archive_capacity"], v = _proof(proofs["archive_capacity"], dict(status="PASS", failures=0, errors=0,
                                                                         live_services_started=False))
    if (v.get("sources") != pipeline or v.get("repaired") != AC.repaired(root) or v.get("original") != AC.ORIGINAL
            or v.get("max_cumulative_jobs") != AC.MAX_CUMULATIVE_JOBS or (v.get("capacity_measured") or {}).get("jobs") != 4000
            or (v.get("service_rehearsal") or {}).get("status") != "PASS"
            or (v.get("service_rehearsal") or {}).get("limits") != C.limits(1000)):
        raise ValueError("archive capacity proof must bind the exact repaired pipeline at the 1,000-frame limits")
    out["mask_lookup"], v = _proof(proofs["mask_lookup"], dict(status="PASS", failures=0, errors=0,
                                                               live_services_started=False))
    if (v.get("sources") != pipeline or v.get("repaired") != MK.repaired(root) or v.get("original") != MK.ORIGINAL
            or v.get("tests", 0) < 142 or (v.get("service_rehearsal") or {}).get("status") != "PASS"
            or (v.get("service_rehearsal") or {}).get("limits") != C.limits(1000)):
        raise ValueError("mask-lookup proof must bind the exact current pipeline")
    out["input_prep_v2"], v = _proof(proofs["input_prep_v2"], dict(status="PASS", failures=0, errors=0,
                                                                   cuda_bit_exact_49=True, live_services_started=False))
    if v.get("sources") != {name: C.sha(root / name) for name in C.INPUT_SOURCES} or v.get("tests", 0) < 13:
        raise ValueError("input-prep v2 proof must bind the exact current input sources")
    out["archive_capacity_substitution"] = dict(original=AC.ORIGINAL, repaired=AC.repaired(root),
        retained_pipeline_proofs_reused_under_substitution=len(reused))
    out["mask_lookup_substitution"] = dict(original=MK.ORIGINAL, repaired=MK.repaired(root))
    out["full_offline"], v = _proof(proofs["full_offline"], dict(status="PASS", failures=0, errors=0, live_services_started=False))
    if v.get("sources") != C.exact_sources(root) or v.get("tests", 0) < 8 or v.get("gc_tests", 0) < 8:
        raise ValueError("GC-qualification offline proof must bind the exact current sources and both suites")
    out["full_startup"], v = _proof(proofs["full_startup"], dict(passed=True, live_services_started=False, cuda_initialized=False))
    if v.get("sources") != C.source_inventory(root):
        raise ValueError("full live startup proof differs from runtime sources")
    out["gc_service_rehearsal"], v = _proof(proofs["gc_service_rehearsal"], dict(status="PASS", live_services_started=False,
                                                                                 all_children_exit_zero=True))
    if (v.get("sources") != C.exact_sources(root) or v.get("limits") != C.limits(1000)
            or set(v.get("arms") or {}) != {"candidate_v2", "candidate_v1_rollback"}
            or any(a.get("status") != "PASS" for a in v["arms"].values()) or v.get("rollback_outputs_identical") is not True):
        raise ValueError("launcher rehearsal must bind the exact sources and 1,000-frame limits, pass v2 and the v1 rollback")
    out["producer_v2_parity"], v = _proof(proofs["producer_v2_parity"], dict(status="PASS"))
    from rl_agent.splitfusion_numeric_gt_producer_v2.source_v2 import package_sources as P2
    if (v.get("package") != P2() or (v.get("pass_a") or {}).get("identical") != 49
            or (v.get("pass_a") or {}).get("retained_digests") != 49 or (v.get("pass_b") or {}).get("identical") != 49):
        raise ValueError("producer v2 parity proof must bind the exact current v2 package and pass all 49 frames")
    out["retention_offline"], v = _proof(proofs["retention_offline"], dict(status="PASS", failures=0, errors=0,
                                                                         live_services_started=False))
    from rl_agent.splitfusion_paired_retention_v2.observer import SEAM_PINS
    if (v.get("sources") != C.exact_sources(root) or v.get("retention_tests", 0) < 31 or v.get("launcher_tests", 0) < 6
            or v.get("seam_pins") != {name: C.sha(root / name) for name in SEAM_PINS} or v.get("seam_pins") != SEAM_PINS
            or v.get("plan") != C.retention_plan(1000) or (v.get("archive_rescore") or {}).get("exact") is not True):
        raise ValueError("retention offline proof must bind the exact sources, seams, the 1,000-frame plan and the archive rescore")
    return out


def verify_prerequisite(output, record):
    path = Path(output) / "PREREQUISITE.json"
    if C.sha(path) != record["prerequisite_file_sha256"]:
        raise ValueError("prerequisite file drift")
    proof = json.loads(path.read_bytes())
    fresh = check_prerequisites(Path(record["worktree"]), {k: proof[k]["path"] for k in PROOFS})
    if fresh != proof:
        raise ValueError("prerequisite evidence changed")
    return proof


def verify_prospective(output):
    output = Path(output).resolve(strict=True)
    record = json.loads((output / "PROSPECTIVE.json").read_bytes())
    if (record.get("schema") != "splitfusion.run5_full_live_retention_v1.prospective.v1" or record.get("budget") not in C.BUDGETS
            or record.get("arm") not in C.ARMS or record.get("producer") not in C.PRODUCERS):
        raise ValueError("foreign GC-qualification registration")
    root = Path(record["worktree"])
    if record.get("runtime_sources") != C.source_inventory(root):
        raise ValueError("complete runtime source inventory drift")
    for name, digest in record["sources"].items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("prospective dependency changed: " + name)
    for name, field in ((C.RETENTION_PLAN_FILE, "retention_plan_file_sha256"),
                        ("registration.json", "registration_file_sha256"), ("arm.json", "arm_file_sha256"),
                        ("compose_partition.yaml", "compose_file_sha256"), ("PREREQUISITE.json", "prerequisite_file_sha256"),
                        ("ACCEPTANCE.json", "acceptance_file_sha256")):
        if C.sha(output / name) != record[field]:
            raise ValueError("prospective file drift: " + name)
    reg = json.loads((output / "registration.json").read_bytes())
    if (reg.get("id") != C.ID or reg.get("transmitted_budget") != record["budget"] or reg.get("child_module") != C.CHILD
            or reg.get("edge_module") != C.EDGE or reg.get("exact_gt_sources") != C.exact_sources(root)
            or reg.get("live_metric_contract") != C.metrics()
            or reg.get("registration_sha256") != C.canonical({k: v for k, v in reg.items() if k != "registration_sha256"})):
        raise ValueError("foreign/tampered full Run-5 live registration")
    if "forced_action" in reg or reg.get("exact_gt_limits") != C.limits(record["budget"]):
        raise ValueError("full Run-5 live registration carries a forced action or foreign limits")
    if reg.get("gt_gc") != C.gc_registration(record["arm"], record["budget"], record["producer"]) or reg["arms"]["C2"][
            "execute_token"] != C.token(record["arm"], record["budget"], record["producer"]):
        raise ValueError("registration GC arm, limits or token differ from the prospective record")
    from rl_agent.splitfusion_paired_retention_v2 import plan as RP
    plan = RP.load_plan(output / C.RETENTION_PLAN_FILE, record["retention_plan_file_sha256"])
    if (RP.canonical(plan) != RP.canonical(C.retention_plan(record["budget"]))
            or reg.get("paired_retention") != C.retention_registration(record["retention_plan_file_sha256"])):
        raise ValueError("retention plan/registration differ from the contract")
    return record


def prepare(output, *, arm, producer, retained_code, proofs, acceptance, budget):
    from rl_agent.splitfusion_numeric_gt_producer_v1.originals import startup_binding
    from rl_agent.splitfusion_live_parity_repair_v2.prediction_archive import native_prediction_contract
    from rl_agent.splitfusion_gt_archive_capacity_v1 import contract as AC
    from rl_agent.splitfusion_gt_mask_lookup_v1 import contract as MK
    from rl_agent.splitfusion_run5_full_live_v1 import actor as PA
    if arm not in C.ARMS or producer not in C.PRODUCERS:
        raise ValueError("unknown GC arm or producer")
    root = Path(__file__).resolve().parents[2]
    output, retained = Path(output).absolute(), Path(retained_code).resolve(strict=True)
    budget = int(budget)
    if budget not in C.BUDGETS:
        raise ValueError("budget is not a registered frame budget")
    if output.exists() or not output.resolve().is_relative_to(root):
        raise ValueError("new owned output directory required")
    if os.getuid() == 0:
        raise ValueError("host publisher must be the unprivileged user")
    inventory = C.source_inventory(root)
    proof = check_prerequisites(root, proofs)
    acceptance_value = json.loads(Path(acceptance).read_bytes())
    if acceptance_value.get("schema") != "splitfusion.run5_full_live_retention_v1.acceptance.v1" or acceptance_value.get("gc") != dict(
            policy=C.GC_POLICY, safety=C.SAFETY, validation=C.VALIDATION, producer=producer,
            substitutions=list(C.SUBSTITUTIONS) if producer == "v2" else []):
        raise ValueError("registered GC-qualification acceptance criteria required (matching the contract)")
    actor = PA.load_export()
    audit = PA.audit()
    if audit["verdict"] != "PASS":
        raise ValueError("full Run-5 actor audit failed: " + json.dumps(audit["checks"]))
    template = root / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                       "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
    reg = json.loads(template.read_bytes())
    sources = {str(root / name): digest for name, digest in inventory.items()}
    sources.update({str(retained / name): C.sha(retained / name) for name in ("host_partition.py", "cpu_partition.py", "edge_gate.py")})
    sources.update({entry["path"]: entry["sha256"] for entry in reg["experiment_binds"].values()})
    if any(C.sha(name) != digest for name, digest in sources.items()):
        raise ValueError("retained experiment binding drift")
    from rl_agent.splitfusion_paired_retention_v2 import plan as RP
    retention_plan = C.retention_plan(budget)
    plan_sha256 = RP.plan_sha256(retention_plan)
    run_binding = C.canonical(dict(sources=inventory, actor=actor.boundary_sha256, metrics=C.metrics(), budget=budget,
                                   output=str(output), support=PA.SUPPORT_SHA256, gt_gc_arm=arm,
                                   numeric_producer=producer, retention_plan_sha256=plan_sha256))
    reg.update(id=C.ID, child_module=C.CHILD, edge_module=C.EDGE, transmitted_budget=budget, g1_boundary_budget=budget,
        controller_lineage_sha256=C.lineage(root, actor.boundary_sha256, PA.SUPPORT_SHA256, budget, arm, producer,
                                            plan_sha256),
        policy=dict(audit_module="rl_agent.splitfusion_run5_full_live_v1.audit_actor",
            binding_path=f"{C.V1.PACKAGE}/actor_export/ACTOR_BINDING.json",
            weights_path=f"{C.V1.PACKAGE}/actor_export/actor_state_dict.pt",
            expected_weights_sha256=PA.WEIGHTS_SHA256),
        live_metric_contract=C.metrics(), exact_gt_sources=C.exact_sources(root),
        numeric_run_binding=run_binding, numeric_producer_binding=startup_binding()["binding_sha256"],
        numeric_prediction_contract=native_prediction_contract().document(), exact_gt_limits=C.limits(budget),
        exact_gt_socket_name=C.SOCKET_NAME, exact_gt_capability_name=C.CAPABILITY_NAME,
        publisher_uid=os.getuid(), publisher_gid=os.getgid(), lifecycle=dict(LIFECYCLE[budget]),
        frozen_deadline_ms=170, quality_evaluation_inside_deadline=True, network_profile="FAVORABLE_STABLE",
        no_retry=True, no_training=True, online_learning=False,
        claim_scope="PAIRED_INPUT_RETENTION_QUALITY_DIAGNOSTIC__HOST_MEMORY_PRESSURE__NOT_AN_UNINSTRUMENTED_TIMING_RUN",
        gt_gc=C.gc_registration(arm, budget, producer), paired_retention=C.retention_registration(plan_sha256),
        policy_performance_claim=False, support_sha256=PA.SUPPORT_SHA256, selection=PA.SELECTED,
        selection_scope=PA.SELECTION_SCOPE, actor_audit=audit,
        retained_decision_cycle_boundary_exception=True, pipeline_delta_from_v1=proof["pipeline_delta_from_v1"],
        authorization=acceptance_value["authorization"], prerequisite_binding=proof, acceptance_sha256=C.sha(acceptance),
        archive_capacity_repair=dict(id="GT_ARCHIVE_CAPACITY_REPAIR_V1", original=AC.ORIGINAL, repaired=AC.repaired(root),
            max_cumulative_jobs=AC.MAX_CUMULATIVE_JOBS, configured_archive_jobs=4 * C.limits(budget)["max_frames"],
            configured_scorer_frames=4 * C.limits(budget)["max_frames"],
            unchanged="outstanding job/byte credits, back-pressure, duplicate refusal, durability, teardown"),
        ue_input_candidate=dict(id="UE_INPUT_SINGLE_BUFFER_V2", function="splitfusion_ue_input_prep_v2.candidate."
            "prepare_live_input_fused", installed_inside="install_bgr2rgb_candidate", bit_exact_to="v1 candidate, "
            "corrected original and 49 recorded live tensors", unchanged="validation/messages, PIL bilinear resize, float32 "
            "normalization order, radar channels, 16 zero rows, single .to(device); fresh output per call"),
        mask_lookup_candidate=dict(id="GT_MASK_LOOKUP_V1", original=MK.ORIGINAL, repaired=MK.repaired(root),
            change="target-mask allowed-label check: np.unique set-subset -> exact per-uint8 lookup reduction",
            unchanged="type/length/identity/invalid-label checks, exception and messages, GT eligibility, scores"))
    reg["semantics"]["actor_loader"] = "rl_agent.splitfusion_run5_full_live_v1.actor:load_registered_actor"
    reg["profiler"]["minimum_free_disk_bytes"] = 8_000_000_000
    reg["arms"]["C2"]["execute_token"] = C.token(arm, budget, producer)
    reg["arms"]["C2"]["label"] = (f"Frozen full Run-5 actor (seed 43 u10000, 12 modes) drives decisions, budget {budget}, "
                                  f"early-GT pipeline, retained CPU partition, GT GC arm {arm}, numeric producer {producer}, "
                                  f"paired-input retention plan {plan_sha256[:12]}")
    reg["experiment_binds"].update({"run5_full_live_gc:" + str(i): dict(path=name, sha256=digest)
                                    for i, (name, digest) in enumerate(sorted(sources.items()))})
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = C.canonical(reg)
    output.mkdir(parents=True, exist_ok=False)
    if RP.write_plan(output / C.RETENTION_PLAN_FILE, retention_plan) != plan_sha256:
        raise ValueError("retention plan file digest differs from the plan digest")
    write(output / "PREREQUISITE.json", proof)
    write(output / "ACCEPTANCE.json", acceptance_value)
    write(output / "registration.json", reg)
    write(output / "arm.json", dict(arm="treatment", original_cpus=list(range(24)),
        radio_cpus=list(range(4)) + list(range(8, 20)), edge_cpus=list(range(4, 8)) + list(range(20, 24))))
    from rl_agent.splitfusion_numeric_gt_runtime_v1.deploy_live import write_bytes
    write_bytes(output / "compose_partition.yaml", ("services:\n  oai-perception-rx:\n    volumes:\n"
        f"      - {output}:/partition_output:rw\n      - {retained}:/cpu_partition_code:ro\n").encode())
    record = dict(schema="splitfusion.run5_full_live_retention_v1.prospective.v1", budget=budget, arm=arm, producer=producer,
        worktree=str(root),
        retained_code=str(retained), sources=sources, runtime_sources=inventory,
        registration_file_sha256=C.sha(output / "registration.json"),
        compose_file_sha256=C.sha(output / "compose_partition.yaml"), arm_file_sha256=C.sha(output / "arm.json"),
        prerequisite_file_sha256=C.sha(output / "PREREQUISITE.json"),
        acceptance_file_sha256=C.sha(output / "ACCEPTANCE.json"), source_acceptance_sha256=C.sha(acceptance),
        retention_plan_file_sha256=C.sha(output / C.RETENTION_PLAN_FILE),
        prepared_before_any_frame=True, no_retry=True, live_performance_not_yet_known=True)
    write(output / "PROSPECTIVE.json", record)
    verify_prospective(output)
    return dict(status="PREPARED_NOT_RUN", output=str(output), budget=budget, arm=arm, producer=producer,
                token=C.token(arm, budget, producer), retention_plan_sha256=plan_sha256)


def launch(output, *, execute_token):
    from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H
    source = inspect.getsource(H.launch)
    for before, after in (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
                          ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION)):
        if source.count(before) != 1:
            raise ValueError("retained lifecycle seam drift")
        source = source.replace(before, after)
    namespace = dict(vars(H), _verify_prospective=verify_prospective, _verify_prerequisite=verify_prerequisite)
    exec(compile(source, "<gc-qualification-retained-host-lifecycle>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-code", type=Path)
    for name in PROOFS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--arm", choices=C.ARMS)
    parser.add_argument("--producer", choices=C.PRODUCERS)
    parser.add_argument("--execute", default="")
    args = parser.parse_args()
    if args.operation == "prepare":
        proofs = {k: getattr(args, k) for k in PROOFS}
        print(json.dumps(prepare(args.output, arm=args.arm, producer=args.producer, retained_code=args.retained_code, proofs=proofs,
                                 acceptance=args.acceptance, budget=args.budget)))
        return 0
    return 0 if launch(args.output, execute_token=args.execute) else 1


if __name__ == "__main__":
    raise SystemExit(main())
