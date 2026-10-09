"""Per-arm preparation and launch of the three-arm mode-11 frontier measurement (retained Run-5 host lifecycle).

Identical to ``splitfusion_fixed_action_screen_v2.host`` except: arms are keyed by (mode_id, q_e4); the forced
action must equal the dynamic execution contract's own resolution (anchor or off-anchor) at preparation; a
third screen proof (``screen_identity``: offline lifecycle of the three exact identities through the real
service/scorer/archive with exact re-scoring) is required.


prepare: one fresh attempt root per registered arm, create-only: PREREQUISITE.json, ACCEPTANCE.json,
         SELECTION.json, registration.json (forced action, arm index/order), arm.json,
         compose_partition.yaml and PROSPECTIVE.json. Arms are prepared before any arm is launched.
launch:  the retained ``host_capture.launch`` with only runner/partition modules and the
         prospective/prerequisite verifiers substituted (same pattern as the early-GT check).
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


PROOFS = ("pipeline_offline", "pipeline_rehearsal", "pipeline_container", "screen_offline", "screen_startup",
          "screen_identity")


def check_prerequisites(root, proofs):
    """v2 pipeline proofs (offline suites + parity incl. anchor proof, composed rehearsal, container) and the
    screen proofs (forced seam, startup), each bound to the exact current sources."""
    from rl_agent.splitfusion_exact_gt_live_v2 import contract as E
    delta = C.pipeline_delta(root)
    pipeline = E.exact_sources(root)
    out = dict(schema="splitfusion.fixed_action_screen_v2.prerequisite.v1", pipeline_delta_from_v1=delta)
    out["pipeline_offline"], v = _proof(proofs["pipeline_offline"], dict(status="PASS", failures=0, errors=0,
                                                                         live_services_started=False))
    if v.get("sources") != pipeline or v.get("tests", 0) < 53 or not all(
            p.get("status", "").endswith("EXACT") for p in (v.get("parity") or {}).values()) or set(v.get("parity") or {}) != {
            "step1", "step2", "step3", "anchor"}:
        raise ValueError("v2 offline proof must bind the exact pipeline, all suites and all four exact parity runs")
    out["pipeline_rehearsal"], v = _proof(proofs["pipeline_rehearsal"], dict(status="PASS", all_children_exit_zero=True,
                                                                             live_services_started=False))
    if v.get("sources") != pipeline:
        raise ValueError("v2 composed rehearsal does not bind the exact pipeline")
    out["pipeline_container"], v = _proof(proofs["pipeline_container"], dict(status="PASS", live_services_started=False))
    if v.get("sources") != pipeline:
        raise ValueError("v2 container proof does not bind the exact pipeline")
    out["screen_offline"], v = _proof(proofs["screen_offline"], dict(status="PASS", failures=0, errors=0, live_services_started=False))
    if v.get("sources") != C.exact_sources(root) or v.get("tests", 0) < 12:
        raise ValueError("forced-seam offline proof must bind the exact current screen sources")
    out["screen_identity"], v = _proof(proofs["screen_identity"], dict(status="PASS", live_services_started=False,
                                                                       all_children_exit_zero=True))
    if v.get("sources") != C.exact_sources(root) or sorted(v.get("arms_exercised") or []) != sorted(k for k, _, _ in C.ARMS):
        raise ValueError("identity lifecycle proof must bind the exact sources and exercise all three arms")
    out["screen_startup"], v = _proof(proofs["screen_startup"], dict(passed=True, live_services_started=False, cuda_initialized=False))
    if v.get("sources") != C.source_inventory(root):
        raise ValueError("screen startup proof differs from runtime sources")
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
    if record.get("schema") != "splitfusion.fixed_action_screen_v3.prospective.v1" or record.get("budget") != C.BUDGET:
        raise ValueError("foreign fixed-action registration")
    root = Path(record["worktree"])
    if record.get("runtime_sources") != C.source_inventory(root):
        raise ValueError("complete runtime source inventory drift")
    for name, digest in record["sources"].items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or C.sha(path) != digest:
            raise ValueError("prospective dependency changed: " + name)
    for name, field in (("registration.json", "registration_file_sha256"), ("arm.json", "arm_file_sha256"),
                        ("compose_partition.yaml", "compose_file_sha256"), ("PREREQUISITE.json", "prerequisite_file_sha256"),
                        ("ACCEPTANCE.json", "acceptance_file_sha256"), ("SELECTION.json", "selection_file_sha256")):
        if C.sha(output / name) != record[field]:
            raise ValueError("prospective file drift: " + name)
    reg = json.loads((output / "registration.json").read_bytes())
    if (reg.get("id") != C.ID or reg.get("transmitted_budget") != C.BUDGET or reg.get("child_module") != C.CHILD
            or reg.get("edge_module") != C.EDGE or reg.get("exact_gt_sources") != C.exact_sources(root)
            or reg.get("live_metric_contract") != C.metrics()
            or reg.get("registration_sha256") != C.canonical({k: v for k, v in reg.items() if k != "registration_sha256"})):
        raise ValueError("foreign/tampered fixed-action registration")
    C.check_forced_shape(reg.get("forced_action") or {})
    return record


def prepare(output, *, retained_code, proofs, acceptance, selection, arm):
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
    proof = check_prerequisites(root, proofs)
    acceptance_value = json.loads(Path(acceptance).read_bytes())
    if acceptance_value.get("schema") != "splitfusion.fixed_action_screen_v3.acceptance.v1":
        raise ValueError("registered screen acceptance criteria required")
    selection_value = json.loads(Path(selection).read_bytes())
    if selection_value.get("schema") != "splitfusion.fixed_action_screen_v3.selection.v1":
        raise ValueError("registered selection required")
    order = list(selection_value["arm_order"])
    if order != [k for k, _, _ in C.ARMS] or arm not in order or len(order) > C.MAX_ARMS:
        raise ValueError("arm is not a registered arm or the order differs from the contract")
    entry = next(a for a in selection_value["arms"] if a["arm"] == arm)
    forced = C.check_forced_shape(C.forced_document(entry["forced"]))
    if forced != C.resolve(entry["mode_id"], entry["q_e4"]) or (forced["mode_id"], forced["q_e4"]) != (entry["mode_id"], entry["q_e4"]):
        raise ValueError("forced identity differs from the dynamic execution contract's own resolution")
    PB.verify_packaged_export(root / "rl_agent/splitfusion_run5_live_avo_v1/actor_export", root)
    actor = load_registered_actor()
    if not actor._lower[forced["mode_id"]] <= forced["q_e4"] <= actor._upper[forced["mode_id"]]:
        raise ValueError("forced action outside the frozen actor's registered support")
    template = root / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                       "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
    reg = json.loads(template.read_bytes())
    sources = {str(root / name): digest for name, digest in inventory.items()}
    sources.update({str(retained / name): C.sha(retained / name) for name in ("host_partition.py", "cpu_partition.py", "edge_gate.py")})
    sources.update({entry["path"]: entry["sha256"] for entry in reg["experiment_binds"].values()})
    if any(C.sha(name) != digest for name, digest in sources.items()):
        raise ValueError("retained experiment binding drift")
    run_binding = C.canonical(dict(sources=inventory, actor=actor.boundary_sha256, metrics=C.metrics(), budget=C.BUDGET,
                                   output=str(output), forced_action=forced))
    reg.update(id=C.ID, child_module=C.CHILD, edge_module=C.EDGE, transmitted_budget=C.BUDGET, g1_boundary_budget=C.BUDGET,
        controller_lineage_sha256=C.lineage(root, actor.boundary_sha256, forced),
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
        claim_scope="FIXED_ACTION_SCREEN_ENGINEERING_ONLY", policy_performance_claim=False,
        policy_decision_agreement_gate_claimed=False, historical_quality_not_comparable=True,
        forced_action=forced, arm_key=arm, arm_roles=entry["roles"],
        arm_index=order.index(arm), arm_order=order, selection_sha256=C.sha(selection),
        retained_decision_cycle_boundary_exception=True, pipeline_delta_from_v1=proof["pipeline_delta_from_v1"],
        authorization=selection_value["authorization"], builds_on=selection_value["builds_on"],
        prerequisite_binding=proof, acceptance_sha256=C.sha(acceptance))
    reg["semantics"]["actor_loader"] = "rl_agent.splitfusion_run5_live_avo_v1.audit_actor:load_registered_actor"
    reg["profiler"]["minimum_free_disk_bytes"] = 8_000_000_000
    reg["arms"]["C2"]["execute_token"] = C.token(forced)
    reg["arms"]["C2"]["label"] = (f"Original Run5 actor in shadow, forced mode {forced['mode_id']} q_e4 {forced['q_e4']} "
                                  f"(action_id {forced['action_id']}, {forced['measurement_status']}), early-GT pipeline, "
                                  f"retained CPU partition")
    reg["experiment_binds"].update({"fixed_action_m11_frontier:" + str(i): dict(path=name, sha256=digest)
                                    for i, (name, digest) in enumerate(sorted(sources.items()))})
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = C.canonical(reg)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "PREREQUISITE.json", proof)
    write(output / "ACCEPTANCE.json", acceptance_value)
    write(output / "SELECTION.json", selection_value)
    write(output / "registration.json", reg)
    write(output / "arm.json", dict(arm="treatment", original_cpus=list(range(24)),
        radio_cpus=list(range(4)) + list(range(8, 20)), edge_cpus=list(range(4, 8)) + list(range(20, 24))))
    from rl_agent.splitfusion_numeric_gt_runtime_v1.deploy_live import write_bytes
    write_bytes(output / "compose_partition.yaml", ("services:\n  oai-perception-rx:\n    volumes:\n"
        f"      - {output}:/partition_output:rw\n      - {retained}:/cpu_partition_code:ro\n").encode())
    record = dict(schema="splitfusion.fixed_action_screen_v3.prospective.v1", budget=C.BUDGET, worktree=str(root),
        retained_code=str(retained), sources=sources, runtime_sources=inventory, forced_action=forced,
        registration_file_sha256=C.sha(output / "registration.json"),
        compose_file_sha256=C.sha(output / "compose_partition.yaml"), arm_file_sha256=C.sha(output / "arm.json"),
        prerequisite_file_sha256=C.sha(output / "PREREQUISITE.json"),
        acceptance_file_sha256=C.sha(output / "ACCEPTANCE.json"), selection_file_sha256=C.sha(output / "SELECTION.json"),
        source_acceptance_sha256=C.sha(acceptance), source_selection_sha256=C.sha(selection),
        prepared_before_any_frame=True, no_retry=True, live_performance_not_yet_known=True)
    write(output / "PROSPECTIVE.json", record)
    verify_prospective(output)
    return dict(status="PREPARED_NOT_RUN", output=str(output), budget=C.BUDGET, forced_action=forced,
                token=C.token(forced))


def launch(output, *, execute_token):
    from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H
    source = inspect.getsource(H.launch)
    for before, after in (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
                          ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION)):
        if source.count(before) != 1:
            raise ValueError("retained lifecycle seam drift")
        source = source.replace(before, after)
    namespace = dict(vars(H), _verify_prospective=verify_prospective, _verify_prerequisite=verify_prerequisite)
    exec(compile(source, "<fixed-action-retained-host-lifecycle>", "exec"), namespace)
    return namespace["launch"](output, execute_token=execute_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-code", type=Path)
    for name in PROOFS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--execute", default="")
    args = parser.parse_args()
    if args.operation == "prepare":
        proofs = {k: getattr(args, k) for k in PROOFS}
        print(json.dumps(prepare(args.output, retained_code=args.retained_code, proofs=proofs, acceptance=args.acceptance,
                                 selection=args.selection, arm=args.arm)))
        return 0
    return 0 if launch(args.output, execute_token=args.execute) else 1


if __name__ == "__main__":
    raise SystemExit(main())
