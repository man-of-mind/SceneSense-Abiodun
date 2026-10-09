"""Prepare/launch helpers for one concrete M1 fixed-v2 contract."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path


@contextmanager
def bound_host(C):
    from rl_agent.splitfusion_run5_full_live_retention_v1 import host as retained
    old = retained.C
    retained.C = C
    try:
        yield retained
    finally:
        retained.C = old


def resolved_fixed_action(C):
    from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import (
        load_dynamic_execution_contract)
    profile = load_dynamic_execution_contract().resolve_q_e4(
        C.FIXED_ACTION["mode_id"], C.FIXED_ACTION["q_e4"])
    return dict(action_id=profile.action_id, profile_id=profile.profile_id,
        family=profile.family, quantizer=profile.quantizer, mode_id=profile.mode_id,
        q_e4=profile.q_e4, keep_count=profile.keep_count, drop_count=profile.drop_count,
        measurement_status=profile.measurement_status,
        execution_bundle_sha256=profile.execution_bundle_sha256,
        support=C.FIXED_ACTION["support"])


def rewrite_prelaunch(output, C):
    output = Path(output)
    reg_path, pro_path = output / "registration.json", output / "PROSPECTIVE.json"
    reg = json.loads(reg_path.read_bytes())
    if resolved_fixed_action(C) != C.FIXED_ACTION:
        raise ValueError("dynamic execution contract differs from registered fixed identity")
    reg["external_fixed_action"] = dict(C.FIXED_ACTION)
    reg["selection_semantics"] = dict(
        source="EXTERNAL_FIXED_BASELINE", actor_proposal_required=False,
        actor_called=False, actor_state_admission_required=False,
        telemetry="OBSERVATIONAL", typed_admission_refusals="LOG_AND_CONTINUE_FIXED",
        fallback_substitution=False, clamp_to_actor_support=False)
    reg["zero_completion_semantics"] = dict(C.ZERO_COMPLETION)
    reg["retention_resource_plan"] = dict(C.RETENTION_RESOURCE_PLAN)
    root = Path(__file__).resolve().parents[2]
    from .source_manifest import LAUNCHER_SOURCES
    from .source_verify import verify_launcher_sources, verify_registered_sources
    launcher_sources = {name: C.sha(root / name) for name in LAUNCHER_SOURCES}
    reg["import_safe_source_verification"] = dict(
        schema="splitfusion.m1_fixed_live_v2.import_safe_verification.v1",
        dependency_policy="STDLIB_ONLY_NO_REPLAY_OR_GEOMETRY_IMPORT",
        exact_sources=verify_registered_sources(reg["exact_gt_sources"], root),
        launcher_sources=launcher_sources,
        launcher_check=verify_launcher_sources(launcher_sources, root))
    reg["policy"]["runtime_role"] = (
        "deployment identity/startup audit only; actor is not called for fixed selection")
    reg["arms"]["C2"]["label"] = (
        f"M1 external fixed {C.KEY}; direct decision admission; two-frame retained controller; "
        "1,000 transmitted-frame budget; outcome-independent stride-5 retention")
    reg["claim_scope"] = (
        "M1_INTERNAL_FIXED_LIVE_MEASUREMENT_WITH_OUTCOME_INDEPENDENT_RETENTION__"
        "ZERO_COMPLETION_REGISTERED__NOT_ACTOR_DRIVEN__NOT_EXTERNAL_PAPER_REPRODUCTION")
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = C.canonical(reg)
    reg_path.write_text(json.dumps(reg, sort_keys=True, indent=2, allow_nan=False) + "\n")
    prospective = json.loads(pro_path.read_bytes())
    prospective.update(
        registration_file_sha256=C.sha(reg_path), fixed_profile=C.KEY,
        fixed_selection_frozen_before_any_frame=True,
        zero_completion_semantics_frozen_before_any_frame=True,
        retention_resources_checked_before_any_frame=True)
    pro_path.write_text(json.dumps(prospective, sort_keys=True, indent=2, allow_nan=False) + "\n")


def prepare(C, output, **kwargs):
    with bound_host(C) as retained:
        result = retained.prepare(output, **kwargs)
        rewrite_prelaunch(output, C)
        retained.verify_prospective(output)
    result.update(fixed_action=dict(C.FIXED_ACTION),
                  retention_resource_plan=dict(C.RETENTION_RESOURCE_PLAN),
                  zero_completion=dict(C.ZERO_COMPLETION))
    return result


def verify_prospective(C, output):
    with bound_host(C) as retained:
        record = retained.verify_prospective(output)
    reg = json.loads((Path(output) / "registration.json").read_bytes())
    if (reg.get("external_fixed_action") != C.FIXED_ACTION
            or reg.get("zero_completion_semantics") != C.ZERO_COMPLETION
            or reg.get("retention_resource_plan") != C.RETENTION_RESOURCE_PLAN
            or reg.get("selection_semantics", {}).get("fallback_substitution") is not False
            or reg.get("selection_semantics", {}).get("actor_called") is not False
            or reg.get("selection_semantics", {}).get("clamp_to_actor_support") is not False
            or not reg.get("import_safe_source_verification")):
        raise ValueError("M1 fixed prospective identity drift")
    return record


def launch(C, output, *, execute_token):
    verify_prospective(C, output)
    with bound_host(C) as retained:
        return retained.launch(output, execute_token=execute_token)


def main(C, argv=None):
    with bound_host(C) as retained:
        proofs = retained.PROOFS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-code", type=Path)
    for name in proofs:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--arm", choices=C.ARMS)
    parser.add_argument("--producer", choices=C.PRODUCERS)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        proof_values = {name: getattr(args, name) for name in proofs}
        print(json.dumps(prepare(C, args.output, arm=args.arm, producer=args.producer,
            retained_code=args.retained_code, proofs=proof_values,
            acceptance=args.acceptance, budget=args.budget)))
        return 0
    return 0 if launch(C, args.output, execute_token=args.execute) else 1

