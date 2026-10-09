"""Prepare and launch the sealed M1 static baseline through the retained lifecycle.

The mature retention host owns all proof, partition, actor-hash, lifecycle and
retention-plan checks.  This wrapper binds it to the M1 contract and adds the
fixed-selection and zero-completion declarations before the prepared root is
frozen.  No mutation is permitted after launch begins.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from . import contract as C


@contextmanager
def _bound_host():
    from rl_agent.splitfusion_run5_full_live_retention_v1 import host as retained
    old = retained.C
    retained.C = C
    try:
        yield retained
    finally:
        retained.C = old


def _rewrite_prelaunch(output):
    """Finalize the M1-specific registration before any frame can be launched."""
    output = Path(output)
    reg_path, pro_path = output / "registration.json", output / "PROSPECTIVE.json"
    reg = json.loads(reg_path.read_bytes())
    reg["external_fixed_action"] = dict(C.FIXED_ACTION)
    reg["selection_semantics"] = dict(
        source="EXTERNAL_FIXED_BASELINE", actor_proposal_required=False,
        actor_called=False, actor_state_admission_required=False,
        telemetry="OBSERVATIONAL", typed_admission_refusals="LOG_AND_CONTINUE_FIXED",
        fallback_substitution=False)
    reg["zero_completion_semantics"] = dict(C.ZERO_COMPLETION)
    root = Path(__file__).resolve().parents[2]
    from .source_manifest import LAUNCHER_SOURCES
    from .source_verify import verify_launcher_sources, verify_registered_sources
    launcher_sources = {name: C.sha(root / name) for name in LAUNCHER_SOURCES}
    reg["import_safe_source_verification"] = dict(
        schema="splitfusion.m1_static_live_v1.import_safe_verification.v1",
        dependency_policy="STDLIB_ONLY_NO_REPLAY_OR_GEOMETRY_IMPORT",
        exact_sources=verify_registered_sources(reg["exact_gt_sources"], root),
        launcher_sources=launcher_sources,
        launcher_check=verify_launcher_sources(launcher_sources, root))
    reg["policy"]["runtime_role"] = (
        "deployment identity/startup audit only; actor is not called for M1 static selection")
    reg["arms"]["C2"]["label"] = (
        "M1 external static action 0 (noAE/UINT8/q0/no drop), direct decision admission, "
        "two-frame retained controller, 1,000 transmitted-frame budget, paired retention")
    reg["claim_scope"] = (
        "M1_STATIC_LIVE_MEASUREMENT_WITH_OUTCOME_INDEPENDENT_RETENTION__"
        "ZERO_COMPLETION_REGISTERED__NOT_ACTOR_DRIVEN")
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = C.canonical(reg)
    reg_path.write_text(json.dumps(reg, sort_keys=True, indent=2, allow_nan=False) + "\n")
    prospective = json.loads(pro_path.read_bytes())
    prospective.update(
        registration_file_sha256=C.sha(reg_path),
        fixed_selection_frozen_before_any_frame=True,
        zero_completion_semantics_frozen_before_any_frame=True)
    pro_path.write_text(json.dumps(prospective, sort_keys=True, indent=2, allow_nan=False) + "\n")


def prepare(output, **kwargs):
    with _bound_host() as retained:
        result = retained.prepare(output, **kwargs)
        _rewrite_prelaunch(output)
        retained.verify_prospective(output)
    result.update(fixed_action=dict(C.FIXED_ACTION), zero_completion=dict(C.ZERO_COMPLETION))
    return result


def verify_prospective(output):
    with _bound_host() as retained:
        record = retained.verify_prospective(output)
    reg = json.loads((Path(output) / "registration.json").read_bytes())
    if (reg.get("external_fixed_action") != C.FIXED_ACTION
            or reg.get("zero_completion_semantics") != C.ZERO_COMPLETION
            or reg.get("selection_semantics", {}).get("fallback_substitution") is not False
            or reg.get("selection_semantics", {}).get("actor_called") is not False
            or not reg.get("import_safe_source_verification")):
        raise ValueError("M1 fixed selection/zero-completion registration drift")
    return record


def verify_prerequisite(output, record):
    with _bound_host() as retained:
        return retained.verify_prerequisite(output, record)


def launch(output, *, execute_token):
    verify_prospective(output)
    with _bound_host() as retained:
        return retained.launch(output, execute_token=execute_token)


def main(argv=None):
    with _bound_host() as retained:
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
        print(json.dumps(prepare(args.output, arm=args.arm, producer=args.producer,
            retained_code=args.retained_code, proofs=proof_values,
            acceptance=args.acceptance, budget=args.budget)))
        return 0
    return 0 if launch(args.output, execute_token=args.execute) else 1


if __name__ == "__main__":
    raise SystemExit(main())
