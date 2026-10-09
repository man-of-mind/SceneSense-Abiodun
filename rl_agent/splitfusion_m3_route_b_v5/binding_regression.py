"""Focused regression for the v4 wrong-verifier defect and the v5 binding."""
import argparse
import hashlib
import json
from pathlib import Path

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v5"


class OldVerifierSelected(RuntimeError):
    pass


def run(name):
    from rl_agent.splitfusion_m3_route_b_v3 import host as v3_host
    from rl_agent.splitfusion_m3_route_b_v3 import runner as v3_runner
    from rl_agent.splitfusion_m3_route_b_v4 import runner as v4_runner
    from . import runner as repaired
    from .host import expected_binding

    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    called = []
    original_verify = v3_host.verify_prospective
    original_contract = v3_runner.C

    def sentinel(_output):
        called.append("v3.host.verify_prospective")
        raise OldVerifierSelected("v4 delegated install_parent selected the v3 verifier")

    v3_host.verify_prospective = sentinel
    old_exposed = False
    try:
        v4_runner.main(["--registration", "/tmp/m3_v4_wrong_verifier_fixture/registration.json",
                        "--arm", "C2", "--output-root", "/tmp/m3_v4_wrong_verifier_fixture/capture",
                        "--preflight"])
    except OldVerifierSelected:
        old_exposed = called == ["v3.host.verify_prospective"]
    finally:
        v3_host.verify_prospective = original_verify
        v3_runner.C = original_contract
    binding = repaired.binding_identities()
    repaired_ok = binding == expected_binding()
    passed = old_exposed and repaired_ok
    result = dict(
        schema="splitfusion.m3_route_b_v5.runner_binding_regression.v1",
        status="PASS" if passed else "FAIL",
        old_wrong_resolution=dict(runner_module="rl_agent.splitfusion_m3_route_b_v3.runner",
                                  verifier_module="rl_agent.splitfusion_m3_route_b_v3.host"),
        old_wrong_verifier_observed=called,
        repaired_binding=binding,
        sources=C.exact_sources(ROOT),
        live_services_started=False,
        carla_started=False, oai_started=False, live_inference_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out.rglob("*")) if path.is_file() and path.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "old_wrong_verifier_observed": called,
                      "repaired_binding": binding}, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
