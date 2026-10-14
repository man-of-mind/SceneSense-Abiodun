"""Registered, version-bound postflight assessment of the V20 live-view GOOD_ALL capture (offline only; copy of V17/V11's).

This is the registered evaluator for this revision, not a correction after the fact: the V20 prospective registers
``splitfusion_m3_route_b_v20_offline_v1`` as its postflight assessment binding, and the capture's own numeric host
exit status is supplied (captured at launch, never inferred).

    python3 -B -m rl_agent.splitfusion_m3_route_b_v20_offline_v1.postflight \\
        --arm <prospective root> --out <new dir> --launch-log <launch.log> --launch-rc <N>

The registered M3 verdict tool of the V3 analysis package is deliberately NOT used as the authority here: its
module-global contract supplies V3's 21.5-dB schedule to three ``SCHEDULE_*`` checks, which is the V10 STOP.  It is
still run, unmodified, and its result recorded beside this one for comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v20.postflight_p3 import install_p3

from . import binding as B
from .integration import evaluate

SCHEMA = "splitfusion.m3_route_b_v20_offline_v1.registered_assessment.v1"


def _json(path):
    return json.loads(Path(path).read_bytes())


def launch_terminal(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    terminals = [row for row in rows if row.get("kind") == "capture_finished"]
    errors = [row for row in rows if row.get("kind") in ("error", "exception", "failed")]
    accepted = (len(terminals) == 1 and terminals[0].get("passed") is True
                and terminals[0].get("verdict") == "CAPTURE_VALID" and not errors
                and rows and rows[-1] is terminals[0])
    return dict(accepted=accepted, source="host launch log", rows=len(rows), error_rows=len(errors),
                terminal_count=len(terminals), terminal=terminals[0] if len(terminals) == 1 else None)


def run(arm, out, launch_log, launch_rc, reference_verdict=None):
    arm, out = Path(arm).resolve(strict=True), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bound = B.resolve(arm)
    terminal = launch_terminal(launch_log)
    with install_p3() as installed:
        value = evaluate(arm, launch_log=str(Path(launch_log).resolve(strict=True)),
                         launch_terminal=terminal, launch_returncode=int(launch_rc),
                         root=Path(__file__).resolve().parents[2], binding=bound, do_rescore=True)
    reference = None
    if reference_verdict is not None and Path(reference_verdict).exists():
        recorded = _json(reference_verdict)
        reference = dict(path=str(Path(reference_verdict).resolve(strict=True)), sha256=B.sha(reference_verdict),
                         verdict=recorded.get("verdict"), reasons=recorded.get("reasons"),
                         note="the registered V3-pinned M3 verdict tool, recorded for comparison only; its "
                              "SCHEDULE_* checks read V3's 21.5-dB constants, not this capture's registration")
    assessment = dict(
        schema=SCHEMA, verdict=value.get("verdict"), complete_assessment=value,
        authority="the registration-bound assessment of this revision (prospective launcher_binding.postflight)",
        exit_status=dict(numeric_host_exit_status=int(launch_rc), captured_at_launch=True,
                         no_numeric_rc_inferred=True, recorded_success_terminal=terminal),
        binding=B.document(bound), p3=dict(installed=installed.document(), rule=bound.POSE),
        v3_pinned_verdict_tool=reference, no_live_services_started=True)
    (out / "VERDICT_registered_binding.json").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    (out / "ASSESSMENT.json").write_text(json.dumps(assessment, indent=1, sort_keys=True) + "\n")
    identities = dict(B.document(bound)["sources"],
                      offline_binding_file_sha256=B.sha(Path(__file__).with_name("binding.py")),
                      offline_integration_file_sha256=B.sha(Path(__file__).with_name("integration.py")),
                      offline_postflight_file_sha256=B.sha(__file__))
    (out / "SOURCE_IDENTITIES.json").write_text(json.dumps(identities, indent=1, sort_keys=True) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.iterdir()) if p.is_file()}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(verdict=value.get("verdict"), reasons=value.get("reasons"),
                          numeric_host_exit_status=int(launch_rc)), sort_keys=True))
    return value.get("verdict") == "PASS_M3_FULL_ROUTE_INTEGRATION"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch-log", type=Path, required=True)
    parser.add_argument("--launch-rc", type=int, required=True)
    parser.add_argument("--reference-verdict", type=Path, default=None)
    args = parser.parse_args(argv)
    return 0 if run(args.arm, args.out, args.launch_log, args.launch_rc, args.reference_verdict) else 3


if __name__ == "__main__":
    raise SystemExit(main())
