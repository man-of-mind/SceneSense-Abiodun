"""Corrected, version-bound assessment of the retained V9 capture (offline only)."""
import argparse
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v9.postflight_p3 import install_p3

from . import binding as B
from .integration import evaluate

SCHEMA = "splitfusion.m3_route_b_v9_offline_v1.corrected_assessment.v1"


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
    return dict(accepted=accepted, numeric_host_exit_status="NOT_CAPTURED",
                source="detached host launch log", rows=len(rows), error_rows=len(errors),
                terminal_count=len(terminals), terminal=terminals[0] if len(terminals) == 1 else None)


def run(arm, out, launch_log, original_verdict):
    arm, out = Path(arm).resolve(strict=True), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bound = B.resolve(arm)
    terminal = launch_terminal(launch_log)
    if not terminal["accepted"]:
        raise ValueError("saved launch log has no unique accepted success terminal")
    original = _json(original_verdict)
    if original.get("verdict") != "STOP":
        raise ValueError("correction requires the preserved original STOP assessment")
    with install_p3() as installed:
        value = evaluate(arm, launch_log=str(Path(launch_log).resolve(strict=True)),
                         launch_terminal=terminal, root=Path(__file__).resolve().parents[2],
                         binding=bound, do_rescore=True)
    assessment = dict(
        schema=SCHEMA, verdict=value.get("verdict"), corrected_complete_assessment=value,
        correction_reason="V9 postflight imported V3 integration constants; the three schedule checks used the "
                          "V3 21.5-dB/controller/trace expectations instead of the verified V9 registration",
        original_assessment=dict(path=str(Path(original_verdict).resolve(strict=True)),
                                 sha256=B.sha(original_verdict), verdict=original.get("verdict"),
                                 reasons=original.get("reasons"), preserved_unchanged=True),
        exit_status=dict(numeric_host_exit_status="NOT_CAPTURED", no_numeric_rc_inferred=True,
                         recorded_success_terminal=terminal),
        binding=B.document(bound), p3=dict(installed=installed.document(),
                                            rule=bound.POSE),
        no_live_services_started=True)
    (out / "VERDICT_corrected.json").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
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
                          numeric_host_exit_status="NOT_CAPTURED"), sort_keys=True))
    return value.get("verdict") == "PASS_M3_FULL_ROUTE_INTEGRATION"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch-log", type=Path, required=True)
    parser.add_argument("--original-verdict", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.arm, args.out, args.launch_log, args.original_verdict) else 3


if __name__ == "__main__":
    raise SystemExit(main())
