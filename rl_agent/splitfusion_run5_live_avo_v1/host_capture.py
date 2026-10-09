"""One bounded, prospectively sealed Run-5 live+AVO capture. No retries.

The retained CPU-affinity ownership/restoration implementation is reused.
This is not the old A/B experiment and never launches its control arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import actor_binding as A
from . import prepare_binding as P
from .quality_adapter import live_metrics_contract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


POSTFLIGHT_CHECKS = frozenset(("profiler_g1_g9", "retained_e1_e5", "actor_audits",
    "causal_snr", "exact_live_gt_avo", "partition_applied", "partition_monitored",
    "partition_restored", "restore_before_edge_stop", "restore_before_radio_stop",
    "prospective_sources_unchanged"))


def _verify_prospective(output):
    output = Path(output).resolve()
    record = json.loads((output / "PROSPECTIVE.json").read_text())
    if not record.get("sources") or not all(sha(p) == h for p, h in record["sources"].items()):
        raise ValueError("Prospectively bound source drift")
    package = Path(record["worktree"]) / "rl_agent/splitfusion_run5_live_avo_v1"
    expected = {str(p.resolve()) for p in package.glob("*.py")}
    recorded = {str(Path(p).resolve()) for p in record["sources"]
                if Path(p).parent == package and Path(p).suffix == ".py"}
    if expected != recorded:
        raise ValueError("Prospectively bound package source set drift")
    for name, key in (("registration.json", "registration_file_sha256"),
                      ("compose_partition.yaml", "compose_file_sha256"), ("arm.json", "arm_file_sha256"),
                      ("PREREQUISITE.json", "prerequisite_file_sha256")):
        if not record.get(key) or sha(output / name) != record[key]:
            raise ValueError(f"Prospective {name} drift")
    return record


def _probe_prerequisite(path, worktree):
    if path is None:
        raise ValueError("300 frames require a successful exact-source CARLA sensor-sync probe")
    from .sync_probe import verify_probe
    path = Path(path).resolve()
    verified = verify_probe(path, source_root=Path(worktree))
    if (verified.get("passed") is not True or verified.get("complete_ticks", 0) < 200
            or verified.get("scored_exact_frames", 0) < 180):
        raise ValueError("Sensor-sync probe did not pass its source-bound verification")
    return {"kind": "EXACT_SOURCE_SENSOR_SYNC_PROBE", "path": str(path),
            "report_sha256": sha(path), "verification": verified}


def _qualification_prerequisite(path, worktree):
    if path is None:
        raise ValueError("1000 frames require a hash-bound successful new Run5 qualification")
    path = Path(path).resolve()
    source = _verify_prospective(path)
    if source["budget"] != 300 or Path(source["worktree"]).resolve() != Path(worktree).resolve():
        raise ValueError("Qualification must be a 300-frame capture from this runtime worktree")
    _verify_prerequisite(path, source)
    report_path, inputs_path = path / "RUN5_POSTFLIGHT.json", path / "RUN5_POSTFLIGHT_INPUTS.json"
    report = json.loads(report_path.read_text())
    if (report.get("passed") is not True or report.get("status") != "RUN5_LIVE_AVO_QUALIFICATION_PASS"
            or set(report.get("checks", {})) != POSTFLIGHT_CHECKS
            or any(value is not True for value in report["checks"].values())):
        raise ValueError("Prior Run5 qualification has no complete successful postflight")
    if sha(inputs_path) != report.get("input_manifest_sha256"):
        raise ValueError("Qualification input manifest drift")
    inputs = json.loads(inputs_path.read_text())
    if not inputs.get("files"):
        raise ValueError("Qualification input manifest is empty")
    for name, digest in inputs["files"].items():
        candidate = (path / name).resolve()
        if not candidate.is_relative_to(path) or sha(candidate) != digest:
            raise ValueError(f"Qualification evidence drift: {name}")
    for mandatory in ("capture/QUALIFICATION_RESULT.json", "SNR_POSTFLIGHT.json", "GT_POSTFLIGHT.json",
                      "AFFINITY_APPLIED.json", "monitor_footer.json", "RESTORATION_LATEST.json"):
        if mandatory not in inputs["files"]:
            raise ValueError(f"Qualification proof lacks {mandatory}")
    return {"kind": "RUN5_LIVE_AVO_300_QUALIFICATION", "path": str(path),
            "report_sha256": sha(report_path), "input_manifest_sha256": sha(inputs_path),
            "source_result_sha256": sha(path / "capture/QUALIFICATION_RESULT.json")}


def _verify_prerequisite(output, record):
    proof = json.loads((Path(output) / "PREREQUISITE.json").read_text())
    actual = (_probe_prerequisite(proof["path"], record["worktree"]) if record["budget"] == 300
              else _qualification_prerequisite(proof["path"], record["worktree"]))
    if actual != proof:
        raise ValueError("Prospectively bound prerequisite changed")
    return proof


def prepare(output, worktree, retained_code, budget, *, sync_probe=None, qualification=None):
    if budget not in (300, 1000):
        raise ValueError("Only bounded 300 qualification / 1000 capture are supported")
    output, worktree, retained_code = (Path(p).resolve() for p in (output, worktree, retained_code))
    package = worktree / "rl_agent/splitfusion_run5_live_avo_v1"
    if output.exists():
        raise FileExistsError(output)
    prerequisite = (_probe_prerequisite(sync_probe, worktree) if budget == 300
                    else _qualification_prerequisite(qualification, worktree))
    P.verify_packaged_export(package / "actor_export", worktree)
    profiler = worktree / "rl_agent/splitfusion_single_gpu_profiler_v1"
    template = worktree / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                          "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
    registration = json.loads(template.read_text())
    sources = list(sorted(package.glob("*.py"))) + list(sorted((package / "actor_export").iterdir()))
    sources += [worktree / name for name in P.FROZEN_DEPENDENCIES]
    sources += [profiler / name for name in ("child_v1_run5_live_avo.py",
                "edge_profiled_v1_cpu_partition_v1_run5_live_avo.py")]
    sources += [retained_code / name for name in ("host_partition.py", "cpu_partition.py", "edge_gate.py")]
    if not all(p.is_file() and not p.is_symlink() for p in sources):
        raise ValueError("Missing/nonregular deployment source")
    token = f"RUN5_LIVE_CORRECTED_METRICS_AVO_{budget}_ONCE_EXECUTE"
    registration.update(id=f"RUN5_LIVE_CORRECTED_METRICS_AVO_{budget}_V1",
        child_module="rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_live_avo",
        transmitted_budget=budget, g1_boundary_budget=budget,
        claim_scope="ORIGINAL_RUN5_LIVE_CORRECTED_SEMANTIC_WITH_EXACT_AVO_NOT_OFFLINE_FCOS_MASK_PARITY",
        policy={"audit_module": "rl_agent.splitfusion_run5_live_avo_v1.audit_actor",
                "binding_path": "rl_agent/splitfusion_run5_live_avo_v1/actor_export/ACTOR_BINDING.json",
                "weights_path": "rl_agent/splitfusion_run5_live_avo_v1/actor_export/actor_state_dict.pt",
                "expected_weights_sha256": A.WEIGHTS_SHA256},
        live_metric_contract=live_metrics_contract(),
        prerequisites={"300": "Exact-source CARLA sensor-sync probe, composed tests and infrastructure preflight",
                       "1000": "Prior 300 Run5 profiler, exact GT/AVO, causal SNR and partition postflight pass"},
        prerequisite_binding=prerequisite,
        additional_acceptance=["Run5 22-D actor replay exact", "Causal admitted SNR receipts",
            "Same-frame synchronized depth/semantic/RGB", "AVO sidecars preserved and identity-bound",
            "Corrected semantic LUT and original AVO scorer unchanged", "Partition ownership/restoration pass"],
        frozen_deadline_ms=170, quality_evaluation_inside_deadline=True,
        network_profile="FAVORABLE_STABLE", historical_run4_quality_not_comparable=True)
    registration["semantics"]["actor_loader"] = "rl_agent.splitfusion_run5_live_avo_v1.audit_actor:load_registered_actor"
    registration["profiler"]["minimum_free_disk_bytes"] = 8_000_000_000
    registration["arms"]["C2"]["execute_token"] = token
    registration["arms"]["C2"]["label"] = "Run5: retained UE cap + CPU radar + disjoint radio/edge partition + live metrics/AVO"
    registration["experiment_binds"].update({f"run5:{i}:{p.name}": {"path": str(p), "sha256": sha(p)}
                                             for i, p in enumerate(sources)})
    registration.pop("registration_sha256", None)
    registration["registration_sha256"] = canonical(registration)
    output.mkdir(parents=False, exist_ok=False)
    write(output / "PREREQUISITE.json", prerequisite)
    write(output / "registration.json", registration)
    write(output / "arm.json", {"arm": "treatment", "original_cpus": list(range(24)),
          "radio_cpus": list(range(4)) + list(range(8, 20)),
          "edge_cpus": list(range(4, 8)) + list(range(20, 24))})
    with (output / "compose_partition.yaml").open("x") as handle:
        handle.write("services:\n  oai-perception-rx:\n    volumes:\n"
                     f"      - {output}:/partition_output:rw\n"
                     f"      - {retained_code}:/cpu_partition_code:ro\n")
    write(output / "PROSPECTIVE.json", {"budget": budget, "worktree": str(worktree),
          "retained_code": str(retained_code), "sources": {str(p): sha(p) for p in sources},
          "registration_file_sha256": sha(output / "registration.json"),
          "compose_file_sha256": sha(output / "compose_partition.yaml"),
          "arm_file_sha256": sha(output / "arm.json"), "prepared_before_any_frame": True,
          "prerequisite_file_sha256": sha(output / "PREREQUISITE.json"),
          "no_retraining": True, "no_retry": True, "no_heavy_nsight": True,
          "live_performance_not_yet_known": True})
    return {"status": "PREPARED_NOT_RUN", "output": str(output), "budget": budget}


def launch(output, *, execute_token):
    output = Path(output).resolve()
    record = _verify_prospective(output)
    _verify_prerequisite(output, record)
    registration = json.loads((output / "registration.json").read_text())
    if execute_token != registration["arms"]["C2"]["execute_token"]:
        raise ValueError("Wrong explicit live execute token")
    worktree, retained = Path(record["worktree"]), Path(record["retained_code"])
    if any(os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "LD_PRELOAD")):
        raise ValueError("Parent environment must not carry numerical caps/injection")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise ValueError("Live CARLA/UE need GPU visibility, not the offline empty mask")
    if (output / "controller_events.jsonl").exists() or (output / "capture").exists():
        raise FileExistsError("Attempt is create-only and cannot be retried")
    def event(kind, **fields):
        row = {"kind": kind, "raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), **fields}
        with (output / "controller_events.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    def command(argv, log, timeout=60):
        with (output / log).open("x") as handle:
            result = subprocess.run(argv, cwd=worktree, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=timeout, env=env)
        if result.returncode:
            raise RuntimeError(f"Command failed {result.returncode}: {log}")
    env = dict(os.environ, SCENESENSE_PARTITION_OUTPUT=str(output),
               SCENESENSE_PARTITION_CODE=str(retained), PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    argv = [sys.executable, "-m", "rl_agent.splitfusion_run5_live_avo_v1.runner",
            "--registration", str(output / "registration.json"), "--arm", "C2",
            "--output-root", str(output / "capture")]
    command([*argv, "--preflight"], "preflight.log", 180)
    # The unchanged runner performs the authoritative cold-host guard before
    # setup. Do not replace it with a weaker process-name approximation here.
    write(output / "launch_environment.json", {"cuda_visible_devices": "UNSET",
          "parent_omp": env.get("OMP_NUM_THREADS"), "parent_mkl": env.get("MKL_NUM_THREADS")})
    monitor = app = None
    log = (output / "runner.log").open("x")
    try:
        app = subprocess.Popen([*argv, "--execute", execute_token], cwd=worktree, env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        event("launch", pid=app.pid)
        deadline = time.monotonic() + 300
        while not (output / "worker_ready.json").is_file():
            if app.poll() is not None:
                raise RuntimeError("Runner ended before edge READY")
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded edge READY wait")
            time.sleep(.1)
        command(["sudo", "-n", sys.executable, "-m", "rl_agent.splitfusion_run5_live_avo_v1.partition_apply",
                 str(output), str(retained)], "apply.log", 35)
        info = json.loads((output / "AFFINITY_APPLIED.json").read_text())
        if info.get("passed") is not True:
            raise RuntimeError("Retained partition did not converge")
        monitor = subprocess.Popen(["sudo", "-n", sys.executable, str(retained / "host_partition.py"),
                         "monitor", str(output)], stdout=(output / "monitor.log").open("x"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        write(output / "release.json", {"expected_affinity": info["expected_affinity"]})
        deadline = time.monotonic() + 1000
        last = time.monotonic()
        while app.poll() is None:
            if (output / "SCOPE_VIOLATION.json").exists():
                raise RuntimeError("CPU partition scope violation")
            if time.monotonic() > deadline:
                raise TimeoutError("Bounded route lifecycle")
            if time.monotonic() - last > 20:
                event("running", edge_ready=(output / "ready_released.json").is_file())
                last = time.monotonic()
            time.sleep(.25)
        if app.returncode:
            raise RuntimeError(f"Capture runner exit {app.returncode}")
        (output / "collection_stop").touch()
        if monitor.wait(15) != 0:
            raise RuntimeError("Partition monitor failed")
        for name in ("monitor_footer.json", "RESTORATION_LATEST.json"):
            if json.loads((output / name).read_text()).get("passed") is not True:
                raise RuntimeError(f"Partition verification failed: {name}")
        for name in ("prestop_restore_command.json", "radio_prestop_restore_command.json"):
            if json.loads((output / name).read_text())["returncode"] != 0:
                raise RuntimeError(f"Pre-stop restoration failed: {name}")
        result = json.loads((output / "capture/QUALIFICATION_RESULT.json").read_text())
        event("capture_finished", passed=result["passed"], verdict=result["verdict"],
              note="Additional new Run5/SNR/AVO evidence checks still required")
        return result["passed"]
    except BaseException as exc:
        event("failure", error=repr(exc))
        (output / "collection_stop").touch()
        if (output / "PARTITION_JOURNAL.json").is_file():
            result = subprocess.run(["sudo", "-n", sys.executable, str(retained / "host_partition.py"),
                                     "restore", str(output)], timeout=30, capture_output=True, text=True)
            write(output / "failure_restore.json", {"returncode": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr})
        if app is not None and app.poll() is None:
            os.killpg(app.pid, signal.SIGINT)
            try:
                app.wait(180)
            except subprocess.TimeoutExpired:
                event("cleanup_timeout", pid=app.pid)
        raise
    finally:
        log.close()
        if monitor is not None and monitor.poll() is None:
            (output / "collection_stop").touch()
            monitor.wait(20)


def postflight(output):
    """Read-only capture checks, followed by create-only proof files. No services.

    A profiler PASS alone never qualifies the new actor/GT/SNR integration.
    Raw capture inputs are hash-bound after all readers finish; the 1,000-frame
    prerequisite subsequently checks these exact files, not an unchecked flag.
    """
    from .verify_snr_capture import verify_capture as verify_snr
    from .verify_live_gt_capture import verify_capture as verify_gt
    output = Path(output).resolve()
    outputs = ("SNR_POSTFLIGHT.json", "GT_POSTFLIGHT.json", "RUN5_POSTFLIGHT_INPUTS.json", "RUN5_POSTFLIGHT.json")
    if any((output / name).exists() for name in outputs):
        raise FileExistsError("Postflight is create-only; preserve the prior result")
    record = _verify_prospective(output)
    capture = output / "capture"
    result = json.loads((capture / "QUALIFICATION_RESULT.json").read_text())
    errors = []
    checks = {key: False for key in POSTFLIGHT_CHECKS}
    checks["prospective_sources_unchanged"] = True
    gates = result.get("gates", {})
    checks["profiler_g1_g9"] = (result.get("passed") is True and len(gates) == 9
                                and all(value is True for value in gates.values()))
    extra = result.get("threadcap", {}).get("checks", {})
    checks["retained_e1_e5"] = (len(extra) == 5 and all(value is True for value in extra.values()))
    audits = result.get("audits", {})
    checks["actor_audits"] = all(audits.get(role, {}).get("verdict") == "PASS"
                                  and audits.get(role, {}).get("returncode") == 0
                                  for role in ("before", "after"))
    for name, function, key in (("SNR_POSTFLIGHT.json", verify_snr, "causal_snr"),
                                ("GT_POSTFLIGHT.json", verify_gt, "exact_live_gt_avo")):
        try:
            document = function(capture)
            checks[key] = (document.get("passed") is True if key == "causal_snr" else
                           document.get("status") == "LIVE_GT_AVO_INTEGRATION_VERIFIED")
        except Exception as exc:
            document = {"passed": False, "status": "VERIFICATION_FAILED", "error": repr(exc)}
            errors.append({"check": key, "error": repr(exc)})
        write(output / name, document)
    for name, key, field, expected in (
        ("AFFINITY_APPLIED.json", "partition_applied", "passed", True),
        ("monitor_footer.json", "partition_monitored", "passed", True),
        ("RESTORATION_LATEST.json", "partition_restored", "passed", True),
        ("prestop_restore_command.json", "restore_before_edge_stop", "returncode", 0),
        ("radio_prestop_restore_command.json", "restore_before_radio_stop", "returncode", 0)):
        try:
            value = json.loads((output / name).read_text()).get(field)
            checks[key] = type(value) is type(expected) and value == expected
        except (OSError, ValueError) as exc:
            errors.append({"check": key, "error": repr(exc)})
    paths = list(capture.rglob("*")) + [output / name for name in (
        "SNR_POSTFLIGHT.json", "GT_POSTFLIGHT.json", "PROSPECTIVE.json", "PREREQUISITE.json",
        "registration.json", "arm.json", "compose_partition.yaml", "AFFINITY_APPLIED.json",
        "monitor_footer.json", "RESTORATION_LATEST.json", "prestop_restore_command.json",
        "radio_prestop_restore_command.json")]
    if any(path.is_symlink() for path in paths):
        raise ValueError("Postflight refuses symlinked capture evidence")
    files = {str(path.relative_to(output)): sha(path) for path in sorted(set(paths)) if path.is_file()}
    write(output / "RUN5_POSTFLIGHT_INPUTS.json", {"schema": "splitfusion.run5.live_postflight_inputs.v1",
          "files": files})
    passed = all(checks.values())
    status = ("RUN5_LIVE_AVO_QUALIFICATION_PASS" if record["budget"] == 300 else
              "RUN5_LIVE_AVO_CAPTURE_VALID") if passed else "STOP_RUN5_LIVE_AVO_INTEGRATION_FAILED"
    report = {"schema": "splitfusion.run5.live_postflight.v1", "passed": passed, "status": status,
              "budget": record["budget"], "checks": checks, "errors": errors,
              "input_manifest_sha256": sha(output / "RUN5_POSTFLIGHT_INPUTS.json"),
              "source_result_sha256": sha(capture / "QUALIFICATION_RESULT.json"),
              "policy_performance_claim": False, "no_retry": True}
    write(output / "RUN5_POSTFLIGHT.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch", "postflight"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--retained-code", type=Path)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--sync-probe", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args()
    if args.operation == "prepare":
        if args.worktree is None or args.retained_code is None:
            parser.error("prepare requires --worktree and --retained-code")
        print(json.dumps(prepare(args.output, args.worktree, args.retained_code, args.budget,
                                sync_probe=args.sync_probe, qualification=args.qualification), indent=2))
    elif args.operation == "launch":
        launch_passed = launch(args.output, execute_token=args.execute)
        report = postflight(args.output)
        print(json.dumps(report, indent=2))
        return 0 if launch_passed and report["passed"] else 1
    else:
        report = postflight(args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
