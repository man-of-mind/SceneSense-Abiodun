#!/usr/bin/env python3
"""Profiled single-host Phase-6 cell: one bounded instrumentation qualification.

The lifecycle is the unchanged ``phase6_live_runner_v2.run_one_cell`` (cold
check, qualified OAI launcher, T-tracer, Epic CARLA, packet capture, the
bounded child, reverse teardown, cold postflight, P0-P8).  This runner only:

1. runs the unchanged Phase-6 offline preflight plus profiler preflight
   (Phase-6 sources byte-identical to the base commit, policy identity,
   NVML, free disk);
2. writes the create-only ``PROFILER_CONFIG.json`` the child reads;
3. starts the independent resource sampler before the cell and stops it after
   teardown;
4. points the lifecycle at the profiled child entry point and re-anchors the
   per-cell radio scratch under the hash-bound OAI launcher's own ROOT (the
   same relative namespace; required when running from a worktree);
5. runs the read-only actor audit before and after;
6. evaluates the registered gates from durable evidence, create-only.

``--preflight`` is offline.  The live path needs ``--execute`` with the token.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import threadcap_v1 as TCAP
from .radar_bundle_v1 import COMPONENTS as RB_COMPONENTS

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_single_gpu_profiler_v1"
QUALIFICATION_PATH = Path(__file__).resolve().with_name("qualification_run4_300_v1.json")
ADDENDUM_2_PATH = Path(__file__).resolve().with_name("qualification_addendum_2.json")
CAPTURE_PATH = Path(__file__).resolve().with_name("capture_run4_1000_v1.json")
ATTEMPT_3_RESULT_PATH = Path(__file__).resolve().with_name("QUALIFICATION_ATTEMPT_3_RESULT.json")
REGISTRATION_SCHEMAS = ("scenesense.single_gpu_profiler.qualification.v1",
                        "scenesense.single_gpu_profiler.capture.v1",
                        "scenesense.single_gpu_profiler.experiment.v1")
EXECUTE_TOKEN = "SINGLE_GPU_PROFILER_V1_RUN4_INSTRUMENTATION_QUALIFICATION_300_EXECUTE"
CHILD_MODULE = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1"
SAMPLER_MODULE = "rl_agent.splitfusion_single_gpu_profiler_v1.sampler_v1"
RADIO_SCRATCH_PARENT = Path("experiments/splitfusion_oai_100mhz_4d5u_v1")
G1_BOUNDARY_BUDGET = 300   # addendum 2 default; a registration may set g1_boundary_budget
PROFILER_PROCESS_MARKERS = ("splitfusion_single_gpu_profiler_v1.child_v1",
                            "splitfusion_single_gpu_profiler_v1.edge_profiled_v1",
                            "map_profiled_v1.py",
                            "splitfusion_single_gpu_profiler_v1.sampler_v1")


class ProfilerRunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfilerRunnerError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def verify_addendum_2(path: Path = ADDENDUM_2_PATH) -> dict[str, Any]:
    """The sealed addendum binds the base registration and the preserved failures."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in document.items() if k != "addendum_sha256"}
    require(document.get("id") == "PROFILER_QUALIFICATION_ADDENDUM_2", "foreign addendum")
    require(canonical_sha256(body) == document.get("addendum_sha256"), "addendum seal broken")
    here = Path(path).parent
    amends = document["amends"]
    for name, key in (("qualification", "qualification_sha256"),
                      ("attempts_record", "attempts_record_sha256")):
        require(hashlib.sha256((here / amends[name]).read_bytes()).hexdigest() == amends[key],
                f"addendum-bound {name} drifted")
    return {"id": document["id"], "addendum_sha256": document["addendum_sha256"],
            "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def load_qualification(path: Path = QUALIFICATION_PATH) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    require(document.get("schema") in REGISTRATION_SCHEMAS, "foreign registration document")
    if "registration_sha256" in document:          # sealed experiment registration
        body = {k: v for k, v in document.items() if k != "registration_sha256"}
        require(canonical_sha256(body) == document["registration_sha256"],
                "experiment registration seal broken")
    return document


def verify_experiment_binding(registration: Mapping[str, Any], *,
                              at_commit: Optional[str] = None) -> dict[str, Any]:
    """An experiment registration also pins its offline parity evidence and inputs.

    ``at_commit``: for a *completed* experiment whose tracked source files have
    since moved on, a repo-tracked bind may instead match the file content at
    the commit the experiment ran from (``git show <commit>:<path>``).  A live
    launch never passes it: the working tree must match."""
    here = Path(__file__).resolve().parent
    out = {}
    for name, entry in (registration.get("experiment_binds") or {}).items():
        path = Path(entry["path"])
        if not path.is_absolute():
            path = (here / path) if (here / path).exists() else (ROOT / path)
        require(path.is_file(), f"experiment-bound {name} missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"] and at_commit:
            rel = os.path.relpath(path.resolve(), ROOT)
            shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{at_commit}:{rel}"],
                                   capture_output=True, check=False)
            if shown.returncode == 0:
                digest = hashlib.sha256(shown.stdout).hexdigest()
        require(digest == entry["sha256"], f"experiment-bound {name} drifted")
        out[name] = digest
    return out


def select_arm(registration: Mapping[str, Any], arm: Optional[str]) -> dict[str, Any]:
    arms = registration.get("arms")
    if not arms:
        require(arm is None, "this registration has no arms")
        return {"arm": None, "env": {}, "execute_token": registration.get("execute_token",
                                                                           EXECUTE_TOKEN)}
    require(arm in arms, f"--arm must be one of {sorted(arms)}")
    selected = dict(arms[arm])
    require(str(selected.get("bundle", "production")) == TCAP.BUNDLES[arm],
            "registered arm bundle differs from threadcap_v1")
    require(str(selected.get("ordering", "serial")) == TCAP.ORDERINGS.get(arm, "serial"),
            "registered arm ordering differs from threadcap_v1")
    return {"arm": arm, "env": dict(selected.get("env") or {}),
            "bundle": str(selected.get("bundle", "production")),
            "ordering": str(selected.get("ordering", "serial")),
            "execute_token": str(selected["execute_token"])}


def verify_capture_binding(capture: Mapping[str, Any]) -> dict[str, Any]:
    """A capture may only run on the instrumentation that qualified."""
    binds = capture["binds"]
    here = Path(__file__).resolve().parent
    for name, key in (("qualification_result", "qualification_result_sha256"),
                      ("addendum", "addendum_file_sha256")):
        require(hashlib.sha256((here / binds[name]).read_bytes()).hexdigest() == binds[key],
                f"capture-bound {name} drifted")
    result = json.loads((here / binds["qualification_result"]).read_text(encoding="utf-8"))
    require(result.get("verdict") == "INSTRUMENTATION_QUALIFIED"
            and all((result.get("gates") or {}).values()),
            "capture requires a qualified instrumentation result")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", binds["qualified_commit"],
                               "HEAD"], cwd=str(ROOT)).returncode == 0
    require(ancestor, "qualified commit is not an ancestor of HEAD")
    changed = subprocess.run(["git", "diff", "--name-only", binds["qualified_commit"], "--",
                              *[f"{PACKAGE}/{name}" for name in capture["instrumentation_files"]]],
                             cwd=str(ROOT), capture_output=True, text=True,
                             check=True).stdout.split()
    require(not changed, f"instrumentation changed since qualification: {changed}")
    return {"qualified_commit": binds["qualified_commit"], "qualification_verdict":
            result["verdict"], "instrumentation_unchanged_since_qualification": True}


# ---------------------------------------------------------------------------
# preflight (offline)
# ---------------------------------------------------------------------------


def base_commit_identity(base_commit: str, root: Path = ROOT) -> dict[str, Any]:
    """Every tracked file outside this package equals the base commit."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True,
                          text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "--name-only", "--ignore-submodules=all",
                           base_commit, "--", ".", f":(exclude){PACKAGE}"],
                          cwd=str(root), capture_output=True,
                          text=True, check=True).stdout.split()
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", base_commit, "HEAD"],
                              cwd=str(root)).returncode == 0
    return {"head": head, "base_commit": base_commit, "base_is_ancestor": ancestor,
            "tracked_changes_outside_package": diff,
            "unchanged": ancestor and not diff}


def launcher_root(config: Mapping[str, Any]) -> Path:
    """The fixed ROOT of the hash-bound OAI launcher (read from its bytes)."""
    launcher = ROOT / str(config["runtime"]["oai_registered_profile_launcher"])
    match = re.search(r'^ROOT="([^"]+)"$', launcher.read_text(encoding="utf-8"), re.M)
    require(match is not None, "qualified OAI launcher declares no ROOT")
    root = Path(match.group(1)).resolve(strict=True)
    require((root / RADIO_SCRATCH_PARENT).is_dir(), "launcher radio-state parent is absent")
    return root


def anchored_live_radio_state(supervisor: Any, root: Path) -> Any:
    """Supervisor scratch re-anchored under the launcher ROOT (same namespace)."""
    original = supervisor._live_radio_state

    def _live_radio_state(config, cell, attempt):
        namespace, state = original(config, cell, attempt)
        relative = Path(namespace).resolve().relative_to(Path(supervisor.ROOT).resolve())
        require(relative.parts[:2] == RADIO_SCRATCH_PARENT.parts,
                "supervisor radio scratch left its registered parent")
        anchored = root / relative
        anchored_state = anchored / Path(state).name
        require(not anchored_state.exists(), f"live radio scratch already exists: {anchored_state}")
        return anchored, anchored_state

    _live_radio_state.original = original  # type: ignore[attr-defined]
    return _live_radio_state


def host_contention_state() -> dict[str, Any]:
    """Pre-existing host conditions recorded, never changed: MPS daemons, other
    GPU clients and foreign long-running processes are reported as found."""
    mps, foreign = [], []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            comm = Path(f"/proc/{name}/comm").read_text().strip()
            cmdline = Path(f"/proc/{name}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
        except OSError:
            continue
        if comm.startswith("nvidia-cuda-mps"):
            mps.append({"pid": int(name), "comm": comm, "cmdline": cmdline[:120]})
        elif cmdline.split(" ", 1)[0].rsplit("/", 1)[-1] in ("bfs", "find", "rsync", "tar",
                                                               "du", "updatedb"):
            foreign.append({"pid": int(name), "comm": comm, "cmdline": cmdline[:160]})
    return {"mps_processes": mps, "mps_default_pipe_dir_present": Path("/tmp/nvidia-mps").is_dir(),
            "cuda_mps_pipe_directory_env": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
            "foreign_io_processes": foreign,
            "loadavg": Path("/proc/loadavg").read_text().strip()}


def profiler_preflight(qualification: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    from . import identity_v1 as IV
    from .nvml_v1 import NvmlDeviceV1

    policy = qualification["policy"]
    identity = IV.binding_identity(ROOT / policy["binding_path"], ROOT / policy["weights_path"])
    require(identity["weights_file_sha256_observed"] == policy["expected_weights_sha256"],
            "policy weights differ from the registered SHA-256")
    require(identity["weights_match_declaration"], "binding declaration disagrees with weights")
    commit = base_commit_identity(qualification["base_commit"])
    require(commit["unchanged"], f"tracked sources drifted from the base commit: "
                                 f"{commit['tracked_changes_outside_package'][:8]}")
    nvml = NvmlDeviceV1()
    try:
        gpu = {"name": nvml.name(), "driver": nvml.driver_version(),
               "sample": nvml.device_sample()}
    finally:
        nvml.close()
    require(gpu["name"] == "NVIDIA GeForce RTX 5090", "registered RTX 5090 not visible")
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize
    minimum = int(qualification["profiler"]["minimum_free_disk_bytes"])
    require(free >= minimum, f"free disk {free} below the registered {minimum}")
    return {"policy_identity": identity, "base_commit": commit, "gpu": gpu,
            "free_disk_bytes": free, "host_contention": host_contention_state()}


def preflight(config_path: Path, *, output_root: Path, transmitted_budget: int,
              registration_path: Path = QUALIFICATION_PATH) -> tuple[dict, list, dict, dict]:
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R

    qualification = load_qualification(registration_path)
    require(int(transmitted_budget) == int(qualification["transmitted_budget"]),
            "transmitted budget differs from the registration")
    config, cells, phase6 = R.offline_preflight(config_path, output_root=output_root,
                                                transmitted_budget=int(transmitted_budget))
    report = {"schema": "scenesense.single_gpu_profiler.preflight.v1",
              "qualification_id": qualification["id"],
              "qualification_sha256": hashlib.sha256(QUALIFICATION_PATH.read_bytes()).hexdigest(),
              "registration": {"path": str(registration_path), "kind": qualification.get(
                  "kind", "INSTRUMENTATION_QUALIFICATION"), "sha256": hashlib.sha256(
                  Path(registration_path).read_bytes()).hexdigest()},
              "addendum_2": verify_addendum_2(),
              "capture_binding": (verify_capture_binding(qualification)
                                  if "binds" in qualification else None),
              "experiment_binding": (verify_experiment_binding(qualification)
                                     if "experiment_binds" in qualification else None),
              "child_module": str(qualification.get("child_module", CHILD_MODULE)),
              "phase6": phase6, "profiler": profiler_preflight(qualification,
                                                               output_root=output_root),
              "launcher_root": str(launcher_root(config))}
    importlib.import_module(report["child_module"])          # importable before any launch
    return config, cells, report, qualification


# ---------------------------------------------------------------------------
# live helpers
# ---------------------------------------------------------------------------


def write_create_only(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(document), sort_keys=True, indent=1, default=str) + "\n")


def profiler_config(qualification: Mapping[str, Any], run_label: str) -> dict[str, Any]:
    body = {"schema": "scenesense.single_gpu_profiler.config.v1", "enabled": True,
            "run_label": run_label, "qualification_id": qualification["id"],
            "policy": {k: qualification["policy"][k] for k in ("binding_path", "weights_path")},
            "semantics": dict(qualification["semantics"]),
            "gauge_interval_s": qualification["profiler"]["gauge_interval_s"],
            "flush_interval_s": qualification["profiler"]["flush_interval_s"],
            "sampler_interval_ms": qualification["profiler"]["sampler_interval_ms"]}
    return {**body, "sha256": canonical_sha256(body)}


def run_audit(module: str, path: Path) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run([sys.executable, "-m", module], cwd=str(ROOT), env=env,
                               capture_output=True, text=True, timeout=600)
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(completed.stdout)
    document = json.loads(completed.stdout) if completed.returncode == 0 else {}
    return {"returncode": completed.returncode, "document": document,
            "stderr_tail": completed.stderr[-400:]}


def start_sampler(output_root: Path, interval_ms: float) -> subprocess.Popen:
    directory = output_root / "profiler"
    directory.mkdir(parents=True, exist_ok=False)
    ready = directory / "sampler_ready.json"
    process = subprocess.Popen(
        [sys.executable, "-m", SAMPLER_MODULE, "--out", str(directory / "sampler.jsonl"),
         "--interval-ms", str(float(interval_ms)), "--watch-pid", str(os.getpid()),
         "--role", f"{os.getpid()}=runner", "--ready-file", str(ready)],
        cwd=str(ROOT), stdin=subprocess.DEVNULL,
        stdout=(directory / "sampler_stdout.log").open("x"),
        stderr=(directory / "sampler_stderr.log").open("x"), start_new_session=True)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and not ready.is_file():
        require(process.poll() is None, "resource sampler exited during startup")
        time.sleep(0.05)
    require(ready.is_file(), "resource sampler never became ready")
    return process


def stop_sampler(process: Optional[subprocess.Popen]) -> dict[str, Any]:
    if process is None:
        return {"stopped": True, "started": False}
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
            return {"stopped": True, "killed": True, "returncode": process.returncode}
    return {"stopped": process.poll() is not None, "killed": False,
            "returncode": process.returncode}


def leftover_processes() -> list[dict[str, Any]]:
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            cmdline = Path(f"/proc/{name}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
        except OSError:
            continue
        if any(marker in cmdline for marker in PROFILER_PROCESS_MARKERS):
            found.append({"pid": int(name), "cmdline": cmdline[:200]})
    return found


# ---------------------------------------------------------------------------
# gates (pure over durable evidence)
# ---------------------------------------------------------------------------


def evaluate_gates(*, cell_report: Mapping[str, Any], analysis: Mapping[str, Any],
                   timing: Mapping[str, Any], audits: Mapping[str, Any],
                   expected_weights_sha256: str, sampler_stop: Mapping[str, Any],
                   leftovers: Sequence[Any], transmitted_budget: int,
                   boundary_budget: int = G1_BOUNDARY_BUDGET) -> dict[str, Any]:
    child = dict(cell_report.get("child") or {})
    phase6 = dict((cell_report.get("phase6") or {}).get("gates") or {})
    integration = dict(timing.get("integration") or {})
    counts = dict(timing.get("counts") or {})
    validity = dict(analysis.get("validity") or {})
    identity = dict(analysis.get("identity") or {})
    overhead = dict(analysis.get("overhead") or {})
    semantics = dict(analysis.get("semantics") or {})
    sampler = dict(analysis.get("sampler") or {})
    cleanup = dict(cell_report.get("cleanup") or {})

    before, after = audits.get("before") or {}, audits.get("after") or {}

    def audit_ok(audit: Mapping[str, Any]) -> bool:
        document = audit.get("document") or {}
        return (audit.get("returncode") == 0 and bool(document.get("checks"))
                and all(document["checks"].values())
                and document.get("weights_file_sha256") == expected_weights_sha256)

    def stable(document: Mapping[str, Any]) -> dict:
        return {k: v for k, v in document.items() if k not in ("weights_file",)}

    ue_notes = analysis.get("policy_runtime") or {}
    gates = {
        "G1_FRAME_ACCOUNTING": g1_frame_accounting(
            child=child, timing=timing, analysis=analysis,
            transmitted_budget=int(transmitted_budget),
            boundary_budget=int(boundary_budget))["passed"],
        "G2_ACTOR_HASH_UNCHANGED": (
            audit_ok(before) and audit_ok(after)
            and stable(before.get("document") or {}) == stable(after.get("document") or {})
            and ue_notes.get("weights_before") == expected_weights_sha256
            and ue_notes.get("weights_after") == expected_weights_sha256
            and ue_notes.get("runtime_boundary_matches_binding") is True),
        "G3_ZERO_IDENTITY_CONFLICTS": (
            all(v.get("conflicts") == 0 for v in identity.values())
            and all(identity.get(role, {}).get("foreign_frames") == 0 for role in ("edge", "map"))
            and not integration.get("evaluation_identity_mismatches")
            and not integration.get("feedback_unknown_orphans")
            and not integration.get("duplicate_feedback_frames")
            and phase6.get("P1_IDENTITY") is True),
        "G4_COVERAGE_REPORTED": bool(analysis.get("coverage")) and all(
            "records" in row for row in (analysis.get("coverage") or {}).values()),
        "G5_SAMPLER_REPORTED": (
            sampler.get("present") is True and sampler.get("footer_present") is True
            and (sampler.get("interval_ms") or {}).get("n", 0) > 0
            and sampler.get("dropped_samples") is not None),
        "G6_NO_PROFILER_INDUCED_FAULT": (
            child.get("return_code") == 0 and phase6.get("P8_NO_INFRASTRUCTURE_FAULT") is True
            and not integration.get("faulted")
            and all(_profiler_clean(v) for v in overhead.values())),
        "G7_INSTRUMENTATION_EVIDENCE_VALID": all(v.get("passed") for v in validity.values()),
        "G8_REWARD_TIMEOUT_SEMANTICS": (
            (semantics.get("actor_replay") or {}).get("passed") is True
            and (semantics.get("reward") or {}).get("passed") is True
            and (semantics.get("receipt_marks") or {}).get("passed") is True
            and all(phase6.get(k) is True for k in ("P3_HOLD_TICKET", "P4_ACCOUNTING",
                                                    "P5_NO_ACTOR_AFTER_REFUSAL"))),
        "G9_HOST_RESTORED_CLEAN": (
            bool(cleanup.get("all_gates_passed")) and "cold_after" in cleanup
            and sampler_stop.get("stopped") is True and not sampler_stop.get("killed")
            and not leftovers),
    }
    return {"gates": gates, "passed": all(gates.values()),
            "g1_detail": g1_frame_accounting(child=child, timing=timing, analysis=analysis,
                                             transmitted_budget=int(transmitted_budget),
                                             boundary_budget=int(boundary_budget)),
            "verdict": ("INSTRUMENTATION_QUALIFIED" if all(gates.values())
                        else "INSTRUMENTATION_NOT_QUALIFIED"),
            "phase6_report_only": {"status": cell_report.get("status"), "gates": phase6}}




def g1_frame_accounting(*, child: Mapping[str, Any], timing: Mapping[str, Any],
                        analysis: Mapping[str, Any], transmitted_budget: int,
                        boundary_budget: int = G1_BOUNDARY_BUDGET) -> dict[str, Any]:
    """Addendum-2 G1: 300 at TRANSMITTED_BUDGET_REACHED, or exactly 299 at a verified
    DECISION_CYCLE_BOUNDARY (budget 300, no decision cap); both branches require
    matching transmission counts, one terminal per sent frame and zero unresolved
    tickets."""
    integration = dict(timing.get("integration") or {})
    sent = (timing.get("counts") or {}).get("sent_frames")
    collector = dict(child.get("collector") or {})
    cycle = dict(analysis.get("phase6_cycle") or {})
    stop = child.get("stop_reason")
    budget = int(transmitted_budget)
    checks = {
        "counts_match": (sent is not None
                         and sent == (analysis.get("counts") or {}).get("transmitted")
                         and sent == cycle.get("transmitted_identities")
                         and sent == collector.get("transmitted_frames")),
        "collector_budget": collector.get("transmitted_budget") == budget,
        "one_terminal_per_sent_frame": (bool(integration)
                                        and not integration.get("frames_without_terminal")
                                        and not integration.get("frames_with_multiple_terminals")
                                        and not integration.get("terminals_for_untransmitted_frames")),
        "zero_unresolved_tickets": (integration.get("unresolved_tickets_at_close") == 0
                                    and cycle.get("unresolved_tickets_at_close") == 0
                                    and not integration.get("tickets_without_resolution")),
    }
    budget_branch = stop == "TRANSMITTED_BUDGET_REACHED" and sent == budget
    boundary_branch = (stop == "DECISION_CYCLE_BOUNDARY"
                       and budget == int(boundary_budget) and sent == int(boundary_budget) - 1
                       and cycle.get("cycle_boundary_reached") is True
                       and cycle.get("cycle_budget") == {"frame_budget": int(boundary_budget),
                                                         "decision_cap": None})
    branch = ("TRANSMITTED_BUDGET_REACHED" if budget_branch
              else f"VERIFIED_DECISION_CYCLE_BOUNDARY_{int(boundary_budget) - 1}"
              if boundary_branch else None)
    return {"branch": branch, "stop_reason": stop, "sent": sent, "checks": checks,
            "passed": branch is not None and all(checks.values())}


def experiment_checks(*, cell: Path, arm: str, inventory: Optional[Mapping[str, Any]]
                      ) -> dict[str, Any]:
    """Thread-cap experiment evidence (reported beside G1-G9, not a capture gate).

    E1: both UE settings snapshots exist;  E2: the UE child saw the arm's
    variables and, for B, every observed OpenMP/MKL/PyTorch pool reports one
    thread (for A the variables are absent);  E3: no process outside the UE child
    carried a cap variable away from its baseline value.
    """
    leaf = Path(cell) / TCAP.EVIDENCE_LEAF
    snapshots = {}
    for stage in ("PRE", "POST"):
        path = leaf / TCAP.UE_SETTINGS_NAME.format(stage=stage)
        snapshots[stage.lower()] = (json.loads(path.read_text(encoding="utf-8"))
                                    if path.is_file() else None)
    post = snapshots["post"] or {}
    pre = snapshots["pre"] or {}
    env_post = dict(post.get("environment") or {})
    expected = TCAP.ARMS[arm]
    capped = bool(expected)                     # arms B, C, T carry the UE cap; A does not
    ue_env_ok = (bool(pre) and bool(post)
                 and all((pre.get("environment") or {}).get(k) == v for k, v in expected.items())
                 and all(env_post.get(k) == v for k, v in expected.items())
                 and (capped or all(env_post.get(k) is None for k in TCAP.CAP_VARIABLES)))
    cap_state = post.get("cap_effective")
    e2 = ue_env_ok and (cap_state is True if capped else cap_state is False)
    bundle_path = Path(cell) / "radar_bundle" / "RADAR_BUNDLE.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8")) if bundle_path.is_file() else None
    expected_bundle = TCAP.BUNDLES[arm]
    if arm in TCAP.BUNDLE_SNAPSHOT_ARMS:
        e4 = (bundle is not None and bundle.get("bundle") == expected_bundle
              and (expected_bundle != "optimized"
                   or (list(bundle.get("components") or []) == list(RB_COMPONENTS)
                       and bundle.get("tracker") == "SingleSortStationaryTrackAccumulator")))
    else:                                  # thread-cap arms predate the bundle: production only
        e4 = bundle is None or bundle.get("bundle") == "production"
    checks = {"E1_UE_SETTINGS_RECORDED": bool(pre) and bool(post),
              "E2_UE_CAP_STATE_AS_REGISTERED": bool(e2),
              "E3_SCOPE_OTHER_PROCESSES_BASELINE": bool(inventory and inventory.get("scope_ok")),
              "E4_RADAR_BUNDLE_AS_REGISTERED": bool(e4)}
    ordering = None
    if arm in TCAP.ORDERING_ARMS:
        ordering = ordering_evidence(cell, expected=TCAP.ORDERINGS[arm])
        checks["E5_EDGE_ORDERING_AS_REGISTERED"] = bool(ordering["passed"])
    torch = post.get("torch") if isinstance(post.get("torch"), Mapping) else {}
    return {"arm": arm, "checks": checks, "all_passed": all(checks.values()),
            "radar_bundle": bundle, "edge_ordering": ordering,
            "effective": {"torch_intra_op_threads": torch.get("intra_op_threads"),
                          "torch_inter_op_threads": torch.get("inter_op_threads"),
                          "openmp_max_threads": post.get("openmp_max_threads"),
                          "threadpools": post.get("threadpools"),
                          "cv2": post.get("cv2"), "cap_effective": cap_state,
                          "ue_environment": env_post,
                          "launch_restore": post.get("launch_restore"),
                          "numerical_libraries_loaded": post.get("numerical_libraries_loaded"),
                          "affinity_cpus": post.get("affinity_cpus")},
            "inventory": inventory}


def ordering_evidence(cell: Path, *, expected: str) -> dict[str, Any]:
    """E5: the edge recorded the registered ordering and submitted each reward
    ticket exactly once (early arm: all from the compute wrapper, none from the
    loop; serial arm: all from the loop, none early)."""
    leaf = Path(cell) / "profiler" / "edge"
    counters_path = leaf / "EDGE_ORDERING.json"
    report_path = Path(cell) / "run4_phase6" / "edge_report.json"
    document = json.loads(counters_path.read_text()) if counters_path.is_file() else None
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    evaluator_submitted = int((report.get("evaluator") or {}).get("submitted") or 0)
    loop_submitted = int((report.get("counters") or {}).get("evaluations_submitted") or 0)
    counters = dict((document or {}).get("counters") or {})
    early = int(counters.get("early_submissions", 0))
    if expected == "early":
        exact_once = (early == evaluator_submitted and loop_submitted == 0
                      and counters.get("early_submit_errors", 0) == 0
                      and counters.get("loop_submission_suppressed", 0) == early
                      and counters.get("live_gate_rejected", 0) == 0)
    else:
        exact_once = early == 0 and loop_submitted == evaluator_submitted
    return {"expected": expected, "recorded": None if document is None else document.get("ordering"),
            "counters": counters, "evaluator_submitted": evaluator_submitted,
            "loop_submitted": loop_submitted,
            "passed": bool(document is not None and document.get("ordering") == expected and exact_once)}


def _profiler_clean(row: Mapping[str, Any]) -> bool:
    counters = dict(row.get("counters") or {})
    return (row.get("footer_present", True) is not False
            and not counters.get("write_failed") and not counters.get("dropped_records")
            and not counters.get("internal_errors") and not counters.get("flush_loop_errors")
            and not counters.get("unbalanced_span"))


def policy_runtime_summary(ue_records: Sequence[Mapping[str, Any]],
                           expected_boundary: Optional[str]) -> dict[str, Any]:
    header = next((r for r in ue_records if r.get("r") == "header"), {}) or {}
    start = ((header.get("identity") or {}).get("policy") or {})
    runtime = next((r for r in ue_records if r.get("r") == "note"
                    and r.get("kind") == "policy_runtime_identity"), {}) or {}
    close = next((r for r in ue_records if r.get("r") == "note"
                  and r.get("kind") == "policy_identity_at_close"), {}) or {}
    vector = next((r for r in ue_records if r.get("r") == "note"
                   and r.get("kind") == "policy_state_vector"), {}) or {}
    boundary = (runtime.get("runtime") or {}).get("boundary_sha256")
    return {"binding_at_start": start, "runtime": runtime.get("runtime"),
            "state_vector": {k: vector.get(k) for k in ("vector_type", "state_dim",
                                                        "feature_schema_id")},
            "weights_before": start.get("weights_file_sha256_observed"),
            "weights_after": close.get("weights_file_sha256_observed"),
            "runtime_boundary": boundary,
            "runtime_boundary_matches_binding": (boundary is not None
                                                 and boundary == start.get("actor_boundary_sha256")
                                                 and (expected_boundary is None
                                                      or boundary == expected_boundary)),
            "declared_state_dim": start.get("state_dim"),
            "observed_state_dim": vector.get("state_dim")}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    from rl_agent import ue_288_campaign_supervisor as SUP
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
        phase6_timing_characterization_v2 as TC,
    )

    from . import analyze_v1 as AN
    from . import recorder_v1 as RV
    from . import sampler_v1 as SA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", default="")
    parser.add_argument("--config", type=Path, default=R.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, default=QUALIFICATION_PATH)
    parser.add_argument("--arm", default=None, help="experiment arm (registrations with arms)")
    parser.add_argument("--transmitted-budget", type=int, default=None)
    parser.add_argument("--safety-timeout-s", type=float, default=120.0)
    parser.add_argument("--child-timeout-s", type=float, default=1200.0)
    parser.add_argument("--carla-port", type=int, default=2000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_root = args.output_root.resolve()
    registration = load_qualification(args.registration.resolve(strict=True))
    if args.transmitted_budget is None:
        args.transmitted_budget = int(registration["transmitted_budget"])
    lifecycle = dict(registration.get("lifecycle") or {})
    if lifecycle:                         # registered bounds; no ad-hoc override
        args.safety_timeout_s = float(lifecycle["safety_timeout_s"])
        args.child_timeout_s = float(lifecycle["child_timeout_s"])
    arm = select_arm(registration, args.arm)
    execute_token = str(arm["execute_token"])
    boundary_budget = int(registration.get("g1_boundary_budget", G1_BOUNDARY_BUDGET))
    config, cells, report, qualification = preflight(
        args.config.resolve(strict=True), output_root=output_root,
        transmitted_budget=int(args.transmitted_budget),
        registration_path=args.registration.resolve(strict=True))
    report["lifecycle"] = {"transmitted_budget": int(args.transmitted_budget),
                           "safety_timeout_s": float(args.safety_timeout_s),
                           "child_timeout_s": float(args.child_timeout_s),
                           "g1_boundary_budget": boundary_budget}
    report["arm"] = {"arm": arm["arm"], "ue_child_env_additions": arm["env"],
                     "parent_env_cap_variables": {k: os.environ.get(k)
                                                  for k in TCAP.CAP_VARIABLES}}
    if args.preflight:
        print(json.dumps(report, sort_keys=True, indent=1, default=str))
        print("SINGLE_GPU_PROFILER_V1_PREFLIGHT_PASS")
        return 0
    require(args.execute == execute_token, f"live launch requires --execute {execute_token}")
    require(len(cells) == 1, "exactly one registered carrier cell is required")
    output_root.mkdir(parents=False, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_create_only(output_root / "PREFLIGHT.json", report)
    profile_config = profiler_config(qualification, run_label=run_id)
    write_create_only(output_root / "PROFILER_CONFIG.json", profile_config)
    audits = {"before": run_audit(qualification["policy"]["audit_module"],
                                  output_root / "ACTOR_AUDIT_BEFORE.json")}
    R.CHILD_MODULE = report["child_module"]
    inventory = None
    if arm["arm"] is not None:
        require(arm["env"] == TCAP.ARMS[arm["arm"]], "registered arm env differs from threadcap_v1")
        TCAP.install_lifecycle_arm(SUP, arm["arm"])
        inventory = TCAP.ProcessEnvironmentInventory(
            output_root / TCAP.EVIDENCE_LEAF / TCAP.INVENTORY_NAME, root_pid=os.getpid(),
            baseline=TCAP.baseline_values(os.environ), classify=SA.classify).start()
    if not hasattr(SUP._live_radio_state, "original"):
        SUP._live_radio_state = anchored_live_radio_state(SUP, launcher_root(config))
    sampler = None
    cell_report: dict[str, Any] = {}
    try:
        sampler = start_sampler(output_root, qualification["profiler"]["sampler_interval_ms"])
        cell_report = R.run_one_cell(
            base_config=config, registered=cells[0], output_root=output_root, run_id=run_id,
            transmitted_budget=int(args.transmitted_budget),
            safety_timeout_s=float(args.safety_timeout_s), carla_port=int(args.carla_port),
            child_timeout_s=float(args.child_timeout_s))
        time.sleep(3.0)                       # post-teardown samples
    finally:
        sampler_stop = stop_sampler(sampler)
        inventory_summary = None if inventory is None else inventory.stop()
    audits["after"] = run_audit(qualification["policy"]["audit_module"],
                                output_root / "ACTOR_AUDIT_AFTER.json")
    leftovers = leftover_processes()
    cell = output_root / "cells" / cells[0].cell_id
    timing: dict[str, Any] = {}
    analysis: dict[str, Any] = {}
    errors = []
    try:
        timing = TC.analyze(cell)
        write_create_only(output_root / "PHASE6_TIMING_CHARACTERIZATION.json", timing)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"timing: {type(exc).__name__}: {exc}")
    try:
        analysis = AN.analyze(output_root, config=profile_config)
        ue_records = list(RV.iter_records(cell / "profiler" / "ue.profile.jsonl"))
        analysis["policy_runtime"] = policy_runtime_summary(
            ue_records, expected_boundary=(audits["before"].get("document") or {})
            .get("actor_boundary_sha256"))
        write_create_only(output_root / "PROFILER_ANALYSIS.json", analysis)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"analysis: {type(exc).__name__}: {exc}")
    result = evaluate_gates(cell_report=cell_report, analysis=analysis, timing=timing,
                            audits=audits,
                            expected_weights_sha256=qualification["policy"]
                            ["expected_weights_sha256"],
                            sampler_stop=sampler_stop, leftovers=leftovers,
                            transmitted_budget=int(args.transmitted_budget),
                            boundary_budget=boundary_budget)
    if arm["arm"] is not None:
        result["threadcap"] = experiment_checks(cell=cell, arm=arm["arm"],
                                                inventory=inventory_summary)
    if qualification.get("kind") == "PROFILING_CAPTURE":
        result["verdict"] = "CAPTURE_VALID" if result["passed"] else "CAPTURE_INVALID"
    result.update({"schema": "scenesense.single_gpu_profiler.qualification_result.v1",
                   "qualification_id": qualification["id"], "run_id": run_id,
                   "addendum_2": report.get("addendum_2"),
                   "claim_scope": qualification.get("claim_scope")
                   or qualification.get("objective"),
                   "registration": report.get("registration"),
                   "arm": report.get("arm"),
                   "capture_binding": report.get("capture_binding"),
                   "experiment_binding": report.get("experiment_binding"),
                   "lifecycle": report.get("lifecycle"),
                   "not_a_profiling_baseline": qualification.get("not_a_profiling_baseline"),
                   "sampler_stop": sampler_stop, "leftover_processes": leftovers,
                   "host_contention_after": host_contention_state(),
                   "post_run_errors": errors,
                   "audits": {k: {"returncode": v.get("returncode"),
                                  "verdict": (v.get("document") or {}).get("verdict")}
                              for k, v in audits.items()}})
    if qualification.get("kind") == "PROFILING_CAPTURE":
        from . import export_v1 as EX

        try:
            manifest = EX.export(output_root, output_root / "export_v1", result=result)
            result["export"] = {"path": str(output_root / "export_v1"),
                                "row_counts": manifest["row_counts"],
                                "verified": EX.verify(output_root / "export_v1")["verified"]}
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            result["export"] = {"error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"export: {type(exc).__name__}: {exc}")
    write_create_only(output_root / "QUALIFICATION_RESULT.json", result)
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"],
                      "phase6": result["phase6_report_only"]["status"],
                      "threadcap": (result.get("threadcap") or {}).get("checks"),
                      "errors": errors}, sort_keys=True, indent=1, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
