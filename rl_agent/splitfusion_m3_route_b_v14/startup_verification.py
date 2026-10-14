"""Fresh-process startup verification of one prepared v14 prospective, every process role, before any launch.

Copy of the v12 verification with the v14 identities; the GT role checks the unchanged v12 worker binding (the
corrected exact storage-accounting V3.v2 session and its bounded per-call timer installed and restored in prep only,
the fast path active, the bounded caches recorded and closed, at the qualified sources; the tilt-aware AVO switch and
the v7 FREEZE lifetime repair still installed and restored in prep; the score worker untouched; no failure snapshot
on a clean start/close).

Each role runs in its own interpreter through its production entry point and executes the SAME validation
functions the live start executes, then stops before hardware start (no CARLA, OAI, GPU inference or capture):

  host                 python -m <v12 host> verify            (verify_prospective + verify_prerequisite)
  runner_parent        python -m <v12 runner> ... --verify-only (validate_parent + retained child-module import)
  runner_cli_preflight python -m <v12 runner> ... --preflight  (the exact first command of host launch)
  ue_child             python -m <v12 child trampoline> --m3-verify-startup --output <root>
  edge                 the admitted oai-perception-rx image, live entrypoint/PYTHONPATH, worktree at /work/abiodun,
                       the prospective's own compose_partition.yaml mounts (read-only), no GPU, no network:
                       python3 -m <v12 edge trampoline> --m3-verify-startup
  gt_service_workers   the v14 UE seam starts the real GT service (v12 trampoline) which starts both real workers
                       (v12 -> v8 -> v7 -> v4 trampoline): service source verification, worker INIT role/producer
                       checks, accounting + timer + AVO + prep lifetime binding, READY, CLOSE

Negative controls (fresh processes, on copies inside the evidence directory): wrong schema, wrong launcher
version, altered runtime-source binding, altered registered-source hash, altered tilt-aware AVO source hash, altered
storage-accounting source hash, altered prerequisite binding, the V12 GOOD_ALL channel section inside an otherwise
valid v14 registration (host and child), the actual V12 registration, and a tampered GT-service source digest.

  cd <deployment worktree>; env -u PYTHONPATH -u CUDA_VISIBLE_DEVICES -u OMP_NUM_THREADS -u MKL_NUM_THREADS \\
    /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v14.startup_verification run --name <dir> --prospective <root>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import contract as C
from .host import STARTUP_ROLES, STARTUP_SCHEMA, compose_text

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v14"
EDGE_IMAGE = "sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba"   # admitted live edge image
EDGE_PYTHONPATH = "/work/abiodun:/work/abiodun/rl_agent/feature_ae"                      # receiver compose env
MARK = "M3_V14_STARTUP_VERIFICATION="
V12_ROOT = ROOT / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v12/"
                  "bad_all_prospective_02")
V12_FIXTURE_FILES = ("PROSPECTIVE.json", "registration.json", "PREREQUISITE.json", "ACCEPTANCE.json", "arm.json",
                    "compose_partition.yaml", C.RETENTION_PLAN_FILE)     # not the capture logs of the V12 run
PROCESS_NEEDLES = ("CarlaUE4", "nr-softmodem", "nr-uesoftmodem", "splitfusion_exact_gt_live_v2.service",
                   "service_entry", "worker_entry", "edge_profiled_v1_m3_route_b", "child_v1_m3_route_b")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def host_env(prospective, retained):
    """The environment the retained host lifecycle gives its runner (and, through it, the child)."""
    env = dict(os.environ, SCENESENSE_PARTITION_OUTPUT=str(prospective), SCENESENSE_PARTITION_CODE=str(retained),
               PYTHONDONTWRITEBYTECODE="1")
    for name in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LD_PRELOAD"):
        env.pop(name, None)
    return env


def cpu_env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return env


def processes():
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and str(os.getpid()) != path.parent.name and any(n in command for n in PROCESS_NEEDLES) \
                and "startup_verification" not in command:
            found.append(dict(pid=int(path.parent.name), command=command[:400]))
    containers = subprocess.run(["sudo", "-n", "docker", "ps", "-q", "--filter", f"ancestor={EDGE_IMAGE}"],
                                capture_output=True, text=True).stdout.split()
    return dict(processes=found, edge_image_containers=containers)


def fresh(argv, log, env, timeout=900):
    started = time.monotonic()
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    Path(log).write_text("$ " + " ".join(map(str, argv)) + "\n" + proc.stdout + proc.stderr)
    marks = [json.loads(line[len(MARK):]) for line in proc.stdout.splitlines() if line.startswith(MARK)]
    tail = (proc.stdout + proc.stderr).rstrip().splitlines()
    return dict(returncode=proc.returncode, document=marks[-1] if len(marks) == 1 else None,
                last_line=tail[-1] if tail else None, wall_s=round(time.monotonic() - started, 3),
                log=str(Path(log).relative_to(EVIDENCE)))


def py(*args):
    return ["/usr/bin/python3", "-B", *args]


def compose_mounts(path):
    mounts = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("- "):
            source, target, mode = line[2:].rsplit(":", 2)
            mounts.append((source, target, mode))
    return mounts


def edge_argv(partition_root):
    argv = ["sudo", "-n", "docker", "run", "--rm", "--network", "none",
            "-e", "PYTHONPATH=" + EDGE_PYTHONPATH, "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{ROOT}:/work/abiodun:ro"]
    for source, target, _mode in compose_mounts(Path(partition_root) / "compose_partition.yaml"):
        argv += ["-v", f"{source}:{target}:ro"]          # live: /partition_output rw; verification never writes
    return argv + [EDGE_IMAGE, "/bin/bash", "-lc", f"exec python3 -u -B -m {C.EDGE} {C.VERIFY_FLAG}"]


# ---------------------------------------------------------------------------- negative fixtures
def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _reseal(fixture, registration=None, record_update=None):
    record = json.loads((fixture / "PROSPECTIVE.json").read_bytes())
    if registration is not None:
        registration.pop("registration_sha256", None)
        registration["registration_sha256"] = C.canonical(registration)
        _write_json(fixture / "registration.json", registration)
        record["registration_file_sha256"] = sha(fixture / "registration.json")
    record.update(record_update or {})
    _write_json(fixture / "PROSPECTIVE.json", record)


def make_fixture(prospective, destination, variant):
    """A copy of the prospective that is legitimate except for ONE registered alteration."""
    destination.mkdir(parents=True)
    for path in Path(prospective).iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    record = json.loads((destination / "PROSPECTIVE.json").read_bytes())
    (destination / "compose_partition.yaml").write_text(compose_text(destination.resolve(), record["retained_code"]))
    _reseal(destination, record_update=dict(compose_file_sha256=sha(destination / "compose_partition.yaml")))
    record = json.loads((destination / "PROSPECTIVE.json").read_bytes())
    registration = json.loads((destination / "registration.json").read_bytes())
    if variant == "legitimate_copy":
        pass
    elif variant == "wrong_schema":
        _reseal(destination, record_update=dict(schema="splitfusion.m3_route_b_v12.prospective.v1"))
    elif variant == "wrong_launcher_version":
        registration["launcher_revision"] = C.V12.LAUNCHER_REVISION
        _reseal(destination, registration)
    elif variant == "wrong_registration_id":
        registration["id"] = C.V4.ID
        _reseal(destination, registration)
    elif variant == "altered_runtime_source":
        key = sorted(record["runtime_sources"])[0]
        record["runtime_sources"][key] = "0" * 64
        _reseal(destination, record_update=dict(runtime_sources=record["runtime_sources"]))
    elif variant == "altered_registered_source":
        key = next(k for k in sorted(registration["exact_gt_sources"]) if k.startswith(C.PACKAGE))
        registration["exact_gt_sources"][key] = "0" * 64
        _reseal(destination, registration)
    elif variant == "altered_avo_source":
        registration["exact_gt_sources"][f"{C.AVO_TILT_PACKAGE}/geometry.py"] = "0" * 64
        _reseal(destination, registration)
    elif variant == "altered_accounting_source":
        registration["exact_gt_sources"][f"{C.ACCOUNTING_PACKAGE}/accounting.py"] = "0" * 64
        _reseal(destination, registration)
    elif variant == "wrong_schedule_bad_all":
        registration["channel"] = json.loads((V12_ROOT / "registration.json").read_bytes())["channel"]
        _reseal(destination, registration)
    elif variant == "altered_prerequisite_binding":
        proof = json.loads((destination / "PREREQUISITE.json").read_bytes())
        proof["binding_regression"]["sha256"] = "0" * 64
        _write_json(destination / "PREREQUISITE.json", proof)
        _reseal(destination, record_update=dict(prerequisite_file_sha256=sha(destination / "PREREQUISITE.json")))
    else:
        raise ValueError(variant)
    return destination


def make_v12_fixture(destination):
    """The actual V12 prospective's top-level files (its registration, record and prerequisite), byte-identical."""
    destination.mkdir(parents=True)
    for name in V12_FIXTURE_FILES:
        shutil.copy2(V12_ROOT / name, destination / name)
    return destination


EXPECTED = {
    "wrong_schema": "foreign prospective registration",
    "wrong_launcher_version": "foreign or incoherent v14 launcher binding",
    "altered_runtime_source": "complete runtime source inventory drift",
    "altered_registered_source": "foreign/tampered full Run-5 live registration",
    "altered_avo_source": "foreign/tampered full Run-5 live registration",
    "altered_accounting_source": "foreign/tampered full Run-5 live registration",
    "altered_prerequisite_binding": "sealed evidence descriptor drift",
    "wrong_schedule_bad_all": "registration schedule differs from the registered BAD_TO_GOOD schedule / carrier cell",
    "v12_registration": "foreign prospective registration",
}
# the child derives its profiler configuration from the mounted registration, whose id is the V12 id
CHILD_EXPECTED = dict(EXPECTED, v12_registration="sealed enabled full Run-5 live profiler configuration required")
EDGE_EXPECTED = {
    "wrong_launcher_version": "registration launcher binding differs from the v14 contract",
    "wrong_registration_id": "full Run-5 live entry requires its exact sealed registration",
    "altered_registered_source": "registered source drift",
    "altered_avo_source": "registered source drift",
    "altered_accounting_source": "registered source drift",
    "v12_registration": "full Run-5 live entry requires its exact sealed registration",
}
CHILD_VARIANTS = ("wrong_schema", "wrong_launcher_version", "altered_runtime_source", "altered_registered_source",
                  "altered_avo_source", "altered_accounting_source", "wrong_schedule_bad_all", "v12_registration")
HOST_VARIANTS = (*CHILD_VARIANTS, "altered_prerequisite_binding")


def _rejected(result, expected):
    return result["returncode"] != 0 and result["document"] is None and (expected in (result["last_line"] or ""))


# ---------------------------------------------------------------------------- GT service + workers role
def service_role(prospective, out, variant):
    """UE side of the GT service start, in this fresh process: v14 child registration check, v14 seam, real service."""
    from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
    from rl_agent.splitfusion_exact_gt_live_v2.ue import EarlyGtUeHandoff
    from rl_agent.splitfusion_numeric_gt_runtime_v1.async_live_rehearsal import _owned_collector
    from .entries import registration
    from .gc_seams import install_ue_service_seam
    P.environment_ok()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    reg = registration(Path(prospective) / "registration.json", ROOT)
    corpus = P.Corpus(49)
    limits = reg["exact_gt_limits"]
    sources = dict(reg["exact_gt_sources"])
    if variant == "tampered_service_source":
        sources[sorted(sources)[0]] = "0" * 64
    handoff_registration = dict(          # live limits/sources/names; offline corpus bindings (rehearsal pattern)
        exact_gt_limits=limits, exact_gt_sources=sources, numeric_run_binding=corpus.binding,
        numeric_producer_binding=corpus.startup["binding_sha256"], numeric_prediction_contract=corpus.contract,
        exact_gt_socket_name=reg["exact_gt_socket_name"], exact_gt_capability_name=reg["exact_gt_capability_name"])
    engine = dict(producer_args=dict(corpus.producer_args(), max_frames=int(limits["max_frames"])),
                  run_binding=corpus.binding, producer_binding=corpus.startup["binding_sha256"],
                  prediction_contract=corpus.contract, max_frames=int(limits["max_frames"]),
                  max_retained_jobs=int(limits["prep_jobs"]),
                  max_retained_bytes=int(limits["prep_jobs"]) * 64 * 1024 * 1024, offline_replay=False)
    ue_dir, evidence = out / "ue_attempt", out / "edge_evidence"
    ue_dir.mkdir()
    evidence.mkdir()
    state = dict(ready=None, closed=None, error=None)
    with install_ue_service_seam(registration=reg, output=out) as trace_dir:
        handoff = EarlyGtUeHandoff(attempt_dir=ue_dir, registration=handoff_registration, service_config={})
        handoff.collector = _owned_collector(corpus.cases)
        handoff.evidence_dir = evidence
        try:
            handoff.start_service(engine)
            state["ready"] = {k: v for k, v in handoff.ready.items() if k != "session"}
            state["closed"] = handoff.close()
        except BaseException as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"[:1000]
            if handoff.process is not None and handoff.process.poll() is None:
                handoff.abort(exc)
    service_log = (ue_dir / "numeric_gt/service.log")
    log_text = service_log.read_text(errors="replace") if service_log.exists() else ""
    docs = {}
    for name in ("service_substitution.json", "prep_gc_trace.json", "score_gc_trace.json", "ue_substitution.json",
                 C.PREP_BINDING_RECORD, C.AVO_BINDING_RECORD, C.ACCOUNTING_BINDING_RECORD,
                 C.ACCOUNTING_TIMING_RECORD):
        path = trace_dir / name
        docs[name] = json.loads(path.read_text()) if path.exists() else None
    arm, producer = reg["gt_gc"]["arm"], reg["gt_gc"]["producer"]
    if variant == "tampered_service_source":
        checks = dict(service_refused=state["ready"] is None and "exact-GT service source drift" in log_text,
                      no_worker_started=docs["prep_gc_trace.json"] is None and docs["score_gc_trace.json"] is None
                      and docs[C.ACCOUNTING_BINDING_RECORD] is None)
    else:
        service_sub, ue_sub = docs["service_substitution.json"] or {}, docs["ue_substitution.json"] or {}
        prep, score = docs["prep_gc_trace.json"] or {}, docs["score_gc_trace.json"] or {}
        lifetime = docs[C.PREP_BINDING_RECORD] or {}
        avo = docs[C.AVO_BINDING_RECORD] or {}
        accounting = docs[C.ACCOUNTING_BINDING_RECORD] or {}
        session = accounting.get("accounting_session") or {}
        timing = docs[C.ACCOUNTING_TIMING_RECORD] or {}
        worker_argv = [s.get("argv", []) for s in service_sub.get("substitutions", [])]
        checks = dict(
            service_ready=state["ready"] is not None and state["error"] is None,
            service_closed_clean=state["closed"] is not None and state["closed"]["service_returncode"] == 0
                and not state["closed"]["errors"] and service_sub.get("returncode") == 0,
            ue_seam_started_registered_service_trampoline=len(ue_sub.get("substitutions", [])) == 1
                and C.SERVICE_ENTRY in ue_sub["substitutions"][0]["argv"] and ue_sub.get("restored") is True
                and ue_sub.get("seam_module") == C.GC_SEAM and ue_sub.get("contract_module") == C.MODULE + ".contract"
                and service_sub.get("entry_module") == C.SERVICE_ENTRY
                and service_sub.get("contract_module") == C.MODULE + ".contract",
            service_started_registered_worker_trampoline=len(worker_argv) == 2
                and all(C.WORKER_ENTRY in argv for argv in worker_argv)
                and sorted((s["role"], s["gc_policy"], s["producer"]) for s in service_sub["substitutions"])
                == sorted([("prep", arm, producer), ("score", "control", "v1")]),
            worker_roles_verified=prep.get("role_seen") == "prep" and score.get("role_seen") == "score"
                and prep.get("refusal") is None and score.get("refusal") is None,
            prep_producer_constructed=prep.get("producer_constructed") == C.PRODUCER_CLASS[producer],
            gc_policy_only_prep=prep.get("policy_installed") is (arm == "candidate") and score.get("policy_installed") is False,
            gc_state_restored=prep.get("restored_equals_initial") is True and score.get("restored_equals_initial") is True,
            capacity_and_pose_substitutions=[r["id"] for r in (prep.get("m3_capacity") or {}).get("installed", [])]
                == ["C1_PREP_DEFERRED_ENGINE", "C2_PREP_PRODUCER_SOURCE", "P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"]
                and [r["id"] for r in (score.get("m3_capacity") or {}).get("installed", [])]
                == ["P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"]
                and [r["id"] for r in (service_sub.get("m3_capacity") or {}).get("installed", [])]
                == ["C3_SERVICE_ARCHIVE_SIZING"],
            registered_limits=prep.get("limits") == C.trace_limits(limits["max_frames"]),
            prep_lifetime_repair_bound=lifetime.get("installed") is True and lifetime.get("restored") is True
                and lifetime.get("entry_module") == C.V7_WORKER_ENTRY
                and lifetime.get("retained_trampoline") == C.RETAINED_WORKER_ENTRY
                and lifetime.get("repair_module") == C.PREP_LIFETIME_MODULE
                and lifetime.get("repair_sha256") == C.PREP_LIFETIME_QUALIFIED[f"{C.PREP_LIFETIME_PACKAGE}/repair.py"]
                and lifetime.get("failure_directory") == str(trace_dir) and lifetime.get("pid") == prep.get("pid")
                and lifetime.get("exit") is None,
            prep_tilt_aware_avo_bound=avo.get("installed") is True and avo.get("restored") is True
                and avo.get("exit") is None and avo.get("entry_module") == C.V8_WORKER_ENTRY
                and avo.get("v7_worker_entry") == C.V7_WORKER_ENTRY
                and avo.get("geometry_version") == C.AVO_GEOMETRY_VERSION
                and avo.get("sources") == {name: C.AVO_TILT_QUALIFIED[f"{C.AVO_TILT_PACKAGE}/{name}.py"]
                                           for name in ("geometry", "quality", "switch")}
                and avo.get("pid") == prep.get("pid") == lifetime.get("pid"),
            prep_storage_accounting_bound=accounting.get("installed") is True
                and accounting.get("timer_installed") is True and accounting.get("timer_removed") is True
                and accounting.get("restored") is True and accounting.get("exit") is None
                and accounting.get("entry_module") == C.WORKER_ENTRY
                and accounting.get("v8_worker_entry") == C.V8_WORKER_ENTRY
                and accounting.get("version") == C.ACCOUNTING_VERSION
                and accounting.get("substitution") == C.ACCOUNTING_SUBSTITUTION
                and accounting.get("sources") == {name: C.ACCOUNTING_QUALIFIED[f"{C.ACCOUNTING_PACKAGE}/{name}.py"]
                                                  for name in ("accounting", "switch")}
                and accounting.get("pid") == prep.get("pid") == avo.get("pid"),
            # This role starts and closes the service without issuing a decision, so the session makes no calls
            # here: what is verifiable is that the interpreter running the real prep worker HAS the CPython dict
            # mutation tag the registered fast path needs (without it the entry refuses), and that the session is
            # the registered bounded one.  Reuse actually happening is proved by the real-service proof.
            prep_accounting_fast_path_available=accounting.get("fast_path_available") is True
                and all(reason is None for reason in
                        (accounting.get("capabilities") or {"missing": "capability"}).values())
                and accounting.get("max_objects") == C.ACCOUNTING_MAX_OBJECTS
                and (session or {}).get("version") == C.ACCOUNTING_VERSION
                and (session or {}).get("max_objects") == C.ACCOUNTING_MAX_OBJECTS
                and all(reason is None for reason in
                        ((session or {}).get("capability") or {"missing": "capability"}).values())
                and (session or {}).get("snapshot_fallbacks") == 0
                and (session or {}).get("owner_shortcut_rejections") == 0,
            prep_accounting_session_bounded_and_closed=(session or {}).get("memo_objects", 1 << 30)
                <= C.ACCOUNTING_MAX_OBJECTS
                and accounting.get("accounting_session_closed") is True,
            prep_accounting_timer_recorded=timing.get("schema") == "splitfusion.m3_route_b_v14."
                "storage_accounting_timing.v1" and timing.get("capacity") == C.ACCOUNTING_TIMING_CAPACITY
                and timing.get("dropped") == 0 and timing.get("errors") == 0
                and timing.get("foreign_frames") == 0 and timing.get("recorded") == timing.get("calls"),
            score_worker_untouched=score.get("pid") not in (None, lifetime.get("pid")),
            no_failure_snapshot_on_clean_close=not (trace_dir / C.FAILURE_SNAPSHOT_DIR).exists())
    document = dict(role="gt_service_workers", variant=variant, status="PASS" if all(checks.values()) else "FAIL",
                    checks=checks, state=state, arm=arm, producer=producer,
                    resolved=dict(seam=C.GC_SEAM, service_trampoline=C.SERVICE_ENTRY, worker_trampoline=C.WORKER_ENTRY,
                                  service_contract=service_sub.get("contract_module") if variant != "tampered_service_source"
                                  else None,
                                  worker_contract="rl_agent.splitfusion_m3_route_b_v4.contract "
                                                  "(v12 -> v8 -> v7 -> v4 trampoline)",
                                  prep_accounting=C.launcher_binding()["gt_workers"]["prep_accounting"],
                                  prep_avo=C.launcher_binding()["gt_workers"]["prep_avo"],
                                  prep_core=C.launcher_binding()["gt_workers"]["prep_core"],
                                  service_registration_check="splitfusion_exact_gt_live_v2.service.verify_sources"),
                    service_log_tail=log_text[-1500:])
    print(MARK + json.dumps(document, sort_keys=True, default=str), flush=True)
    return 0 if document["status"] == "PASS" else 1


# ---------------------------------------------------------------------------- orchestration
def _role_pass(name, result, expected_module=None):
    doc = result["document"] or {}
    ok = result["returncode"] == 0 and doc.get("status") == "PASS" and doc.get("cuda_initialized", False) is False
    if expected_module is not None:
        ok = ok and doc.get("module") == expected_module
    return dict(result, status="PASS" if ok else "FAIL")


def run(name, prospective):
    from .host import verify_prospective
    prospective = Path(prospective).resolve(strict=True)
    record = verify_prospective(prospective)
    retained = record["retained_code"]
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    logs = out / "logs"
    logs.mkdir()
    before = processes()
    binding = C.launcher_binding()
    v13c = C.MODULE + ".contract"
    henv, registration = host_env(prospective, retained), prospective / "registration.json"
    roles = {}

    result = fresh(py("-m", C.HOST, "verify", "--output", str(prospective)), logs / "host.log", henv)
    roles["host"] = _role_pass("host", result)
    roles["host"]["status"] = "PASS" if roles["host"]["status"] == "PASS" and (result["document"] or {}).get(
        "resolved") == dict(verify_prospective=C.HOST, verify_prerequisite=C.HOST, check_prerequisites=C.HOST,
                            verify_startup_proof=C.HOST, contract=v13c) else "FAIL"

    result = fresh(py("-m", C.RUNNER, "--registration", str(registration), "--arm", "C2", "--output-root",
                      str(out / "unused_capture"), "--verify-only"), logs / "runner_parent.log", henv)
    roles["runner_parent"] = _role_pass("runner_parent", result)
    if (result["document"] or {}).get("resolved") != dict(validate_parent=C.RUNNER, verify_prospective=C.HOST,
                                                          verify_prerequisite=C.HOST, contract=v13c):
        roles["runner_parent"]["status"] = "FAIL"

    preflight_root = out / "unused_preflight_capture"
    result = fresh(py("-m", C.RUNNER, "--registration", str(registration), "--arm", "C2", "--output-root",
                      str(preflight_root), "--preflight"), logs / "runner_cli_preflight.log", henv)
    text = (logs / "runner_cli_preflight.log").read_text()
    lines = [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("M3_VERSION_BINDING=")]
    printed = json.loads(lines[-1]) if len(lines) == 1 else None
    roles["runner_cli_preflight"] = dict(result, printed_binding=printed, capture_created=preflight_root.exists(),
        status="PASS" if result["returncode"] == 0 and printed is not None
        and {k: v for k, v in printed.items() if k != "resolved"} == binding
        and printed.get("resolved") == dict(validate_parent=C.RUNNER, verify_prospective=C.HOST,
                                            verify_prerequisite=C.HOST, contract=v13c)
        and not preflight_root.exists() else "FAIL")

    result = fresh(py("-m", C.CHILD, C.VERIFY_FLAG, "--output", str(prospective)), logs / "ue_child.log", henv)
    roles["ue_child"] = _role_pass("ue_child", result, C.CHILD)
    doc = result["document"] or {}
    if (doc.get("resolved") != dict(verify_prospective=C.HOST, registration=C.ENTRIES, load_observer=C.RETENTION,
                                    seam_parameters=C.GC_SEAM, contract=v13c)
            or doc.get("retention_plan_id") != C.RETENTION_PLAN_ID
            or (doc.get("seam") or {}).get("service_entry") != C.SERVICE_ENTRY or doc.get("contract_id") != C.ID):
        roles["ue_child"]["status"] = "FAIL"

    mounts = compose_mounts(prospective / "compose_partition.yaml")
    result = fresh(edge_argv(prospective), logs / "edge.log", dict(os.environ))
    roles["edge"] = _role_pass("edge", result, C.EDGE)
    doc = result["document"] or {}
    if (doc.get("resolved") != dict(validate_edge_startup=C.ENTRIES, registration=C.ENTRIES, contract=v13c)
            or doc.get("registration_id") != C.ID or doc.get("edge_gate_imported") is not True
            or doc.get("root") != "/work/abiodun" or doc.get("uid") != 0
            or [(s, t) for s, t, _ in mounts] != [(str(prospective), "/partition_output"),
                                                 (retained, "/cpu_partition_code")]):
        roles["edge"]["status"] = "FAIL"
    roles["edge"]["image"] = EDGE_IMAGE
    roles["edge"]["mounts"] = [dict(source=s, target=t, live_mode=m, verification_mode="ro") for s, t, m in mounts]

    result = fresh(py("-m", "rl_agent.splitfusion_m3_route_b_v14.startup_verification", "service-role",
                      "--prospective", str(prospective), "--out", str(out / "gt_service_workers"),
                      "--variant", "legitimate"), logs / "gt_service_workers.log", cpu_env())
    roles["gt_service_workers"] = _role_pass("gt_service_workers", result)

    # ---- negative controls (each fresh)
    negatives = dict(host={}, ue_child={}, edge={}, gt_service_workers={})
    fixtures = out / "negative_fixtures"
    built = {variant: make_fixture(prospective, fixtures / variant, variant)
             for variant in ("legitimate_copy", "wrong_schema", "wrong_launcher_version", "wrong_registration_id",
                             "altered_runtime_source", "altered_registered_source", "altered_avo_source",
                             "altered_accounting_source", "wrong_schedule_bad_all", "altered_prerequisite_binding")}
    built["v12_registration"] = make_v12_fixture(fixtures / "v12_registration")
    sanity = fresh(py("-m", C.HOST, "verify", "--output", str(built["legitimate_copy"])),
                   logs / "negative_host_legitimate_copy.log", henv)
    negatives["host"]["legitimate_copy_accepted"] = dict(sanity, rejected=False,
                                                         ok=sanity["returncode"] == 0 and sanity["document"] is not None)
    for variant in HOST_VARIANTS:
        result = fresh(py("-m", C.HOST, "verify", "--output", str(built[variant])),
                       logs / f"negative_host_{variant}.log", henv)
        negatives["host"][variant] = dict(result, expected=EXPECTED[variant], ok=_rejected(result, EXPECTED[variant]))
    for variant in CHILD_VARIANTS:
        result = fresh(py("-m", C.CHILD, C.VERIFY_FLAG, "--output", str(built[variant])),
                       logs / f"negative_child_{variant}.log", henv)
        expected = CHILD_EXPECTED[variant]
        negatives["ue_child"][variant] = dict(result, expected=expected, ok=_rejected(result, expected))
    for variant, expected in EDGE_EXPECTED.items():
        argv = _v12_edge_argv(built[variant], retained) if variant == "v12_registration" else edge_argv(built[variant])
        result = fresh(argv, logs / f"negative_edge_{variant}.log", dict(os.environ))
        negatives["edge"][variant] = dict(result, expected=expected, ok=_rejected(result, expected))
    result = fresh(py("-m", "rl_agent.splitfusion_m3_route_b_v14.startup_verification", "service-role",
                      "--prospective", str(prospective), "--out", str(out / "gt_service_tampered"),
                      "--variant", "tampered_service_source"), logs / "negative_gt_service_tampered.log", cpu_env())
    doc = result["document"] or {}
    negatives["gt_service_workers"]["tampered_service_source"] = dict(result, ok=doc.get("status") == "PASS"
                                                                     and result["returncode"] == 0)
    after = processes()

    all_negative = all(item["ok"] for group in negatives.values() for item in group.values())
    clean = after == before == dict(processes=[], edge_image_containers=[])
    passed = all(roles[name]["status"] == "PASS" for name in STARTUP_ROLES) and all_negative and clean
    result = dict(schema=STARTUP_SCHEMA, status="PASS" if passed else "FAIL", prospective=str(prospective),
                  prospective_file_sha256=sha(prospective / "PROSPECTIVE.json"),
                  registration_file_sha256=sha(registration),
                  registration_content_sha256=json.loads(registration.read_bytes())["registration_sha256"],
                  sources=C.source_inventory(Path(record["worktree"])), launcher_binding=binding, roles=roles,
                  negative_controls=negatives, all_negative_controls_rejected=all_negative,
                  process_snapshot_before=before, process_snapshot_after=after, interpreter="/usr/bin/python3",
                  edge_image=EDGE_IMAGE, live_services_started=False, carla_started=False, oai_started=False,
                  live_inference_started=False, capture_directory_created=(prospective / "capture").exists())
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "OUTPUT_SHA256.json" and "archive" not in p.parts}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"], roles={k: v["status"] for k, v in roles.items()},
                          negatives={g: {k: v["ok"] for k, v in items.items()} for g, items in negatives.items()},
                          clean=clean), indent=1, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return result


def _v12_edge_argv(fixture, retained):
    """The V12 registration mounted where the v14 edge reads its registration."""
    return ["sudo", "-n", "docker", "run", "--rm", "--network", "none", "-e", "PYTHONPATH=" + EDGE_PYTHONPATH,
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{ROOT}:/work/abiodun:ro",
            "-v", f"{fixture}:/partition_output:ro", "-v", f"{retained}:/cpu_partition_code:ro",
            EDGE_IMAGE, "/bin/bash", "-lc", f"exec python3 -u -B -m {C.EDGE} {C.VERIFY_FLAG}"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    one = sub.add_parser("run")
    one.add_argument("--name", required=True)
    one.add_argument("--prospective", type=Path, required=True)
    two = sub.add_parser("service-role")
    two.add_argument("--prospective", type=Path, required=True)
    two.add_argument("--out", type=Path, required=True)
    two.add_argument("--variant", choices=("legitimate", "tampered_service_source"), required=True)
    args = parser.parse_args(argv)
    if args.operation == "run":
        run(args.name, args.prospective)
        return 0
    return service_role(args.prospective, args.out, args.variant)


if __name__ == "__main__":
    raise SystemExit(main())
