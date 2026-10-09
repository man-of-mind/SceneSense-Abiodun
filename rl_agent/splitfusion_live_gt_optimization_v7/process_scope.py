"""Offline, source-bound classification of the two registered CPU children.

This is a prospective verifier, not a replacement for a saved qualification
result. It admits only an exact owned PID pair backed by the registered source
maps, native GT READY/RUN/CLOSE receipts and archive startup/close evidence.
All other processes retain the original OMP/MKL baseline comparison. Unreadable
processes remain explicit coverage gaps; this module never launches anything.
"""
import collections
import hashlib
import json
from pathlib import Path
import shlex

SCHEMA = "splitfusion.registered_cpu_process_scope.v1"
FACTORY = "rl_agent.splitfusion_live_parity_runtime_v8.worker_observer.ObservedClient"
BOOTSTRAP = "rl_agent.splitfusion_live_gt_handoff_v1.bootstrap"
ARCHIVE = "rl_agent.splitfusion_live_gt_handoff_v1.archive"
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
CAP_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pid(value):
    require(type(value) is int and value > 1, "positive non-init native PID required")
    return value


def _context(context, pid, sources):
    require(isinstance(context, dict) and context.get("pid") == pid
        and context.get("bootstrap_sources") == sources and context.get("environment") == ENVIRONMENT
        and type(context.get("torch_threads")) is int and context["torch_threads"] == 1
        and context.get("cuda_initialized") is False, "native GT PID/source/effective caps differ")


def _command(row, module, *, parent=None):
    args = shlex.split(row.get("cmdline", ""))
    expected_length = 6 if parent is None else 8
    require(len(args) == expected_length and args[:4] == ["/usr/bin/python3", "-u", "-m", module]
        and args[4] == "--fd" and args[5].isdigit(), "registered child command differs")
    if parent is not None:
        require(args[6:] == ["--parent-pid", str(parent)], "archive command parent differs")


def verify_registered_children(registration, scope, timeline, inventory_rows, *, baseline,
        expected_worker_sources, expected_bootstrap_sources, expected_binding, expected_factory=FACTORY,
        expected_bootstrap_module=BOOTSTRAP, expected_archive_module=ARCHIVE):
    """Return exactly two evidenced exceptions; a name alone grants none.

    ``expected_*`` are trusted prospective anchors supplied by the caller. The
    saved-evidence wrapper below checks their actual registered source bytes.
    GT /proc environ is an exec-time snapshot: the native bootstrap receipts,
    not a rewritten inventory row, prove its later effective CPU caps.
    """
    require(isinstance(expected_worker_sources, dict) and bool(expected_worker_sources)
        and isinstance(expected_bootstrap_sources, dict) and bool(expected_bootstrap_sources),
        "nonempty registered source maps required")
    for module in (expected_bootstrap_module, expected_archive_module):
        require(module.replace(".", "/") + ".py" in expected_bootstrap_sources,
                "registered child executable source missing")
    require(canonical(dict(schema="splitfusion.gt_scoring_before_archive.v1",
        sources=expected_bootstrap_sources)) == expected_binding, "bootstrap source binding differs")
    require(registration.get("worker_sources") == expected_worker_sources
        and registration.get("bootstrap_sources") == expected_bootstrap_sources
        and registration.get("handoff_binding") == expected_binding
        and registration.get("worker_client_factory") == expected_factory,
        "registered worker/bootstrap/factory binding differs")
    require(scope.get("schema") == "splitfusion.gt_scoring_handoff.runtime_scope.v1"
        and scope.get("sources") == expected_worker_sources
        and scope.get("bootstrap_sources") == expected_bootstrap_sources
        and scope.get("complete") is True and scope.get("errors") == []
        and scope.get("joint_drained") is True and scope.get("worker_close_succeeded") is True
        and scope.get("preserved_before_teardown") is True and scope.get("no_cuda_work") is True
        and scope.get("new_cpu_processes") == 2, "native CPU worker scope incomplete or foreign")
    require(timeline.get("factory") == expected_factory and timeline.get("sources") == expected_worker_sources
        and timeline.get("bootstrap_sources") == expected_bootstrap_sources
        and timeline.get("handoff_binding") == expected_binding
        and timeline.get("complete") is True and timeline.get("worker_drained") is True
        and timeline.get("child_stopped") is True and timeline.get("dropped") == 0
        and timeline.get("errors") == [], "source-bound native process timeline incomplete")
    clients, requests = timeline.get("clients"), timeline.get("requests")
    require(isinstance(clients, list) and len(clients) == 1
        and isinstance(requests, list) and 0 < len(requests) <= registration.get("transmitted_budget", 0),
        "one bounded native GT lifecycle required")
    client = clients[0]
    gt_pid = _pid(scope.get("child_pid"))
    require(client.get("pid") == gt_pid and client.get("exitcode") == 0 and client.get("failed") is False
        and client.get("failure_close") is None, "GT native lifecycle failed or PID differs")
    ready, close = client.get("ready", {}), client.get("close", {})
    require(ready == scope.get("child_ready") and ready.get("status") == "OK"
        and ready.get("pid") == gt_pid and ready.get("cuda_initialized") is False
        and close.get("status") == "OK" and close.get("processed") == len(requests),
        "native READY/CLOSE identity or processed count differs")
    _context(ready.get("process_context"), gt_pid, expected_bootstrap_sources)
    _context(close.get("process_context"), gt_pid, expected_bootstrap_sources)
    frames = []
    for row in requests:
        require(row.get("status") == "OK" and row.get("error") is None
            and row.get("worker", {}).get("pid") == gt_pid
            and row.get("ipc", {}).get("child_pid") == gt_pid
            and row.get("ipc", {}).get("frame_id") == row.get("identity", {}).get("frame_id"),
            "native RUN identity/status differs")
        _context(row.get("process_context"), gt_pid, expected_bootstrap_sources)
        frames.append(row["identity"]["frame_id"])
    require(len(set(frames)) == len(frames)
        and scope.get("parent_ipc_records") == [dict(row["ipc"], stage="parent_snapshot_and_ipc")
                                                for row in requests], "native RUN/parent IPC population differs")
    archive = close.get("timing", {}).get("archive", {})
    startup = client.get("archive_startup", {})
    archive_pid = _pid(archive.get("child_pid"))
    require(gt_pid != archive_pid and scope.get("archive") == archive
        and archive.get("binding") == dict(schema="splitfusion.gt_scoring_before_archive.v1", source_binding=expected_binding)
        and archive.get("child_returncode") == 0 and archive.get("failed") is False
        and archive.get("outstanding_jobs") == archive.get("outstanding_bytes") == 0
        and archive.get("completed") == archive.get("submitted") == len(requests),
        "native archive source/PID/drain differs")
    require(startup.get("pid") == archive_pid and startup.get("parent_pid") == gt_pid
        and startup.get("environment") == ENVIRONMENT
        and startup.get("inspection") == "PROCFS_BEFORE_SENSOR_STARTUP"
        and startup.get("numerical_library_settings_observed") is False,
        "actual archive startup parent/effective caps missing")
    for key, field in (("uid", "publisher_uid"), ("gid", "publisher_gid")):
        require(type(registration.get(field)) is int and startup.get(key) == [registration[field]] * 4,
                "archive startup credentials differ from registration")
    completions = archive.get("completions")
    require(isinstance(completions, list) and len(completions) == len(requests)
        and all(row.get("pid") == archive_pid and row.get("status") == "COMPLETE"
                and row.get("durable") is True for row in completions), "archive native completion PID/status differs")
    by_pid = collections.defaultdict(list)
    for row in inventory_rows:
        if "summary" not in row:
            by_pid[row.get("pid")].append(row)
    require(len(by_pid[gt_pid]) == len(by_pid[archive_pid]) == 1, "owned PID inventory missing or duplicated")
    gt, writer = by_pid[gt_pid][0], by_pid[archive_pid][0]
    parent_pid = _pid(gt.get("ppid"))
    require(len(by_pid[parent_pid]) == 1, "registered UE parent inventory missing or ambiguous")
    parent = by_pid[parent_pid][0]
    parent_args = shlex.split(parent.get("cmdline", ""))
    require(parent.get("role") == "ue" and parent_args[:3] ==
        ["/usr/bin/python3", "-m", registration.get("child_module")], "GT parent is not the registered UE child")
    for row in (gt, writer):
        require(row.get("role") is None and row.get("descendant_of_runner") is True
            and row.get("in_container") is False and row.get("environ_source") == "direct"
            and isinstance(row.get("variables"), dict), "owned child inventory is unreadable or foreign")
    _command(gt, expected_bootstrap_module)
    _command(writer, expected_archive_module, parent=gt_pid)
    require(writer.get("ppid") == gt_pid, "archive inventory parent differs")
    for key, expected in ENVIRONMENT.items():
        require(writer["variables"].get(key) == expected, "archive inventory caps/CUDA differ")
        permitted = (baseline.get(key), expected) if key in CAP_VARIABLES else (expected,)
        require(key in gt["variables"] and gt["variables"][key] in permitted,
                "GT exec-time inventory differs from baseline/bootstrap caps")
    return dict(gt_pid=gt_pid, archive_pid=archive_pid, ue_parent_pid=parent_pid,
        requests=len(requests), source_binding=expected_binding, factory=expected_factory,
        roles={str(gt_pid): "registered_cpu_gt", str(archive_pid): "registered_cpu_archive"},
        effective_environment=dict(ENVIRONMENT), native_contexts_checked=len(requests) + 2,
        gt_exec_snapshot={key: gt["variables"][key] for key in ENVIRONMENT},
        gt_effective_caps_source="NATIVE_READY_EVERY_RUN_AND_CLOSE",
        archive_caps_source="OWNED_PROCFS_BEFORE_SENSOR_STARTUP_AND_NATIVE_COMPLETION_CLOSE",
        inventory_sha256=canonical(inventory_rows),
        exception_scope="EXACT_SOURCE_BOUND_PID_PAIR_ONLY")


def classify_inventory(inventory_rows, *, baseline, children, inventory_errors=0):
    """Retain original checks for every process outside the verified pair."""
    roles, violations, unreadable, ue, exceptions = {}, [], [], [], []
    require(set(children.get("roles", {})) == {str(children.get("gt_pid")), str(children.get("archive_pid"))},
            "verified exact child pair required")
    require(children.get("inventory_sha256") == canonical(inventory_rows), "inventory changed after native child verification")
    rows = [row for row in inventory_rows if "summary" not in row]
    for row in rows:
        role = row.get("role") or ("runner_descendant" if row.get("descendant_of_runner") else "other")
        variables = row.get("variables")
        if variables is None:
            unreadable.append(dict(pid=row.get("pid"), role=role, comm=row.get("comm"), reason=row.get("environ_source")))
            continue
        if str(row.get("pid")) in children["roles"]:
            exceptions.append(dict(pid=row["pid"], original_role=role, classified_role=children["roles"][str(row["pid"])],
                observed_environment={key: variables.get(key) for key in ENVIRONMENT}))
            continue
        if role not in roles:
            roles[role] = {key: set() for key in CAP_VARIABLES}
        for key in CAP_VARIABLES:
            roles[role][key].add(variables.get(key))
        if row.get("role") == "ue":
            ue.append(dict(pid=row["pid"], variables={key: variables.get(key) for key in CAP_VARIABLES}))
            continue
        for key in CAP_VARIABLES:
            if variables.get(key) != baseline.get(key):
                violations.append(dict(pid=row["pid"], role=role, comm=row.get("comm"),
                    variable=key, value=variables.get(key), baseline=baseline.get(key)))
    require(len(exceptions) == 2, "registered exception population differs")
    observed_ok = not violations and bool(ue) and inventory_errors == 0
    return dict(rows=len(rows), errors=inventory_errors, scope_violations=violations, unreadable=unreadable,
        ue_processes=ue, registered_exceptions=exceptions,
        other_process_roles={role: {key: sorted(map(str, values)) for key, values in caps.items()}
                             for role, caps in sorted(roles.items())},
        observed_other_processes_baseline=observed_ok,
        all_processes_verified=observed_ok and not unreadable,
        coverage="READABLE_PROCESSES_ONLY" if unreadable else "COMPLETE_RECORDED_INVENTORY")


def _json(path, root):
    path, root = Path(path), Path(root).resolve(strict=True)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
        and not any(p.is_symlink() for p in path.parents if p.is_relative_to(root)), "unsafe evidence path")
    require(path.stat().st_size <= 32 * 1024 * 1024, "bounded process evidence required")
    return json.loads(path.read_bytes())


def verify_saved_scope(qualification_directory, *, source_root=None, expected_factory=FACTORY):
    """Read a sealed registration and genuine saved receipts; write nothing."""
    output = Path(qualification_directory).resolve(strict=True)
    record = _json(output / "PROSPECTIVE.json", output)
    source_root = Path(record["worktree"] if source_root is None else source_root).resolve(strict=True)
    registration = _json(output / "registration.json", output)
    require(sha(output / "registration.json") == record.get("registration_file_sha256")
        and registration.get("registration_sha256") == canonical({k: v for k, v in registration.items()
                                                                 if k != "registration_sha256"}),
        "prospective registration digest differs")
    worker, bootstrap = record.get("worker_sources"), record.get("bootstrap_sources")
    require(isinstance(worker, dict) and isinstance(bootstrap, dict), "registered process source maps missing")
    factory_path = expected_factory.rsplit(".", 1)[0].replace(".", "/") + ".py"
    sources = {**worker, **bootstrap}
    require(factory_path in record.get("runtime_sources", {}), "registered worker factory source missing")
    sources[factory_path] = record["runtime_sources"][factory_path]
    for name, digest in sources.items():
        relative = Path(name)
        path = source_root / relative
        require(not relative.is_absolute() and ".." not in relative.parts
            and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source_root)
            and not any(p.is_symlink() for p in path.parents if p.is_relative_to(source_root))
            and sha(path) == digest, "registered process source bytes differ: " + str(name))
    cells = sorted((output / "capture/cells").iterdir())
    require(len(cells) == 1 and cells[0].is_dir() and not cells[0].is_symlink(), "one owned qualification cell required")
    cell = cells[0]
    paths = dict(scope=cell / "GT_WORKER_SCOPE.json", timeline=cell / "CPU_GT_PROCESS_TIMELINE.json",
        optimization=cell / "GT_OPTIMIZATION_SCOPE.json", qualification=output / "capture/QUALIFICATION_RESULT.json")
    documents = {key: _json(path, output) for key, path in paths.items()}
    scope, timeline, optimization = (documents[key] for key in ("scope", "timeline", "optimization"))
    require(optimization.get("prospective_record_sha256") == sha(output / "PROSPECTIVE.json")
        and optimization.get("cpu_worker", {}).get("scope_sha256") == sha(paths["scope"])
        and optimization.get("cpu_worker", {}).get("timeline_sha256") == sha(paths["timeline"]),
        "native process evidence is not bound to the prospective scope")
    result = documents["qualification"]
    baseline = result.get("threadcap", {}).get("effective", {}).get("launch_restore", {}).get("restore")
    require(isinstance(baseline, dict) and set(baseline) == set(CAP_VARIABLES), "retained baseline restore settings missing")
    inventory = output / "capture/threadcap/PROCESS_ENV_INVENTORY.jsonl"
    require(inventory.is_file() and not inventory.is_symlink()
        and not any(p.is_symlink() for p in inventory.parents if p.is_relative_to(output))
        and inventory.stat().st_size <= 32 * 1024 * 1024,
            "bounded regular process inventory required")
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    summaries = [row["summary"] for row in rows if "summary" in row]
    require(len(summaries) == 1 and summaries[0] == result["threadcap"]["inventory"],
            "retained inventory summary differs from original qualification")
    children = verify_registered_children(registration, scope, timeline, rows, baseline=baseline,
        expected_worker_sources=worker, expected_bootstrap_sources=bootstrap,
        expected_binding=record.get("handoff_binding"), expected_factory=expected_factory)
    classified = classify_inventory(rows, baseline=baseline, children=children, inventory_errors=summaries[0]["errors"])
    return dict(schema=SCHEMA, passed=classified["observed_other_processes_baseline"],
        registered_children_verified=True, children=children, inventory=classified,
        original_e3=result["threadcap"]["checks"]["E3_SCOPE_OTHER_PROCESSES_BASELINE"],
        original_qualification_passed=result["passed"], original_checks=dict(result["threadcap"]["checks"]),
        original_gate_relabelled=False, live_qualified=False, deployed=False,
        evidence_sha256={**{key: sha(path) for key, path in paths.items()}, "inventory": sha(inventory),
            "registration": sha(output / "registration.json"), "prospective": sha(output / "PROSPECTIVE.json")},
        checked_source_files=len(sources), note="Separate offline classification; saved E3 and all qualification results remain authoritative and unchanged.")
