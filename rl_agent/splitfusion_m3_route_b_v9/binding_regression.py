"""Binding regression for v9: the v9 roles resolve v9, nothing older is rebound (importing v9 installs neither the
FREEZE repair, the tilt-aware AVO switch, the storage-accounting switch nor the offline P3 rule), the GT
service/worker selection is the v9 one, the v9 service trampoline is the v8 copy and the v9 worker trampoline is the
offline-qualified accounting trampoline plus the bounded timer, and the child/edge production composition is
identical to v8 (and v7/v6/v3) EXCEPT the two registered v9 scopes: the GOOD_ALL controller installer (v9 constant
23.5 dB) and the collector layer (bounded end-of-route drain above the unchanged v3 M3 layer).

Run from the deployment worktree (CPU only, no live service):

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v9.binding_regression \\
      --name binding_regression_01

V8 evidence is never modified: the rejection check runs on byte-identical copies of the V8 prospective's
registration files inside the new evidence directory.  Runtime checks run in fresh interpreters.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import contract as C
from .host import REGRESSION_SCHEMA

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v9"
V8_ROOT = ROOT / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261008_m3_route_b_v8/"
                  "good_all_prospective_01")
V8_FILES = ("PROSPECTIVE.json", "registration.json", "PREREQUISITE.json", "ACCEPTANCE.json", "arm.json",
            "compose_partition.yaml", C.RETENTION_PLAN_FILE)
OLDER = (*(f"splitfusion_m3_route_b_v{n}" for n in (1, 2, 3, 4, 5, 6, 7, 8)), "splitfusion_avo_tilt_v1",
         "splitfusion_numeric_gt_producer_v1", "splitfusion_gt_freeze_lifetime_v1",
         "splitfusion_gt_storage_accounting_v1", "splitfusion_numeric_gt_integration_v2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env():
    value = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    value.pop("PYTHONPATH", None)
    return value


def fresh(argv, log, timeout=600):
    proc = subprocess.run([sys.executable, "-B", *argv], cwd=ROOT, env=env(), capture_output=True, text=True,
                          timeout=timeout)
    Path(log).write_text("$ " + " ".join([sys.executable, "-B", *argv]) + "\n" + proc.stdout + proc.stderr)
    return proc


def _module_imports(tree):
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = node.module + "." + alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split(".")[0]] = alias.name
    return names


def foreign_rebinding(package_dir, older=OLDER):
    """Assignments to attributes of modules of older M3 versions (the v4 ``E.C = C`` pattern)."""
    hits = []
    for path in sorted(Path(package_dir).glob("*.py")):
        tree = ast.parse(path.read_text())
        imports = _module_imports(tree)
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, (ast.AugAssign, ast.AnnAssign)) else []
            for target in targets:
                for item in (target.elts if isinstance(target, ast.Tuple) else [target]):
                    if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name):
                        origin = imports.get(item.value.id, "")
                        if any(name in origin for name in older):
                            hits.append(f"{path.name}:{node.lineno}:{item.value.id}.{item.attr}")
    return hits


def _with_items(source_path, function):
    tree = ast.parse(Path(source_path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    found = [w for w in ast.walk(node) if isinstance(w, ast.With)]
    return [[ast.unparse(item) for item in w.items] for w in found]


# The registered, reversible substitutions v9 installs in its own processes.  ``OLDER`` includes the retained
# numeric-GT integration package, so every assignment to one of its globals must appear here by name; anything else
# is an unregistered rebinding and fails.  (The GT prep timer is installed and removed in the same function; the P3
# rule is the one the live GT workers install, here for the offline tools.)
REGISTERED_REBINDINGS = frozenset((
    "worker_entry.py:OWN.payload_storage_bytes",          # prep: the bounded per-call accounting timer, then removed
    "postflight_p3.py:READY._M3_AGREE_RECONSTRUCTED",     # offline P3, exactly as splitfusion_m3_route_b_v3.worker_entry
))


def unregistered_rebinding(package_dir, older=OLDER):
    """``foreign_rebinding`` minus the registered substitution sites (line numbers are not part of the identity)."""
    out = []
    for hit in foreign_rebinding(package_dir, older):
        name, line, attribute = hit.split(":", 2)
        if f"{name}:{attribute}" not in REGISTERED_REBINDINGS:
            out.append(hit)
    return out


# The only production-scope difference v9 registers, inside the single child ``with`` statement.  The GOOD_ALL
# controller installer is spelled identically but resolves to the native v9 installer (checked by the probe).
REGISTERED_SCOPE_CHANGES = {"RC.install_collector_layer()": "RD.install_collector_layer()"}


def composition_parity():
    """The child/edge scopes equal v8's except the registered v9 change, in the same order and count."""
    v3 = ROOT / "rl_agent/splitfusion_m3_route_b_v3/entries.py"
    v6 = ROOT / "rl_agent/splitfusion_m3_route_b_v6/entries.py"
    v7 = ROOT / "rl_agent/splitfusion_m3_route_b_v7/entries.py"
    v8 = ROOT / f"{C.V8.PACKAGE}/entries.py"
    v9 = ROOT / f"{C.PACKAGE}/entries.py"
    older, mine = _with_items(v8, "_child"), _with_items(v9, "_child")
    expected = [[REGISTERED_SCOPE_CHANGES.get(item, item) for item in group] for group in older]
    return dict(child_older_identical=older == _with_items(v7, "_child") == _with_items(v6, "_child")
                == _with_items(v3, "child"),
                child_matches_v8_except_registered_changes=mine == expected,
                child_changes=sorted({item for group, reference in zip(mine, older)
                                      for item in group if item not in reference}),
                edge=_with_items(v8, "edge") == _with_items(v9, "edge") == _with_items(v7, "edge")
                == _with_items(v6, "edge") == _with_items(v3, "edge"),
                child_items=mine, edge_items=_with_items(v9, "edge"))


def _node(path, name):
    tree = ast.parse(Path(path).read_text())
    return next(n for n in tree.body if getattr(n, "name", None) == name)


def service_entry_parity():
    """v9 service_entry = v8 service_entry (AST of the substitution class and of ``main``; only the docstring and the
    contract module they read differ)."""
    v8, v9 = ROOT / f"{C.V8.PACKAGE}/service_entry.py", ROOT / f"{C.PACKAGE}/service_entry.py"
    return dict(identical_class=ast.dump(_node(v8, "WorkerSubstitution")) == ast.dump(_node(v9, "WorkerSubstitution")),
                identical_main=ast.dump(_node(v8, "main")) == ast.dump(_node(v9, "main")))


CANDIDATE_IMPORT = "from . import accounting as A, switch"
V9_IMPORT = "from rl_agent.splitfusion_gt_storage_accounting_v1 import accounting as A, switch"
TIMER_STATEMENTS = ("timer = AccountingTimer(A.payload_storage_bytes, C.ACCOUNTING_TIMING_CAPACITY)",
                    "OWN.payload_storage_bytes = timer")
TIMER_NEEDLES = ("timer", "Timer", "TIMING")
# The one added statement that does not name the timer: it puts the qualified accounting function back as the owner
# global when the timer is removed, so the switch's own uninstall sees exactly what it installed.
TIMER_RESTORE_STATEMENT = "OWN.payload_storage_bytes = A.payload_storage_bytes"


def worker_entry_parity():
    """v9 worker_entry.main = the offline-qualified candidate trampoline
    (``splitfusion_gt_storage_accounting_v1.worker_entry.main``) with the candidate modules imported absolutely and
    exactly the bounded per-call timer added: every statement that is not in the candidate mentions the timer."""
    candidate = ROOT / f"{C.ACCOUNTING_PACKAGE}/worker_entry.py"
    mine = ast.unparse(_node(ROOT / f"{C.PACKAGE}/worker_entry.py", "main"))
    theirs = ast.unparse(_node(candidate, "main"))
    added = [line.strip() for line in mine.replace(V9_IMPORT, CANDIDATE_IMPORT).splitlines()
             if line.strip() not in theirs]
    return dict(candidate_sources_are_qualified=all(sha(ROOT / relative) == digest
                                                    for relative, digest in C.ACCOUNTING_QUALIFIED.items()),
                imports_the_candidate_absolutely=mine.count(V9_IMPORT) == 1 and theirs.count(CANDIDATE_IMPORT) == 1,
                only_addition_is_the_bounded_timer=bool(added) and all(
                    line == TIMER_RESTORE_STATEMENT or any(needle in line for needle in TIMER_NEEDLES)
                    for line in added),
                timer_restore_statement_present=TIMER_RESTORE_STATEMENT in added,
                timer_installed_and_removed=all(line in mine for line in TIMER_STATEMENTS),
                added_statements=added)


RESOLUTION_PROBE = """
import json
from rl_agent.splitfusion_m3_route_b_v9 import entries, host, retention, gc_seams, runner, service_entry, \
    worker_entry, route_drain, radio_telemetry, postflight_p3, good_all_controller
from rl_agent.splitfusion_m3_route_b_v8 import entries as v8_entries, gc_seams as v8_seams, retention as v8_ret, \
    host as v8_host, runner as v8_runner, service_entry as v8_service, worker_entry as v8_worker
from rl_agent.splitfusion_m3_route_b_v3 import entries as v3_entries, gc_seams as v3_seams, retention as v3_ret, \\
    host as v3_host, service_entry as v3_service, worker_entry as v3_worker
from rl_agent.splitfusion_m3_route_b_v6 import entries as v6_entries, gc_seams as v6_seams, retention as v6_ret, \\
    host as v6_host, runner as v6_runner
from rl_agent.splitfusion_m3_route_b_v7 import entries as v7_entries, gc_seams as v7_seams, retention as v7_ret, \\
    host as v7_host, runner as v7_runner, service_entry as v7_service, worker_entry as v7_worker
from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as R
from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA
from rl_agent.splitfusion_avo_tilt_v1 import switch as SW
from rl_agent.splitfusion_gt_storage_accounting_v1 import switch as ASW
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN, ready as READY
print(json.dumps(dict(
    v9=dict(entries_contract=entries.C.__name__, host_contract=host.C.__name__, retention_contract=retention.C.__name__,
            seam_contract=gc_seams.C.__name__, runner_contract=runner.C.__name__,
            service_entry_contract=service_entry.C.__name__, worker_entry_contract=worker_entry.C.__name__,
            verify_prospective=host.verify_prospective.__module__, load_observer=retention.load_observer.__module__,
            seam_parameters=gc_seams.seam_parameters.__module__, registration=entries.registration.__module__,
            validate_parent=runner.validate_parent.__module__, service_main=service_entry.main.__module__,
            worker_main=worker_entry.main.__module__, drain_contract=route_drain.C.__name__,
            telemetry_contract=radio_telemetry.C.__name__, postflight_contract=postflight_p3.C.__name__,
            controller_contract=good_all_controller.C.__name__,
            controller_installer=entries.install_good_all_controller.__module__,
            collector_layer=route_drain.install_collector_layer.__module__),
    v8_untouched=dict(entries=v8_entries.C.__name__, gc_seams=v8_seams.C.__name__, retention=v8_ret.C.__name__,
                      host=v8_host.C.__name__, runner=v8_runner.C.__name__, service_entry=v8_service.C.__name__,
                      worker_entry=v8_worker.C.__name__),
    schedule=dict(target_snr_db=entries.C.SCHEDULE["target_snr_db"],
                  rfsim=entries.C.SCHEDULE["expected_rfsim_command_db"],
                  controller_module=entries.C.CONTROLLER_MODULE,
                  v8_target_snr_db=v8_entries.C.SCHEDULE["target_snr_db"],
                  v8_controller_module=v8_entries.C.CONTROLLER_MODULE),
    v3_untouched=dict(entries=v3_entries.C.__name__, gc_seams=v3_seams.C.__name__, retention=v3_ret.C.__name__,
                      host=v3_host.C.__name__, service_entry=v3_service.C.__name__, worker_entry=v3_worker.C.__name__),
    v6_untouched=dict(entries=v6_entries.C.__name__, gc_seams=v6_seams.C.__name__, retention=v6_ret.C.__name__,
                      host=v6_host.C.__name__, runner=v6_runner.C.__name__),
    v7_untouched=dict(entries=v7_entries.C.__name__, gc_seams=v7_seams.C.__name__, retention=v7_ret.C.__name__,
                      host=v7_host.C.__name__, runner=v7_runner.C.__name__, service_entry=v7_service.C.__name__,
                      worker_entry=v7_worker.C.__name__),
    prep_core_unchanged_by_import=PC.PrepCore is R.ORIGINAL_PREP_CORE,
    numeric_avo_unchanged_by_import=NA.numeric_avo is SW.ORIGINAL_NUMERIC_AVO and not SW.installed(),
    accounting_unchanged_by_import=OWN.payload_storage_bytes is ASW.ORIGINAL_PAYLOAD_STORAGE_BYTES
        and not ASW.installed(),
    offline_p3_not_installed_by_import=not hasattr(READY, "_M3_AGREE_RECONSTRUCTED")), sort_keys=True))
"""


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "v8_prospective_copy"
    fixture.mkdir()
    copied = {}
    for filename in V8_FILES:
        shutil.copy2(V8_ROOT / filename, fixture / filename)
        copied[filename] = sha(V8_ROOT / filename)
    identical = all(sha(fixture / filename) == digest for filename, digest in copied.items())
    checks, details = {}, {}

    # 1-2. v9 resolution in a fresh interpreter; importing v9 rebinds nothing older and installs nothing
    proc = fresh(["-c", RESOLUTION_PROBE], out / "resolution_probe.log")
    resolution = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    v9c = C.MODULE + ".contract"
    checks["v9_functions_resolve_v9"] = bool(resolution) and all(
        value.startswith(C.MODULE) for value in resolution["v9"].values()) and resolution["v9"]["host_contract"] == v9c
    checks["older_globals_untouched"] = bool(resolution) and (
        set(resolution["v3_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v3.contract"}
        and set(resolution["v6_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v6.contract"}
        and set(resolution["v7_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v7.contract"}
        and set(resolution["v8_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v8.contract"}
        and resolution["prep_core_unchanged_by_import"] is True
        and resolution["numeric_avo_unchanged_by_import"] is True
        and resolution["accounting_unchanged_by_import"] is True
        and resolution["offline_p3_not_installed_by_import"] is True)
    details["resolution"] = resolution

    # 3-5. static: no older M3/candidate global rebound; child/edge scopes = v8 except the registered v9 change
    hits = unregistered_rebinding(ROOT / C.PACKAGE)
    checks["no_unregistered_global_rebinding"] = hits == []
    registered = {h.split(":", 2)[0] + ":" + h.split(":", 2)[2] for h in foreign_rebinding(ROOT / C.PACKAGE)}
    checks["registered_rebindings_are_exactly_the_declared_ones"] = registered == set(REGISTERED_REBINDINGS)
    parity = composition_parity()
    checks["child_and_edge_scopes_v8_plus_registered_changes"] = (
        parity["child_older_identical"] and parity["child_matches_v8_except_registered_changes"] and parity["edge"]
        and parity["child_changes"] == sorted(REGISTERED_SCOPE_CHANGES.values()))
    audit = C.role_constant_audit()
    checks["retained_role_constants_identical"] = all(audit.values())
    details.update(unregistered_rebinding=hits, registered_rebindings=sorted(registered), composition=parity,
                   role_constant_audit=audit)

    # 6. the v9 GT trampolines: the v8 service copy; the qualified accounting trampoline plus the bounded timer
    service = service_entry_parity()
    checks["service_entry_identical_to_v8"] = all(service.values())
    worker = worker_entry_parity()
    checks["worker_entry_is_the_qualified_accounting_trampoline_plus_timer"] = all(
        value for key, value in worker.items() if key != "added_statements")
    details.update(service_entry_parity=service, worker_entry_parity=worker)

    # 7. GT selection: the seam starts the v9 service; the v9 service starts the v9 worker entry for BOTH roles
    from .gc_seams import rewrite_service_argv, seam_parameters
    from .service_entry import WorkerSubstitution
    seam_argv = rewrite_service_argv(["python3", "-m", C.SERVICE_MODULE], arm="candidate", trace_dir="/t",
                                     max_frames=C.FRAME_CEILING, producer="v2")
    substitution = WorkerSubstitution(arm="candidate", trace_dir="/t", max_frames=C.FRAME_CEILING, producer="v2")
    worker_argv = {}
    for role in ("prep", "score"):
        substitution._role.value = role
        worker_argv[role] = substitution.rewrite(["python3", "-m", C.WORKER_MODULE, "--fd", "3", "--parent", "1"])
    substitution._role.value = None
    checks["gt_selection_is_v9"] = (seam_argv[2] == C.SERVICE_ENTRY
                                    and all(argv[2] == C.WORKER_ENTRY for argv in worker_argv.values()))
    details.update(seam_argv=seam_argv, worker_argv=worker_argv)

    # 8. the V8 registration's GT trampolines are refused by the v9 seam
    v8_reg = json.loads((fixture / "registration.json").read_bytes())
    try:
        seam_parameters(dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=v8_reg["gt_gc"]))
        refused = None
    except ValueError as exc:
        refused = str(exc)
    checks["v8_gt_trampolines_refused"] = refused == "registered GT service/worker trampolines differ from the v9 contract"
    details["v8_gt_trampolines"] = v8_reg["gt_gc"]["trampolines"]

    # 9. the actual V8 prospective is rejected by the v9 child validation (fresh process)
    proc = fresh(["-m", C.CHILD, C.VERIFY_FLAG, "--output", str(fixture)], out / "v9_child_on_v8.log")
    checks["v9_child_rejects_v8_registration"] = proc.returncode != 0 and "ValueError" in proc.stderr
    details["v9_child_on_v8_last_line"] = (proc.stdout + proc.stderr).rstrip().splitlines()[-1]

    # 10-11. the repair and the tilt-aware AVO are byte-identical to their offline-qualified sources (v8, unchanged)
    checks["prep_repair_is_qualified_source"] = all(sha(ROOT / relative) == digest
                                                    for relative, digest in C.PREP_LIFETIME_QUALIFIED.items())
    checks["avo_tilt_is_qualified_source"] = all(sha(ROOT / relative) == digest
                                                 for relative, digest in C.AVO_TILT_QUALIFIED.items())

    # 12. the storage-accounting candidate is the offline-qualified source, registered in the prep role only
    checks["storage_accounting_is_qualified_source"] = all(sha(ROOT / relative) == digest
                                                           for relative, digest in C.ACCOUNTING_QUALIFIED.items())
    gc = C.gc_registration("candidate", C.FRAME_CEILING, "v2")
    accounting = gc["storage_accounting"]
    checks["prep_substitutions_registered_in_prep_only"] = (
        accounting["roles"] == ["prep"] and accounting["sources"] == C.ACCOUNTING_QUALIFIED
        and accounting["cache"] is None and accounting["reservation_bytes_unchanged"] is True
        and gc["avo_geometry"]["roles"] == ["prep"] and gc["avo_geometry"] == C.V8.avo_registration()
        and gc["prep_freeze_lifetime"]["roles"] == ["prep"])

    # 13. GOOD_ALL is the qualified 23.5 dB anchor, actor input and controller consistent, nothing else in the schedule
    channel = C.channel_registration(ROOT)
    schedule = resolution.get("schedule") or {}
    checks["good_all_is_the_qualified_anchor"] = (
        C.SCHEDULE["target_snr_db"] == 23.5 and C.SCHEDULE["expected_rfsim_command_db"] == -12.0
        and C.SCHEDULE["mapping_point"] == "PHASE14A_100MHZ_ANCHOR_01"
        and channel["actor_input"]["value"] == C.ACTOR_SNR_FEATURE
        and channel["actor_input"]["normalisation_changed"] is False
        and channel["actor_input"]["gnb_measurement_to_actor"] is False
        and channel["carrier_cell"] == C.V8.channel_registration(ROOT)["carrier_cell"]
        and sorted(k for k in set(C.SCHEDULE) | set(C.V8.SCHEDULE)
                   if C.SCHEDULE.get(k) != C.V8.SCHEDULE.get(k)) == sorted(C.SCHEDULE_FIELDS_CHANGED)
        and schedule.get("target_snr_db") == 23.5 and schedule.get("rfsim") == -12.0
        and schedule.get("controller_module") == C.CONTROLLER_MODULE
        and schedule.get("v8_target_snr_db") == C.V8.SCHEDULE["target_snr_db"])
    details["channel"] = dict(schedule=channel["schedule"], actor_input=channel["actor_input"],
                              changed_from_v8=channel["changed_from_v8"])

    # 14. the calibrated radio telemetry binds the tracer database, and the MAC-log statistic stays separate
    telemetry = C.radio_telemetry_registration(ROOT)
    checks["radio_telemetry_binds_the_tracer_database"] = (
        telemetry["database_sha256"] == C.T_MESSAGES_SHA256
        and telemetry["quantities"]["mac_log_snr"]["observer"] == C.MAC_LOG_OBSERVER
        and telemetry["quantities"]["mac_log_snr"]["kept"] is True
        and telemetry["actor_input"].startswith("unchanged")
        and not any(telemetry["decision_path"].values()))
    details["radio_telemetry"] = telemetry

    # 15. the bookkeeping repairs are registered and V8's evidence is untouched
    bookkeeping = C.bookkeeping_registration()
    checks["bookkeeping_repairs_registered"] = (
        bookkeeping["end_of_route_drain"]["bounds"]["ack_horizon"].startswith("unchanged")
        and "170 ms decision deadline and reward resolution" in bookkeeping["end_of_route_drain"]["unchanged"]
        and len(bookkeeping["offline_p3"]["covers"]) == 2
        and bookkeeping["offline_p3"]["rule"] == C.POSE
        and resolution.get("v9", {}).get("collector_layer") == C.ROUTE_DRAIN_MODULE)
    details["bookkeeping"] = bookkeeping

    passed = all(checks.values())
    result = dict(schema=REGRESSION_SCHEMA, status="PASS" if passed else "FAIL", checks=checks, details=details,
                  v8_root=str(V8_ROOT), v8_files=copied, v8_fixture_identical=identical,
                  launcher_binding=C.launcher_binding(), sources=C.exact_sources(ROOT), interpreter=sys.executable,
                  live_services_started=False, carla_started=False, oai_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"], checks=checks), indent=1, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
