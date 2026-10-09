"""Binding regression for v10: the v10 roles resolve v10, nothing older is rebound (importing v10 installs neither the
FREEZE repair, the tilt-aware AVO switch, the storage-accounting switch nor the offline P3 rule), the GT
service/worker selection is the v10 one, the v10 service trampoline is the v9/v8 copy and the v10 worker trampoline is
the offline-qualified CORRECTED (V2) accounting trampoline plus the bounded timer, and the child/edge production
composition is IDENTICAL to v9's (and through it v8/v7/v6/v3): v10 registers no production-scope change at all.

Run from the deployment worktree (CPU only, no live service):

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v10.binding_regression \\
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

from rl_agent.splitfusion_gt_storage_accounting_v2 import accounting as ACCOUNTING_MODULE

from . import contract as C
from .host import REGRESSION_SCHEMA

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v10"
V9_ROOT = ROOT / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v9/"
                  "good_all_prospective_02")
V9_FILES = ("PROSPECTIVE.json", "registration.json", "PREREQUISITE.json", "ACCEPTANCE.json", "arm.json",
            "compose_partition.yaml", C.RETENTION_PLAN_FILE)
OLDER = (*(f"splitfusion_m3_route_b_v{n}" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9)), "splitfusion_avo_tilt_v1",
         "splitfusion_numeric_gt_producer_v1", "splitfusion_gt_freeze_lifetime_v1",
         "splitfusion_gt_storage_accounting_v1", "splitfusion_gt_storage_accounting_v2",
         "splitfusion_numeric_gt_integration_v2")


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


# The registered, reversible substitutions v10 installs in its own processes.  ``OLDER`` includes the retained
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


# v10 registers NO production-scope change: the child/edge ``with`` statements must equal v9's exactly.
REGISTERED_SCOPE_CHANGES = {}


def composition_parity():
    """The child/edge scopes equal v9's exactly, in the same order and count (and v9's equal v8's + its drain)."""
    v3 = ROOT / "rl_agent/splitfusion_m3_route_b_v3/entries.py"
    v6 = ROOT / "rl_agent/splitfusion_m3_route_b_v6/entries.py"
    v7 = ROOT / "rl_agent/splitfusion_m3_route_b_v7/entries.py"
    v8 = ROOT / f"{C.V8.PACKAGE}/entries.py"
    v9 = ROOT / "rl_agent/splitfusion_m3_route_b_v9/entries.py"
    v10 = ROOT / f"{C.PACKAGE}/entries.py"
    older, mine = _with_items(v9, "_child"), _with_items(v10, "_child")
    return dict(child_older_identical=_with_items(v8, "_child") == _with_items(v7, "_child")
                == _with_items(v6, "_child") == _with_items(v3, "child"),
                child_matches_v9_exactly=mine == older,
                child_changes=sorted({item for group, reference in zip(mine, older)
                                      for item in group if item not in reference}),
                edge=_with_items(v9, "edge") == _with_items(v10, "edge") == _with_items(v8, "edge")
                == _with_items(v7, "edge") == _with_items(v6, "edge") == _with_items(v3, "edge"),
                child_items=mine, edge_items=_with_items(v10, "edge"))


def _node(path, name):
    tree = ast.parse(Path(path).read_text())
    return next(n for n in tree.body if getattr(n, "name", None) == name)


def service_entry_parity():
    """v10 service_entry = v9 service_entry = v8 service_entry (AST of the substitution class and of ``main``; only
    the docstring and the contract module they read differ)."""
    v8, v10 = ROOT / f"{C.V8.PACKAGE}/service_entry.py", ROOT / f"{C.PACKAGE}/service_entry.py"
    v9 = ROOT / "rl_agent/splitfusion_m3_route_b_v9/service_entry.py"
    return dict(identical_class=ast.dump(_node(v8, "WorkerSubstitution")) == ast.dump(_node(v10, "WorkerSubstitution"))
                == ast.dump(_node(v9, "WorkerSubstitution")),
                identical_main=ast.dump(_node(v8, "main")) == ast.dump(_node(v10, "main"))
                == ast.dump(_node(v9, "main")))


CANDIDATE_IMPORT = "from . import accounting as A, switch"
V10_IMPORT = "from rl_agent.splitfusion_gt_storage_accounting_v2 import accounting as A, switch"
TIMER_STATEMENTS = ("timer = AccountingTimer(session, C.ACCOUNTING_TIMING_CAPACITY)",
                    "OWN.payload_storage_bytes = timer")
TIMER_NEEDLES = ("timer", "Timer", "TIMING", "accounting_session")
# The registered refusal: V2's snapshot/owner shortcuts need CPython's dict mutation tag.  Without it they fail
# closed (still exact, but the slow traversal), which would silently invalidate this revision's performance
# qualification, so the prep worker refuses to start instead.
FAST_PATH_NEEDLES = ("fast_path", "ACCOUNTING_REQUIRES_DICT_VERSION_TAG", "dict mutation tag")
# The one added statement that names neither the timer nor the capability: it puts the installed session back as the
# owner global when the timer is removed, so the switch's own uninstall sees exactly what it installed.
TIMER_RESTORE_STATEMENT = "OWN.payload_storage_bytes = session"
# The only statement of the candidate that v10 does not keep verbatim: its binding record, superseded by the v10
# record (same fields plus the timer, cache and capability evidence).
CANDIDATE_RECORD_PREFIX = "record = dict(schema='splitfusion.gt_storage_accounting_v2.binding.v1'"
V10_RECORD_PREFIX = "record = dict(schema='splitfusion.m3_route_b_v10.storage_accounting_binding.v1'"


def worker_entry_parity():
    """v10 worker_entry.main = the offline-qualified CORRECTED candidate trampoline
    (``splitfusion_gt_storage_accounting_v2.worker_entry.main``) with the candidate modules imported absolutely, the
    binding record extended, the bounded per-call timer added and the fast-path capability refusal added: every
    statement that is not in the candidate is one of those."""
    candidate = ROOT / f"{C.ACCOUNTING_PACKAGE}/worker_entry.py"
    mine = ast.unparse(_node(ROOT / f"{C.PACKAGE}/worker_entry.py", "main"))
    theirs = ast.unparse(_node(candidate, "main"))
    normalised = mine.replace(V10_IMPORT, CANDIDATE_IMPORT)
    added = [line.strip() for line in normalised.splitlines() if line.strip() not in theirs]
    removed = [line.strip() for line in theirs.splitlines() if line.strip() not in normalised]

    def accounted(line):
        return (line == TIMER_RESTORE_STATEMENT or line.startswith(V10_RECORD_PREFIX)
                or any(needle in line for needle in TIMER_NEEDLES + FAST_PATH_NEEDLES))
    return dict(candidate_sources_are_qualified=all(sha(ROOT / relative) == digest
                                                    for relative, digest in C.ACCOUNTING_QUALIFIED.items()),
                v1_base_sources_are_qualified=all(sha(ROOT / relative) == digest
                                                  for relative, digest in C.ACCOUNTING_V1_QUALIFIED.items()),
                imports_the_candidate_absolutely=mine.count(V10_IMPORT) == 1 and theirs.count(CANDIDATE_IMPORT) == 1,
                only_additions_are_timer_record_and_capability=bool(added) and all(accounted(l) for l in added),
                only_removal_is_the_candidate_record=len(removed) == 1
                and removed[0].startswith(CANDIDATE_RECORD_PREFIX),
                timer_restore_statement_present=TIMER_RESTORE_STATEMENT in added,
                timer_installed_and_removed=all(line in mine for line in TIMER_STATEMENTS),
                fast_path_refusal_present=any(any(n in l for n in FAST_PATH_NEEDLES) for l in added)
                and C.ACCOUNTING_REQUIRES_DICT_VERSION_TAG is True,
                session_cache_recorded_and_closed="record['accounting_session'] = session.snapshot()" in mine
                and "switch.uninstall()" in mine,
                added_statements=added, removed_statements=removed)


RESOLUTION_PROBE = """
import json
from rl_agent.splitfusion_m3_route_b_v10 import entries, host, retention, gc_seams, runner, service_entry, \
    worker_entry, route_drain, radio_telemetry, postflight_p3, good_all_controller
from rl_agent.splitfusion_m3_route_b_v8 import entries as v8_entries, gc_seams as v8_seams, retention as v8_ret, \
    host as v8_host, runner as v8_runner, service_entry as v8_service, worker_entry as v8_worker
from rl_agent.splitfusion_m3_route_b_v9 import entries as v9_entries, gc_seams as v9_seams, retention as v9_ret, \
    host as v9_host, runner as v9_runner, service_entry as v9_service, worker_entry as v9_worker
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
from rl_agent.splitfusion_gt_storage_accounting_v1 import switch as ASW1
from rl_agent.splitfusion_gt_storage_accounting_v2 import switch as ASW
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN, ready as READY
print(json.dumps(dict(
    v10=dict(entries_contract=entries.C.__name__, host_contract=host.C.__name__, retention_contract=retention.C.__name__,
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
    v9_untouched=dict(entries=v9_entries.C.__name__, gc_seams=v9_seams.C.__name__, retention=v9_ret.C.__name__,
                      host=v9_host.C.__name__, runner=v9_runner.C.__name__, service_entry=v9_service.C.__name__,
                      worker_entry=v9_worker.C.__name__),
    schedule=dict(target_snr_db=entries.C.SCHEDULE["target_snr_db"],
                  rfsim=entries.C.SCHEDULE["expected_rfsim_command_db"],
                  controller_module=entries.C.CONTROLLER_MODULE,
                  v8_target_snr_db=v8_entries.C.SCHEDULE["target_snr_db"],
                  v8_controller_module=v8_entries.C.CONTROLLER_MODULE,
                  v9_schedule_identical=entries.C.SCHEDULE == v9_entries.C.SCHEDULE,
                  v9_controller_module=v9_entries.C.CONTROLLER_MODULE),
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
        and not ASW.installed() and not ASW1.installed()
        and ASW1.ORIGINAL_PAYLOAD_STORAGE_BYTES is ASW.ORIGINAL_PAYLOAD_STORAGE_BYTES,
    accounting_fast_path_available=__import__(
        "rl_agent.splitfusion_gt_storage_accounting_v2.accounting", fromlist=["accounting"])._dict_version is not None,
    interpreter=__import__("sys").executable,
    offline_p3_not_installed_by_import=not hasattr(READY, "_M3_AGREE_RECONSTRUCTED")), sort_keys=True))
"""


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "v9_prospective_copy"
    fixture.mkdir()
    copied = {}
    for filename in V9_FILES:
        shutil.copy2(V9_ROOT / filename, fixture / filename)
        copied[filename] = sha(V9_ROOT / filename)
    identical = all(sha(fixture / filename) == digest for filename, digest in copied.items())
    checks, details = {}, {}

    # 1-2. v10 resolution in a fresh interpreter; importing v10 rebinds nothing older and installs nothing
    proc = fresh(["-c", RESOLUTION_PROBE], out / "resolution_probe.log")
    resolution = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    v10c = C.MODULE + ".contract"
    checks["v10_functions_resolve_v10"] = bool(resolution) and all(
        value.startswith(C.MODULE) for value in resolution["v10"].values()) and resolution["v10"]["host_contract"] == v10c
    checks["older_globals_untouched"] = bool(resolution) and (
        set(resolution["v3_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v3.contract"}
        and set(resolution["v6_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v6.contract"}
        and set(resolution["v7_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v7.contract"}
        and set(resolution["v8_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v8.contract"}
        and set(resolution["v9_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v9.contract"}
        and resolution["prep_core_unchanged_by_import"] is True
        and resolution["numeric_avo_unchanged_by_import"] is True
        and resolution["accounting_unchanged_by_import"] is True
        and resolution["offline_p3_not_installed_by_import"] is True)
    checks["accounting_fast_path_available_in_a_fresh_interpreter"] = (
        bool(resolution) and resolution.get("accounting_fast_path_available") is True)
    details["resolution"] = resolution

    # 3-5. static: no older M3/candidate global rebound; child/edge scopes = v8 except the registered v10 change
    hits = unregistered_rebinding(ROOT / C.PACKAGE)
    checks["no_unregistered_global_rebinding"] = hits == []
    registered = {h.split(":", 2)[0] + ":" + h.split(":", 2)[2] for h in foreign_rebinding(ROOT / C.PACKAGE)}
    checks["registered_rebindings_are_exactly_the_declared_ones"] = registered == set(REGISTERED_REBINDINGS)
    parity = composition_parity()
    checks["child_and_edge_scopes_identical_to_v9"] = (
        parity["child_older_identical"] and parity["child_matches_v9_exactly"] and parity["edge"]
        and parity["child_changes"] == sorted(REGISTERED_SCOPE_CHANGES.values()) == [])
    audit = C.role_constant_audit()
    checks["retained_role_constants_identical"] = all(audit.values())
    details.update(unregistered_rebinding=hits, registered_rebindings=sorted(registered), composition=parity,
                   role_constant_audit=audit)

    # 6. the v10 GT trampolines: the v8 service copy; the qualified accounting trampoline plus the bounded timer
    service = service_entry_parity()
    checks["service_entry_identical_to_v9_and_v8"] = all(service.values())
    worker = worker_entry_parity()
    checks["worker_entry_is_the_corrected_accounting_trampoline_plus_timer"] = all(
        value for key, value in worker.items() if key not in ("added_statements", "removed_statements"))
    details.update(service_entry_parity=service, worker_entry_parity=worker)

    # 7. GT selection: the seam starts the v10 service; the v10 service starts the v10 worker entry for BOTH roles
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
    checks["gt_selection_is_v10"] = (seam_argv[2] == C.SERVICE_ENTRY
                                    and all(argv[2] == C.WORKER_ENTRY for argv in worker_argv.values()))
    details.update(seam_argv=seam_argv, worker_argv=worker_argv)

    # 8. the V9 registration's GT trampolines are refused by the v10 seam
    v9_reg = json.loads((fixture / "registration.json").read_bytes())
    try:
        seam_parameters(dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=v9_reg["gt_gc"]))
        refused = None
    except ValueError as exc:
        refused = str(exc)
    checks["v9_gt_trampolines_refused"] = refused == "registered GT service/worker trampolines differ from the v10 contract"
    details["v9_gt_trampolines"] = v9_reg["gt_gc"]["trampolines"]
    details["v9_accounting_registration"] = v9_reg["gt_gc"]["storage_accounting"]

    # 9. the actual V9 prospective is rejected by the v10 child validation (fresh process)
    proc = fresh(["-m", C.CHILD, C.VERIFY_FLAG, "--output", str(fixture)], out / "v10_child_on_v9.log")
    checks["v10_child_rejects_v9_registration"] = proc.returncode != 0 and "ValueError" in proc.stderr
    details["v10_child_on_v9_last_line"] = (proc.stdout + proc.stderr).rstrip().splitlines()[-1]

    # 10-11. the repair and the tilt-aware AVO are byte-identical to their offline-qualified sources (v8, unchanged)
    checks["prep_repair_is_qualified_source"] = all(sha(ROOT / relative) == digest
                                                    for relative, digest in C.PREP_LIFETIME_QUALIFIED.items())
    checks["avo_tilt_is_qualified_source"] = all(sha(ROOT / relative) == digest
                                                 for relative, digest in C.AVO_TILT_QUALIFIED.items())

    # 12. the corrected storage-accounting candidate is the offline-qualified source, prep role only, bounded caches
    checks["storage_accounting_is_qualified_source"] = (
        all(sha(ROOT / relative) == digest for relative, digest in C.ACCOUNTING_QUALIFIED.items())
        and all(sha(ROOT / relative) == digest for relative, digest in C.ACCOUNTING_V1_QUALIFIED.items())
        and C.ACCOUNTING_VERSION == "splitfusion_gt_storage_accounting_v2.bounded_pair_history_summary.v2")
    gc = C.gc_registration("candidate", C.FRAME_CEILING, "v2")
    accounting = gc["storage_accounting"]
    cache = accounting["cache"]
    checks["prep_substitutions_registered_in_prep_only"] = (
        accounting["roles"] == ["prep"] and accounting["sources"] == C.ACCOUNTING_QUALIFIED
        and accounting["v1_sources"] == C.ACCOUNTING_V1_QUALIFIED
        and accounting["reservation_bytes_unchanged"] is True and accounting["capacity_rule_unchanged"] is True
        and gc["avo_geometry"]["roles"] == ["prep"] and gc["avo_geometry"] == C.V8.avo_registration()
        and gc["prep_freeze_lifetime"]["roles"] == ["prep"])
    checks["accounting_caches_are_bounded_and_registered"] = (
        cache["pair_history_entries_max"] == C.ACCOUNTING_MAX_ENTRIES == ACCOUNTING_MODULE.DEFAULT_MAX_ENTRIES
        and cache["rollback_snapshots_max"] == C.ACCOUNTING_MAX_SNAPSHOTS
        and cache["requires_dict_version_tag"] is True and "fail closed" in cache["mutation_guard"]
        and cache["cleanup_record"].endswith(":accounting_session")
        and gc["storage_accounting"]["substitution"]["replacement"].endswith(":AccountingSession")
        and gc["storage_accounting"]["substitution"]["process"] == "GT prep worker only")
    details["accounting"] = accounting

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
        and schedule.get("v8_target_snr_db") == C.V8.SCHEDULE["target_snr_db"]
        and schedule.get("v9_schedule_identical") is True
        and schedule.get("v9_controller_module") != C.CONTROLLER_MODULE)
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
        and resolution.get("v10", {}).get("collector_layer") == C.ROUTE_DRAIN_MODULE)
    details["bookkeeping"] = bookkeeping

    passed = all(checks.values())
    result = dict(schema=REGRESSION_SCHEMA, status="PASS" if passed else "FAIL", checks=checks, details=details,
                  v9_root=str(V9_ROOT), v9_files=copied, v9_fixture_identical=identical,
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
