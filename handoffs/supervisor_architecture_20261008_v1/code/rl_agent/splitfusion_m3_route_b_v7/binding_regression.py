"""Binding regression for v7: the v7 roles resolve v7, nothing older is rebound, the GT service/worker selection is
the v7 one, and the child/edge production composition is identical to v6 (and v3).

Run from the deployment worktree (CPU only, no live service):

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v7.binding_regression \
      --name binding_regression_01

V6 evidence is never modified: the rejection check runs on byte-identical copies of the V6 prospective's
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
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v7"
V6_ROOT = ROOT / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_m3_route_b_v6/"
                  "good_all_prospective_02")
V6_FILES = ("PROSPECTIVE.json", "registration.json", "PREREQUISITE.json", "ACCEPTANCE.json", "arm.json",
            "compose_partition.yaml", C.RETENTION_PLAN_FILE)
OLDER = tuple(f"splitfusion_m3_route_b_v{n}" for n in (1, 2, 3, 4, 5, 6))


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


def composition_parity():
    v3 = ROOT / "rl_agent/splitfusion_m3_route_b_v3/entries.py"
    v6 = ROOT / "rl_agent/splitfusion_m3_route_b_v6/entries.py"
    v7 = ROOT / f"{C.PACKAGE}/entries.py"
    return dict(child=_with_items(v6, "_child") == _with_items(v7, "_child") == _with_items(v3, "child"),
                edge=_with_items(v6, "edge") == _with_items(v7, "edge") == _with_items(v3, "edge"),
                child_items=_with_items(v7, "_child"), edge_items=_with_items(v7, "edge"))


def _node(path, name):
    tree = ast.parse(Path(path).read_text())
    return next(n for n in tree.body if getattr(n, "name", None) == name)


def service_entry_parity():
    """v7 service_entry = v3 service_entry except: the capacity import names v3 explicitly; the record names the
    entry and contract modules.  The substitution class is AST-identical."""
    v3, v7 = ROOT / "rl_agent/splitfusion_m3_route_b_v3/service_entry.py", ROOT / f"{C.PACKAGE}/service_entry.py"
    same_class = ast.dump(_node(v3, "WorkerSubstitution")) == ast.dump(_node(v7, "WorkerSubstitution"))
    main7 = ast.unparse(_node(v7, "main"))
    for after, before in (("from rl_agent.splitfusion_m3_route_b_v3 import capacity as CAP", "from . import capacity as CAP"),
                          ("pid=os.getpid(), entry_module=C.SERVICE_ENTRY, contract_module=C.__name__)",
                           "pid=os.getpid())")):
        if main7.count(after) != 1:
            return dict(identical_class=same_class, main_differs_only_as_registered=False)
        main7 = main7.replace(after, before)
    return dict(identical_class=same_class, main_differs_only_as_registered=main7 == ast.unparse(_node(v3, "main")))


RESOLUTION_PROBE = """
import json
from rl_agent.splitfusion_m3_route_b_v7 import entries, host, retention, gc_seams, runner, service_entry, worker_entry
from rl_agent.splitfusion_m3_route_b_v3 import entries as v3_entries, gc_seams as v3_seams, retention as v3_ret, \\
    host as v3_host, service_entry as v3_service, worker_entry as v3_worker
from rl_agent.splitfusion_m3_route_b_v6 import entries as v6_entries, gc_seams as v6_seams, retention as v6_ret, \\
    host as v6_host, runner as v6_runner
from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as R
print(json.dumps(dict(
    v7=dict(entries_contract=entries.C.__name__, host_contract=host.C.__name__, retention_contract=retention.C.__name__,
            seam_contract=gc_seams.C.__name__, runner_contract=runner.C.__name__,
            service_entry_contract=service_entry.C.__name__, worker_entry_contract=worker_entry.C.__name__,
            verify_prospective=host.verify_prospective.__module__, load_observer=retention.load_observer.__module__,
            seam_parameters=gc_seams.seam_parameters.__module__, registration=entries.registration.__module__,
            validate_parent=runner.validate_parent.__module__, service_main=service_entry.main.__module__,
            worker_main=worker_entry.main.__module__),
    v3_untouched=dict(entries=v3_entries.C.__name__, gc_seams=v3_seams.C.__name__, retention=v3_ret.C.__name__,
                      host=v3_host.C.__name__, service_entry=v3_service.C.__name__, worker_entry=v3_worker.C.__name__),
    v6_untouched=dict(entries=v6_entries.C.__name__, gc_seams=v6_seams.C.__name__, retention=v6_ret.C.__name__,
                      host=v6_host.C.__name__, runner=v6_runner.C.__name__),
    prep_core_unchanged_by_import=PC.PrepCore is R.ORIGINAL_PREP_CORE), sort_keys=True))
"""


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "v6_prospective_copy"
    fixture.mkdir()
    copied = {}
    for filename in V6_FILES:
        shutil.copy2(V6_ROOT / filename, fixture / filename)
        copied[filename] = sha(V6_ROOT / filename)
    identical = all(sha(fixture / filename) == digest for filename, digest in copied.items())
    checks, details = {}, {}

    # 1-2. v7 resolution in a fresh interpreter; importing v7 rebinds nothing in v3/v6 and installs no repair
    proc = fresh(["-c", RESOLUTION_PROBE], out / "resolution_probe.log")
    resolution = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    v7c = C.MODULE + ".contract"
    checks["v7_functions_resolve_v7"] = bool(resolution) and all(
        value.startswith(C.MODULE) for value in resolution["v7"].values()) and resolution["v7"]["host_contract"] == v7c
    checks["older_globals_untouched"] = bool(resolution) and (
        set(resolution["v3_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v3.contract"}
        and set(resolution["v6_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v6.contract"}
        and resolution["prep_core_unchanged_by_import"] is True)
    details["resolution"] = resolution

    # 3-5. static: no older M3 global rebound; child/edge scopes identical to v6 and v3; retained constants identical
    hits = foreign_rebinding(ROOT / C.PACKAGE)
    checks["no_foreign_global_rebinding"] = hits == []
    parity = composition_parity()
    checks["child_and_edge_scopes_identical_to_v6"] = parity["child"] and parity["edge"]
    audit = C.role_constant_audit()
    checks["retained_role_constants_identical"] = all(audit.values())
    details.update(foreign_rebinding=hits, composition=parity, role_constant_audit=audit)

    # 6. the v7 service trampoline is the v3 one except for the registered identity differences
    service = service_entry_parity()
    checks["service_entry_copy_differs_only_as_registered"] = all(service.values())
    details["service_entry_parity"] = service

    # 7. GT selection: the seam starts the v7 service; the v7 service starts the v7 worker entry for BOTH roles
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
    checks["gt_selection_is_v7"] = (seam_argv[2] == C.SERVICE_ENTRY
                                    and all(argv[2] == C.WORKER_ENTRY for argv in worker_argv.values()))
    details.update(seam_argv=seam_argv, worker_argv=worker_argv)

    # 8. the V6 registration's GT trampolines are refused by the v7 seam
    v6_reg = json.loads((fixture / "registration.json").read_bytes())
    try:
        seam_parameters(dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=v6_reg["gt_gc"]))
        refused = None
    except ValueError as exc:
        refused = str(exc)
    checks["v6_gt_trampolines_refused"] = refused == "registered GT service/worker trampolines differ from the v7 contract"
    details["v6_gt_trampolines"] = v6_reg["gt_gc"]["trampolines"]

    # 9. the actual V6 prospective is rejected by the v7 child validation (fresh process)
    proc = fresh(["-m", C.CHILD, C.VERIFY_FLAG, "--output", str(fixture)], out / "v7_child_on_v6.log")
    checks["v7_child_rejects_v6_registration"] = proc.returncode != 0 and "ValueError" in proc.stderr
    details["v7_child_on_v6_last_line"] = (proc.stdout + proc.stderr).rstrip().splitlines()[-1]

    # 10. the repair is byte-identical to the offline-qualified source
    checks["prep_repair_is_qualified_source"] = all(sha(ROOT / relative) == digest
                                                    for relative, digest in C.PREP_LIFETIME_QUALIFIED.items())

    passed = all(checks.values())
    result = dict(schema=REGRESSION_SCHEMA, status="PASS" if passed else "FAIL", checks=checks, details=details,
                  v6_root=str(V6_ROOT), v6_files=copied, v6_fixture_identical=identical,
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
