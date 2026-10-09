"""Binding regression for v8: the v8 roles resolve v8, nothing older is rebound (importing v8 installs neither the
FREEZE repair nor the tilt-aware AVO switch), the GT service/worker selection is the v8 one, the v8 GT trampolines are
the v7 service copy and the qualified candidate worker trampoline, and the child/edge production composition is
identical to v7 (and v6/v3).

Run from the deployment worktree (CPU only, no live service):

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v8.binding_regression \\
      --name binding_regression_01

V7 evidence is never modified: the rejection check runs on byte-identical copies of the V7 prospective's
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
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v8"
V7_ROOT = ROOT / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261008_m3_route_b_v7/"
                  "good_all_prospective_01")
V7_FILES = ("PROSPECTIVE.json", "registration.json", "PREREQUISITE.json", "ACCEPTANCE.json", "arm.json",
            "compose_partition.yaml", C.RETENTION_PLAN_FILE)
OLDER = (*(f"splitfusion_m3_route_b_v{n}" for n in (1, 2, 3, 4, 5, 6, 7)), "splitfusion_avo_tilt_v1",
         "splitfusion_numeric_gt_producer_v1", "splitfusion_gt_freeze_lifetime_v1")


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
    v7 = ROOT / "rl_agent/splitfusion_m3_route_b_v7/entries.py"
    v8 = ROOT / f"{C.PACKAGE}/entries.py"
    return dict(child=_with_items(v7, "_child") == _with_items(v8, "_child") == _with_items(v6, "_child")
                == _with_items(v3, "child"),
                edge=_with_items(v7, "edge") == _with_items(v8, "edge") == _with_items(v6, "edge")
                == _with_items(v3, "edge"),
                child_items=_with_items(v8, "_child"), edge_items=_with_items(v8, "edge"))


def _node(path, name):
    tree = ast.parse(Path(path).read_text())
    return next(n for n in tree.body if getattr(n, "name", None) == name)


def service_entry_parity():
    """v8 service_entry = v7 service_entry (AST of the substitution class and of ``main``; only the docstring and the
    contract module they read differ)."""
    v7, v8 = ROOT / "rl_agent/splitfusion_m3_route_b_v7/service_entry.py", ROOT / f"{C.PACKAGE}/service_entry.py"
    return dict(identical_class=ast.dump(_node(v7, "WorkerSubstitution")) == ast.dump(_node(v8, "WorkerSubstitution")),
                identical_main=ast.dump(_node(v7, "main")) == ast.dump(_node(v8, "main")))


CANDIDATE_IMPORT = "from . import geometry as G, quality as Q, switch"
V8_IMPORT = "from rl_agent.splitfusion_avo_tilt_v1 import geometry as G, quality as Q, switch"


def worker_entry_parity():
    """v8 worker_entry.main = the qualified candidate ``splitfusion_avo_tilt_v1.worker_entry.main`` except that it
    imports the candidate modules absolutely (it is not inside the candidate package)."""
    candidate = ROOT / f"{C.AVO_TILT_PACKAGE}/worker_entry.py"
    mine = ast.unparse(_node(ROOT / f"{C.PACKAGE}/worker_entry.py", "main"))
    theirs = ast.unparse(_node(candidate, "main"))
    return dict(candidate_trampoline_is_qualified=sha(candidate) == C_CANDIDATE_WORKER_SHA256,
                main_differs_only_in_import=mine.count(V8_IMPORT) == 1 and theirs.count(CANDIDATE_IMPORT) == 1
                and mine.replace(V8_IMPORT, CANDIDATE_IMPORT) == theirs)


# the candidate worker trampoline exercised by experiments/splitfusion_avo_tilt_v1/service_proof_02
C_CANDIDATE_WORKER_SHA256 = "37a7c69fc35d1ae0debed7bd5b38980e0d5e843c6666e0ba9b84af4e7d015bb5"


RESOLUTION_PROBE = """
import json
from rl_agent.splitfusion_m3_route_b_v8 import entries, host, retention, gc_seams, runner, service_entry, worker_entry
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
print(json.dumps(dict(
    v8=dict(entries_contract=entries.C.__name__, host_contract=host.C.__name__, retention_contract=retention.C.__name__,
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
    v7_untouched=dict(entries=v7_entries.C.__name__, gc_seams=v7_seams.C.__name__, retention=v7_ret.C.__name__,
                      host=v7_host.C.__name__, runner=v7_runner.C.__name__, service_entry=v7_service.C.__name__,
                      worker_entry=v7_worker.C.__name__),
    prep_core_unchanged_by_import=PC.PrepCore is R.ORIGINAL_PREP_CORE,
    numeric_avo_unchanged_by_import=NA.numeric_avo is SW.ORIGINAL_NUMERIC_AVO and not SW.installed()), sort_keys=True))
"""


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "v7_prospective_copy"
    fixture.mkdir()
    copied = {}
    for filename in V7_FILES:
        shutil.copy2(V7_ROOT / filename, fixture / filename)
        copied[filename] = sha(V7_ROOT / filename)
    identical = all(sha(fixture / filename) == digest for filename, digest in copied.items())
    checks, details = {}, {}

    # 1-2. v8 resolution in a fresh interpreter; importing v8 rebinds nothing in v3/v6/v7 and installs nothing
    proc = fresh(["-c", RESOLUTION_PROBE], out / "resolution_probe.log")
    resolution = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    v8c = C.MODULE + ".contract"
    checks["v8_functions_resolve_v8"] = bool(resolution) and all(
        value.startswith(C.MODULE) for value in resolution["v8"].values()) and resolution["v8"]["host_contract"] == v8c
    checks["older_globals_untouched"] = bool(resolution) and (
        set(resolution["v3_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v3.contract"}
        and set(resolution["v6_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v6.contract"}
        and set(resolution["v7_untouched"].values()) == {"rl_agent.splitfusion_m3_route_b_v7.contract"}
        and resolution["prep_core_unchanged_by_import"] is True
        and resolution["numeric_avo_unchanged_by_import"] is True)
    details["resolution"] = resolution

    # 3-5. static: no older M3/candidate global rebound; child/edge scopes identical to v7/v6/v3; constants identical
    hits = foreign_rebinding(ROOT / C.PACKAGE)
    checks["no_foreign_global_rebinding"] = hits == []
    parity = composition_parity()
    checks["child_and_edge_scopes_identical_to_v7"] = parity["child"] and parity["edge"]
    audit = C.role_constant_audit()
    checks["retained_role_constants_identical"] = all(audit.values())
    details.update(foreign_rebinding=hits, composition=parity, role_constant_audit=audit)

    # 6. the v8 GT trampolines: the v7 service copy; the qualified candidate worker trampoline
    service = service_entry_parity()
    checks["service_entry_identical_to_v7"] = all(service.values())
    worker = worker_entry_parity()
    checks["worker_entry_is_the_qualified_candidate_trampoline"] = all(worker.values())
    details.update(service_entry_parity=service, worker_entry_parity=worker)

    # 7. GT selection: the seam starts the v8 service; the v8 service starts the v8 worker entry for BOTH roles
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
    checks["gt_selection_is_v8"] = (seam_argv[2] == C.SERVICE_ENTRY
                                    and all(argv[2] == C.WORKER_ENTRY for argv in worker_argv.values()))
    details.update(seam_argv=seam_argv, worker_argv=worker_argv)

    # 8. the V7 registration's GT trampolines are refused by the v8 seam
    v7_reg = json.loads((fixture / "registration.json").read_bytes())
    try:
        seam_parameters(dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=v7_reg["gt_gc"]))
        refused = None
    except ValueError as exc:
        refused = str(exc)
    checks["v7_gt_trampolines_refused"] = refused == "registered GT service/worker trampolines differ from the v8 contract"
    details["v7_gt_trampolines"] = v7_reg["gt_gc"]["trampolines"]

    # 9. the actual V7 prospective is rejected by the v8 child validation (fresh process)
    proc = fresh(["-m", C.CHILD, C.VERIFY_FLAG, "--output", str(fixture)], out / "v8_child_on_v7.log")
    checks["v8_child_rejects_v7_registration"] = proc.returncode != 0 and "ValueError" in proc.stderr
    details["v8_child_on_v7_last_line"] = (proc.stdout + proc.stderr).rstrip().splitlines()[-1]

    # 10. the repair is byte-identical to the offline-qualified source
    checks["prep_repair_is_qualified_source"] = all(sha(ROOT / relative) == digest
                                                    for relative, digest in C.PREP_LIFETIME_QUALIFIED.items())

    # 11. the tilt-aware AVO runtime files are byte-identical to the offline-qualified sources and registered
    checks["avo_tilt_is_qualified_source"] = all(sha(ROOT / relative) == digest
                                                 for relative, digest in C.AVO_TILT_QUALIFIED.items())
    gc = C.gc_registration("candidate", C.FRAME_CEILING, "v2")
    checks["avo_geometry_registered_in_prep_only"] = (gc["avo_geometry"]["roles"] == ["prep"]
                                                      and gc["avo_geometry"]["sources"] == C.AVO_TILT_QUALIFIED
                                                      and gc["prep_freeze_lifetime"]["roles"] == ["prep"])

    passed = all(checks.values())
    result = dict(schema=REGRESSION_SCHEMA, status="PASS" if passed else "FAIL", checks=checks, details=details,
                  v7_root=str(V7_ROOT), v7_files=copied, v7_fixture_identical=identical,
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
