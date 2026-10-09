"""Regression proof for the V5 child failure, the latent same-class defects, and the v6 resolution.

Run from the deployment worktree (CPU only, no live service):

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v6.binding_regression \
      --name binding_regression_01

V5 evidence is never modified: the reproduction runs on byte-identical copies (digests recorded) inside the new
evidence directory.  Every runtime reproduction is a fresh interpreter started with ``-m`` exactly like production.
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
from .host import REGRESSION_SCHEMA, compose_text

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v6"
V5_ROOT = ROOT / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_m3_route_b_v5/"
                  "good_all_prospective_04")
V4_ROOT = ROOT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_m3_route_b_v4/good_all"
V5_ATTEMPT = Path("capture/cells/a71__favorable_stable")
V5_CHILD_MODULE = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v4"
V5_ERROR = "ValueError: foreign GC-qualification registration"


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


def copy_v5_fixture(destination):
    """Byte-identical copy of the V5 prospective's top-level files and the attempt layout the child resolved."""
    copied = {}
    for path in sorted(V5_ROOT.iterdir()):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
            copied[path.name] = sha(path)
    (destination / V5_ATTEMPT).mkdir(parents=True)
    shutil.copy2(V5_ROOT / "capture/PROFILER_CONFIG.json", destination / "capture/PROFILER_CONFIG.json")
    copied["capture/PROFILER_CONFIG.json"] = sha(V5_ROOT / "capture/PROFILER_CONFIG.json")
    identical = all(sha(destination / name) == digest for name, digest in copied.items())
    return copied, identical


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


def foreign_rebinding(package_dir):
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
                        if any(f"splitfusion_m3_route_b_v{n}" in origin for n in (1, 2, 3, 4, 5)):
                            hits.append(f"{path.name}:{node.lineno}:{item.value.id}.{item.attr}")
    return hits


def _with_items(source_path, function):
    tree = ast.parse(Path(source_path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    found = [w for w in ast.walk(node) if isinstance(w, ast.With)]
    return [[ast.unparse(item) for item in w.items] for w in found]


def composition_parity():
    v3 = ROOT / "rl_agent/splitfusion_m3_route_b_v3/entries.py"
    v6 = ROOT / f"{C.PACKAGE}/entries.py"
    return dict(child=_with_items(v3, "child") == _with_items(v6, "_child"),
                edge=_with_items(v3, "edge") == _with_items(v6, "edge"),
                child_items=_with_items(v6, "_child"), edge_items=_with_items(v6, "edge"))


RESOLUTION_PROBE = """
import json
from rl_agent.splitfusion_m3_route_b_v6 import entries, host, retention, gc_seams, runner
from rl_agent.splitfusion_m3_route_b_v3 import entries as v3_entries, gc_seams as v3_seams, retention as v3_ret, host as v3_host
print(json.dumps(dict(
    v6=dict(entries_contract=entries.C.__name__, host_contract=host.C.__name__, retention_contract=retention.C.__name__,
            seam_contract=gc_seams.C.__name__, runner_contract=runner.C.__name__,
            verify_prospective=host.verify_prospective.__module__, load_observer=retention.load_observer.__module__,
            seam_parameters=gc_seams.seam_parameters.__module__, registration=entries.registration.__module__,
            validate_parent=runner.validate_parent.__module__),
    v3_untouched=dict(entries=v3_entries.C.__name__, gc_seams=v3_seams.C.__name__, retention=v3_ret.C.__name__,
                      host=v3_host.C.__name__)), sort_keys=True))
"""

V3_RETENTION_PROBE = """
import json, sys
from pathlib import Path
from rl_agent.splitfusion_m3_route_b_v3.retention import load_observer
root = Path(sys.argv[1])
try:
    load_observer(root, json.loads((root / "registration.json").read_bytes()))
    print(json.dumps(dict(rejected=False)))
except ValueError as exc:
    print(json.dumps(dict(rejected=True, reason=str(exc), function=load_observer.__module__)))
"""


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "v5_prospective_copy"
    fixture.mkdir()
    copied, identical = copy_v5_fixture(fixture)
    checks, details = {}, {}

    # 1. the exact V5 child failure, through the V5 child trampoline, in a fresh interpreter
    proc = fresh(["-m", V5_CHILD_MODULE, "--attempt-dir", str(fixture / V5_ATTEMPT)], out / "v5_child.log")
    text = proc.stdout + proc.stderr
    checks["v5_child_failure_reproduced"] = (
        proc.returncode != 0 and text.rstrip().endswith(V5_ERROR)
        and "splitfusion_m3_route_b_v4/entries.py" in text and "splitfusion_m3_route_b_v3/entries.py" in text
        and "splitfusion_m3_route_b_v3/host.py" in text and "in verify_prospective" in text)
    details["v5_child"] = dict(returncode=proc.returncode, last_line=text.rstrip().splitlines()[-1] if text else None,
                               fixture_identical_to_v5=identical, copied=copied)

    # 2. the next latent child failure: v3 retention loader vs the registered v4 plan
    proc = fresh(["-c", V3_RETENTION_PROBE, str(fixture)], out / "v3_retention_probe.log")
    probe = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    checks["latent_v3_retention_mismatch_exposed"] = (
        probe.get("rejected") is True and probe.get("reason") == "retention plan is not the registered rule for this budget")
    details["v3_retention_probe"] = probe

    # 3. the edge's partition mount: the V5 root's compose file binds the V4 root
    compose = (fixture / "compose_partition.yaml").read_text()
    record = json.loads((fixture / "PROSPECTIVE.json").read_bytes())
    mounted = [line.strip()[2:].rsplit(":", 2)[0] for line in compose.splitlines() if ":/partition_output:" in line]
    checks["latent_v5_edge_mount_exposed"] = (mounted == [str(V4_ROOT)]
                                              and compose != compose_text(V5_ROOT, record["retained_code"]))
    details["v5_edge_partition_output_source"] = mounted

    # 4. registered GT trampolines: V5 named the v3 entries while the v4 seam started the v4 entries
    v5_reg = json.loads((fixture / "registration.json").read_bytes())
    from .gc_seams import seam_parameters
    try:
        seam_parameters(dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=v5_reg["gt_gc"]))
        refused = None
    except ValueError as exc:
        refused = str(exc)
    checks["latent_gt_trampoline_mislabel_refused"] = (
        v5_reg["gt_gc"]["trampolines"]["service"] == "rl_agent.splitfusion_m3_route_b_v3.service_entry"
        and refused == "registered GT service/worker trampolines differ from the v6 contract")
    details["v5_gt_trampolines"] = v5_reg["gt_gc"]["trampolines"]

    # 5. v6 resolution in a fresh interpreter; importing v6 rebinds nothing in v3
    proc = fresh(["-c", RESOLUTION_PROBE], out / "resolution_probe.log")
    resolution = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    v6c, v3c = C.MODULE + ".contract", "rl_agent.splitfusion_m3_route_b_v3.contract"
    checks["v6_functions_resolve_v6"] = bool(resolution) and all(
        value in (v6c, C.HOST, C.RETENTION, C.GC_SEAM, C.ENTRIES, C.RUNNER) and value.startswith(C.MODULE)
        for value in resolution["v6"].values())
    checks["v3_globals_untouched"] = bool(resolution) and set(resolution["v3_untouched"].values()) == {v3c}
    details["resolution"] = resolution

    # 6. static: no global of an older M3 module is rebound by v6; composition identical to v3
    hits = foreign_rebinding(ROOT / C.PACKAGE)
    checks["no_foreign_global_rebinding"] = hits == []
    parity = composition_parity()
    checks["child_and_edge_scopes_identical_to_v3"] = parity["child"] and parity["edge"]
    audit = C.role_constant_audit()
    checks["retained_role_constants_identical"] = all(audit.values())
    details.update(foreign_rebinding=hits, composition=parity, role_constant_audit=audit)

    # 7. the V5 prospective is rejected by the v6 child validation (no broad acceptance of older schemas)
    proc = fresh(["-m", C.CHILD, C.VERIFY_FLAG, "--output", str(fixture)], out / "v6_child_on_v5.log")
    checks["v6_child_rejects_v5_registration"] = proc.returncode != 0 and "ValueError" in proc.stderr
    details["v6_child_on_v5_last_line"] = (proc.stdout + proc.stderr).rstrip().splitlines()[-1]

    passed = all(checks.values())
    result = dict(schema=REGRESSION_SCHEMA, status="PASS" if passed else "FAIL", checks=checks, details=details,
                  v5_root=str(V5_ROOT), v5_error=V5_ERROR, launcher_binding=C.launcher_binding(),
                  sources=C.exact_sources(ROOT), interpreter=sys.executable, live_services_started=False,
                  carla_started=False, oai_started=False)
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
