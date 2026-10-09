"""Binding regression of the v13 sources against the V12 sources they derive from (offline; nothing live).

Every V12 -> v13 difference must be exactly a registered one:
  * the UE-child scope list equals V12's minus {BAD_ALL controller installer, R1 route stop rule, retention observer}
    plus {fixed-engine scope, retained-controller requirement}, in the same order;
  * the edge entry, and the copied modules (gc_seams, service_entry, worker_entry, accounting_timing, partition,
    radio_telemetry, postflight_p3, route_drain), are AST-identical to V12 once the package names are mapped;
  * importing every v13 module rebinds no global of an older M3 or retained module (fresh interpreter);
  * the runner leaves the retained G1 unwrapped (budget branch) and installs only the P3 final-cycle exception;
  * retention is registered disabled and no retention scope exists; the controller is the retained one;
  * the frozen fixed identities re-resolve exactly; the FAVORABLE_STABLE trace regenerates to the registered digest;
  * limits differ from V12 only in max_frames; the GT registration only in names and budget-derived trace bounds;
  * qualified prep-repair / tilt-aware AVO / storage-accounting sources are unchanged; role constants identical;
  * the v13 entry rejects the actual V12 registration; the fixed engine refuses a mismatching identity.
Run from the deployment worktree (runner_v1 lives there):
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m rl_agent.splitfusion_static_modes_v13.binding_regression --name <dir>
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

from . import contract as C
from .host import REGRESSION_SCHEMA, V12_ROOT_RELATIVE

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_static_modes_v13"
V12_PACKAGE = ROOT / "rl_agent/splitfusion_m3_route_b_v12"
V13_PACKAGE = ROOT / C.PACKAGE
COPIED = ("gc_seams.py", "service_entry.py", "worker_entry.py", "accounting_timing.py", "partition.py",
          "radio_telemetry.py", "postflight_p3.py", "route_drain.py")
NAME_MAP = (("splitfusion_m3_route_b_v12", "splitfusion_static_modes_v13"), ("m3_route_b_v12", "static_modes_v13"),
            ("M3_V12_", "STATIC_V13_"))
REMOVED_SCOPES = ["install_bad_all_controller(attempt)", "RC.install_child_stop_rule()", "install_retention(observer)"]
ADDED_SCOPES = ["fixed", "require_retained_controller(attempt)"]
PROBE = r"""
import json, sys, types, importlib
watched = ["rl_agent.splitfusion_m3_route_b_v3.contract", "rl_agent.splitfusion_m3_route_b_v3.route_completion",
           "rl_agent.splitfusion_m3_route_b_v4.contract", "rl_agent.splitfusion_m3_route_b_v8.contract",
           "rl_agent.splitfusion_m3_route_b_v11.contract", "rl_agent.splitfusion_m3_route_b_v12.contract",
           "rl_agent.splitfusion_m3_route_b_v12.entries", "rl_agent.splitfusion_m3_route_b_v12.host",
           "rl_agent.splitfusion_run5_full_live_v1.runtime_policy",
           "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_decision_engine_v2",
           "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_live_child_v2",
           "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_live_runner_v2",
           "rl_agent.splitfusion_single_gpu_profiler_v1.runner_v1", "rl_agent.splitfusion_run5_live_avo_v1.live_gt"]
mods = {name: importlib.import_module(name) for name in watched}
def snap():
    return {name: {k: id(v) for k, v in vars(m).items() if not k.startswith("__")} for name, m in mods.items()}
before = snap()
for name in ["contract", "entries", "host", "runner", "gc_seams", "service_entry", "worker_entry",
             "accounting_timing", "partition", "radio_telemetry", "postflight_p3", "route_drain", "fixed_engine",
             "identities"]:
    importlib.import_module("rl_agent.splitfusion_static_modes_v13." + name)
after = snap()
changed = {name: sorted(k for k in set(before[name]) | set(after[name]) if before[name].get(k) != after[name].get(k))
           for name in watched}
from rl_agent.splitfusion_static_modes_v13 import runner as RN
from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_v1 as RV
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_child_v2 as CH
from rl_agent.splitfusion_run5_full_live_v1 import runtime_policy as RP
g1_before = RV.g1_frame_accounting
RN.install_gates()
binding = __import__("rl_agent.splitfusion_static_modes_v13.contract", fromlist=["x"]).launcher_binding()
def modules(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from modules(item)
    elif isinstance(value, str) and value.startswith("rl_agent."):
        yield value
missing = []
for name in sorted(set(modules(binding))):
    target = name.split(":")[0]
    try:
        importlib.import_module(target)
    except ModuleNotFoundError:
        parent, _, attr = target.rpartition(".")
        try:
            if not hasattr(importlib.import_module(parent), attr):
                missing.append(name)
        except Exception:
            missing.append(name)
print("PROBE=" + json.dumps(dict(
    changed_by_import=changed, g1_unwrapped=RV.g1_frame_accounting is g1_before
    and not getattr(RV.g1_frame_accounting, "__m3_route_branch__", False),
    phase6_final_cycle=getattr(R.evaluate_phase6, "__m3_final_cycle__", False) is True,
    route_stop_rule_not_installed=not hasattr(CH, "_M3_ROUTE_ACCEPT"),
    retained_controller=RP.CONTROLLER_MODULE, missing_binding_modules=missing)))
"""


def _mapped(text):
    for old, new in NAME_MAP:
        text = text.replace(old, new)
    return text


def _ast(text):
    return ast.dump(ast.parse(text), include_attributes=False)


def _function(tree_text, name):
    for node in ast.parse(tree_text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(name)


def _with_items(function):
    for node in ast.walk(function):
        if isinstance(node, ast.With):
            return [ast.unparse(item.context_expr) for item in node.items]
    return []


def _code_names(text):
    """Names and attributes used in code (docstrings and comments excluded)."""
    names = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
    return names


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Plan:
    def __init__(self, **fields):
        self.profile = types.SimpleNamespace(**fields)


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    checks, details = {}, {}
    v12_entries = (V12_PACKAGE / "entries.py").read_text()
    v13_entries = (V13_PACKAGE / "entries.py").read_text()
    v12_scopes = _with_items(_function(v12_entries, "_child"))
    v13_scopes = _with_items(_function(v13_entries, "_child"))
    expected = [s for s in v12_scopes if s not in REMOVED_SCOPES]
    details["child_scopes"] = dict(v12=v12_scopes, v13=v13_scopes)
    checks["child_scopes_equal_v12_except_registered"] = (
        all(s in v12_scopes for s in REMOVED_SCOPES)
        and [s for s in v13_scopes if s not in ADDED_SCOPES] == expected and v13_scopes[:2] == ADDED_SCOPES)
    checks["edge_entry_identical_to_v12"] = (
        ast.dump(_function(_mapped(v12_entries), "edge")) == ast.dump(_function(v13_entries, "edge")))
    identical = {f: _ast(_mapped((V12_PACKAGE / f).read_text())) == _ast((V13_PACKAGE / f).read_text())
                 for f in COPIED}
    details["copied_modules_identical"] = identical
    checks["copied_modules_ast_identical_to_v12"] = all(identical.values())
    probe = subprocess.run([sys.executable, "-B", "-c", PROBE], cwd=ROOT, capture_output=True, text=True,
                           env={k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"})
    lines = [l for l in probe.stdout.splitlines() if l.startswith("PROBE=")]
    result = json.loads(lines[-1][6:]) if lines else {}
    details["probe"] = dict(result, returncode=probe.returncode, stderr_tail=probe.stderr[-1500:])
    checks["import_rebinds_no_older_global"] = bool(result) and not any(result["changed_by_import"].values())
    checks["runner_keeps_retained_g1_budget_branch"] = result.get("g1_unwrapped") is True
    checks["runner_installs_only_p3_final_cycle_exception"] = result.get("phase6_final_cycle") is True
    code_names = _code_names(v13_entries)
    checks["child_route_stop_rule_not_installed"] = (result.get("route_stop_rule_not_installed") is True
                                                     and "install_child_stop_rule" not in code_names)
    checks["launcher_binding_names_existing_modules"] = bool(result) and result.get("missing_binding_modules") == []
    checks["controller_is_retained_unsubstituted"] = (result.get("retained_controller") == C.RETAINED_CONTROLLER_MODULE
                                                      and C.CONTROLLER_MODULE == C.RETAINED_CONTROLLER_MODULE
                                                      and not (V13_PACKAGE / "bad_all_controller.py").exists())
    try:
        C.retention_registration("0" * 64)
        refuses_plan = False
    except ValueError:
        refuses_plan = True
    checks["retention_registered_disabled"] = (C.retention_registration()["enabled"] is False and refuses_plan
                                               and not (V13_PACKAGE / "retention.py").exists()
                                               and "install_retention" not in code_names
                                               and "finalize_after_child" not in code_names)
    resolved = {arm: C.resolved_fixed_action(arm) == C.fixed_action(arm) for arm in C.MODE_ARMS}
    details["fixed_identities"] = resolved
    checks["fixed_identities_reresolve_exactly"] = all(resolved.values()) and C.fixed_action(C.AGENT_ARM) is None
    prefix = C.favorable_prefix_check(ROOT)
    details["favorable_prefix"] = prefix
    checks["favorable_trace_regenerates_registered_digest"] = prefix["ok"] is True
    v12_reg = json.loads((ROOT / V12_ROOT_RELATIVE / "registration.json").read_bytes())
    mine = C.limits(C.ARM_BUDGET)
    checks["limits_differ_from_v12_only_in_max_frames"] = (
        {k: v for k, v in mine.items() if k != "max_frames"}
        == {k: v for k, v in v12_reg["exact_gt_limits"].items() if k != "max_frames"} and mine["max_frames"] == 1000)
    gc_v13 = C.gc_registration("candidate", C.ARM_BUDGET, "v2")
    text = json.dumps(gc_v13)
    back = json.loads(text.replace("splitfusion_static_modes_v13", "splitfusion_m3_route_b_v12")
                      .replace("static_modes_v13", "m3_route_b_v12"))
    checks["gc_registration_differs_from_v12_only_in_names_and_trace_limits"] = (
        "m3_route_b_v12" not in text and dict(back, trace_limits=None) == dict(v12_reg["gt_gc"], trace_limits=None)
        and gc_v13["trace_limits"] == C.trace_limits(1000))
    audit = C.role_constant_audit()
    budget_key = "rl_agent.splitfusion_m3_route_b_v3.worker_entry[C=v4].BUDGETS"
    worker_text = (ROOT / "rl_agent/splitfusion_m3_route_b_v3/worker_entry.py").read_text()
    details["role_constant_audit"] = audit
    checks["role_constants_identical_except_registered_budget"] = (
        all(v for k, v in audit.items() if k != budget_key) and audit.get(budget_key) is False
        and C.V4.BUDGETS == (3900,) and C.BUDGETS == (1000,)
        and "if a.max_frames not in C.BUDGETS and a.max_frames < 1:" in worker_text)   # the guard admits 1,000
    qualified = {**C.PREP_LIFETIME_QUALIFIED, **C.AVO_TILT_QUALIFIED, **C.ACCOUNTING_QUALIFIED,
                 **C.ACCOUNTING_V1_QUALIFIED, **C.ACCOUNTING_V2_QUALIFIED}
    checks["qualified_repair_avo_accounting_sources_unchanged"] = all(sha(ROOT / k) == v for k, v in qualified.items())
    from .entries import registration
    try:
        registration(ROOT / V12_ROOT_RELATIVE / "registration.json", ROOT)
        rejects = False
    except ValueError as exc:
        rejects = "static-mode v13 entry requires its exact sealed registration" in str(exc)
    checks["v13_entry_rejects_v12_registration"] = rejects
    from . import fixed_engine as FE
    fixed = C.fixed_action("mode08")
    good = _Plan(**{k: fixed[k] for k in ("mode_id", "q_e4", "action_id", "profile_id", "execution_bundle_sha256",
                                          "keep_count", "drop_count", "family", "quantizer")})
    bad = _Plan(**dict(vars(good.profile), q_e4=100))
    log = types.SimpleNamespace(counters=dict(executed_mismatch=0))
    FE._check(good, log, fixed)
    try:
        FE._check(bad, log, fixed)
        refused = False
    except FE.FixedBaselineError:
        refused = True
    checks["fixed_engine_refuses_mismatching_identity"] = refused and log.counters["executed_mismatch"] == 1
    status = "PASS" if all(checks.values()) else "FAIL"
    document = dict(schema=REGRESSION_SCHEMA, status=status, checks=checks, details=details,
                    sources=C.exact_sources(ROOT), launcher_binding=C.launcher_binding(), live_services_started=False,
                    carla_server_started=False, oai_started=False)
    (out / "RESULT.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({"RESULT.json": sha(out / "RESULT.json")}, indent=1) + "\n")
    print(json.dumps(dict(status=status, failed=[k for k, v in checks.items() if not v]), indent=1))
    if status != "PASS":
        raise SystemExit(1)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
