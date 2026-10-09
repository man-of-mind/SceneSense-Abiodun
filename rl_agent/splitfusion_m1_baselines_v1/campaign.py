"""Run the two pre-sealed M1 live arms serially, once each, with no retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rl_agent.splitfusion_fixed_action_screen_analysis_v3.campaign import live_window

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
ROOTS = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_m1_baselines_v1"
EVIDENCE = MAIN / "experiments/splitfusion_m1_baselines_v1"
ARMS = ("agent", "static")
HOST = {"agent": "rl_agent.splitfusion_run5_full_live_retention_v1.host",
        "static": "rl_agent.splitfusion_m1_static_live_v1.host"}
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
GT_PATTERN = ("splitfusion_m1_static_live_v1|splitfusion_run5_full_live_retention_v1|"
              "splitfusion_exact_gt_live_v2.service|splitfusion_exact_gt_service_proto_v2.worker")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(log, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = json.dumps(value, sort_keys=True)
    log.write(line + "\n"); log.flush(); print(line, flush=True)


def mem_available():
    line = next(x for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:"))
    return int(line.split()[1]) * 1024


def leftovers():
    found = subprocess.run(["pgrep", "-af", GT_PATTERN], capture_output=True, text=True)
    return [line for line in found.stdout.splitlines()
            if "pgrep" not in line and "campaign" not in line]


def wait_window(*, timeout_s, max_load=None, min_wait_s=0):
    start = time.monotonic()
    while True:
        window = live_window()
        ok = window["exclusive"] and (max_load is None or window["loadavg_1min"] <= max_load)
        if ok and time.monotonic() - start >= min_wait_s:
            return window, True
        if time.monotonic() - start > timeout_s:
            return window, False
        time.sleep(30)


def freeze(out):
    from rl_agent.splitfusion_run5_full_live_retention_v1 import contract as A
    from rl_agent.splitfusion_m1_static_live_v1 import contract as S
    roots = {arm: ROOTS / arm for arm in ARMS}
    files = ("registration.json", "PROSPECTIVE.json", "PREREQUISITE.json", "ACCEPTANCE.json",
             "arm.json", "compose_partition.yaml", "RETENTION_PLAN.json")
    record = dict(
        schema="splitfusion.m1_baselines_v1.freeze.v1", worktree=str(WT),
        actor_boundary_sha256=(json.loads((roots["agent"] / "registration.json").read_bytes())
                               ["policy"]["expected_weights_sha256"]),
        runtime_sources={"agent": A.source_inventory(WT), "static": S.source_inventory(WT)},
        arms={arm: {name: sha(roots[arm] / name) for name in files} for arm in ARMS},
        acceptance_sources={name: sha(EVIDENCE / name) for name in
                            ("AGENT_ACCEPTANCE.json", "ACCEPTANCE.json", "STORAGE_PLAN.json")},
        campaign_sha256=sha(__file__),
        retention_plan_equal=(sha(roots["agent"] / "RETENTION_PLAN.json") ==
                              sha(roots["static"] / "RETENTION_PLAN.json")),
        zero_completion_semantics=json.loads((EVIDENCE / "ACCEPTANCE.json").read_bytes())["zero_completion"],
        prepared_before_either_launch=True, frozen_before_either_launch=True,
        route="FAVORABLE_STABLE", no_retry=True, separate_sessions_not_identical_scenes=True)
    (out / "FREEZE.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def retention_postflight(root, out, arm):
    from rl_agent.splitfusion_paired_retention_v2 import validate as V, gt_adapter as G
    reg = json.loads((root / "registration.json").read_bytes())
    plan_sha = reg["paired_retention"]["plan_sha256"]
    result = json.loads((root / "RETENTION_RESULT.json").read_bytes())
    archive = root / "paired_retention_v2"
    manifest = archive / "MANIFEST.json"
    digest = sha(manifest) if manifest.exists() else None
    doc = dict(retention_result=result, manifest_sha256=digest,
               manifest_matches_child=digest == result.get("manifest_sha256"), plan_sha256=plan_sha)
    if digest and doc["manifest_matches_child"]:
        verified = V.verify_archive_v2(archive, expected_manifest_sha256=digest,
            expected_plan_sha256=plan_sha, plan_path=root / "RETENTION_PLAN.json", strict=False)
        doc["verify_strict_false"] = dict(status_counts=verified["status_counts"],
            records=len(verified["records"]), not_retained=verified["not_retained"],
            observed=verified["observed_policy_decisions"])
        cell = next((root / "capture/cells").iterdir())
        try:
            refs = G.resolve_gt_references(archive, manifest_sha256=digest, plan_sha256=plan_sha,
                archive_root=cell / "numeric_gt/service/archive",
                output_path=out / f"GT_REFERENCES_{arm}.json")
            doc["gt_references"] = dict(sha256=refs["references_sha256"],
                status_counts=refs["status_counts"],
                ordinal_crosscheck_available=refs["ordinal_crosscheck_available"])
        except Exception as exc:
            doc["gt_references"] = dict(error=f"{type(exc).__name__}: {exc}")
    (out / f"RETENTION_POSTFLIGHT_{arm}.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True, default=str))
    return doc


def find_one(root, name):
    values = list(root.rglob(name))
    if len(values) != 1:
        raise ValueError(f"expected one {name}, found {len(values)}")
    return values[0]


def static_verdict(root, launch_rc):
    from rl_agent.splitfusion_run5_full_live_retention_v1_analysis.checks import check as gc_check
    evidence = json.loads(find_one(root / "capture", "PHASE6_UE_EVIDENCE.json").read_bytes())
    fixed = json.loads(find_one(root / "capture", "FIXED_ACTION_SUMMARY.json").read_bytes())
    counters, fixed_counters = evidence["counters"], fixed["counters"]
    identities = evidence["transmitted_identities"]
    selection = dict(
        decisions=counters["policy_decisions"], holds=counters["policy_holds"],
        fallbacks=counters["fallbacks"], actor_calls=counters["actor_calls"],
        transmitted=len(identities), fixed_counters=fixed_counters)
    selection_pass = (
        counters["policy_decisions"] > 0 and counters["policy_holds"] > 0
        and counters["fallbacks"] == 0 and counters["actor_calls"] == 0
        and fixed_counters["fallbacks"] == 0 and fixed_counters["actor_calls"] == 0
        and fixed_counters["executed_mismatch"] == 0
        and fixed_counters["decisions"] == counters["policy_decisions"]
        and fixed_counters["holds"] == counters["policy_holds"])
    gc = gc_check(root)
    protocol_faults = []
    if evidence.get("faulted"): protocol_faults.append(evidence["faulted"])
    for candidate in root.rglob("EDGE_EARLY_GT.json"):
        value = json.loads(candidate.read_bytes())
        if value.get("error"): protocol_faults.append(value["error"])
    edge_report = json.loads(find_one(root / "capture", "edge_report.json").read_bytes())
    completions = int(edge_report.get("publications", 0))
    delivery = (dict(status="NOT_EXERCISED_ZERO_COMPLETION", completed=0,
                     quality="UNAVAILABLE", latency="UNAVAILABLE") if completions == 0 else
                dict(status="OBSERVED_COMPLETIONS", completed=completions,
                     quality="ELIGIBLE_COMPLETE_PREDICTIONS_ONLY"))
    qualification = json.loads((root / "capture/QUALIFICATION_RESULT.json").read_bytes())
    # The retained profiler's G8/P7 gates require feedback and therefore fail
    # by construction for the preregistered zero-completion outcome.  P0/P5
    # encode actor-admission behavior and are not applicable to this external
    # fixed selector.  All identity, accounting, instrumentation,
    # infrastructure and restoration gates remain mandatory.
    core_gates = tuple(f"G{i}_{name}" for i, name in (
        (1, "FRAME_ACCOUNTING"), (2, "ACTOR_HASH_UNCHANGED"),
        (3, "ZERO_IDENTITY_CONFLICTS"), (4, "COVERAGE_REPORTED"),
        (5, "SAMPLER_REPORTED"), (6, "NO_PROFILER_INDUCED_FAULT"),
        (7, "INSTRUMENTATION_EVIDENCE_VALID"), (9, "HOST_RESTORED_CLEAN")))
    phase_gates = qualification.get("phase6_report_only", {}).get("gates", {})
    evidence_gates_pass = (all(qualification.get("gates", {}).get(key) is True for key in core_gates)
        and all(phase_gates.get(key) is True for key in
                ("P1_IDENTITY", "P3_HOLD_TICKET", "P4_ACCOUNTING",
                 "P6_RESTORE_COLD", "P8_NO_INFRASTRUCTURE_FAULT")))
    zero_completion_valid = completions == 0 and evidence_gates_pass
    passed = selection_pass and gc["passed"] and not protocol_faults and (
        launch_rc == 0 or zero_completion_valid)
    retention_result = json.loads((root / "RETENTION_RESULT.json").read_bytes())
    return dict(schema="splitfusion.m1_static_live_v1.verdict.v1",
        verdict=("VALID_STATIC_ZERO_COMPLETION_MEASUREMENT" if passed and completions == 0 else
                 "VALID_STATIC_MEASUREMENT" if passed else "STOP_INVALID_STATIC_MEASUREMENT"),
        passed=passed, launch_returncode=launch_rc, selection=selection,
        selection_pass=selection_pass, protocol_faults=protocol_faults,
        evidence_collection_status="COMPLETE" if passed else "FAILED",
        prediction_delivery=delivery, gc_trace=gc,
        evidence_gates_pass=evidence_gates_pass,
        generic_launcher_gate_status=dict(
            qualification_passed=qualification.get("passed"), gates=qualification.get("gates"),
            phase6_gates=phase_gates,
            interpretation=("G8/P7 feedback checks NOT_EXERCISED under preregistered zero-completion; "
                            "P0/P5 actor checks not applicable to the external fixed selector")
                if zero_completion_valid else "no exception applied"),
        edge_outcomes=dict(publications=completions, counters=edge_report.get("counters", {}),
                           failures=edge_report.get("failures", [])),
        decision_outcomes=dict(timeout=sum(row.get("terminal") == "TIMEOUT"
            for row in evidence.get("resolutions", [])), total=len(evidence.get("resolutions", []))),
        retention=dict(status_counts=retention_result.get("status_counts", {}),
                       observed_policy_decisions=retention_result.get("observed_policy_decisions"),
                       storage_bytes=retention_result.get("storage_bytes")),
        zero_completion_rule="delivery checks are not exercised; protocol faults still fail")


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=False, exist_ok=False)
    log = open(out / "CAMPAIGN.jsonl", "x", buffering=1)
    frozen = freeze(out)
    event(log, kind="joint_freeze", actor_boundary=frozen["actor_boundary_sha256"],
          retention_plan_equal=frozen["retention_plan_equal"])
    if not frozen["retention_plan_equal"]:
        event(log, kind="stop", reason="RETENTION_PLANS_DIFFER"); return 2
    plan = json.loads((EVIDENCE / "STORAGE_PLAN.json").read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in STRIP}
    check_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                     MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    results = {}
    for index, arm in enumerate(ARMS):
        root = ROOTS / arm
        reg = json.loads((root / "registration.json").read_bytes())
        if any((root / name_).exists() for name_ in ("capture", "gt_gc_trace", "paired_retention_v2",
                                                      "RETENTION_RESULT.json")):
            event(log, kind="stop", arm=arm, reason="ROOT_NOT_FRESH"); return 2
        required = (plan["required_before_first_launch_bytes"] if arm == "agent" else
                    plan["static_capture_and_retention_budget_bytes"] + plan["analysis_budget_bytes"] +
                    plan["reserve_bytes"])
        free, memory, left = shutil.disk_usage("/").free, mem_available(), leftovers()
        event(log, kind="preflight", arm=arm, free_bytes=free, required_bytes=required,
              mem_available=memory, leftovers=left)
        if free < required or memory < 4 * 2**30 or left:
            event(log, kind="stop", arm=arm, reason="STORAGE_MEMORY_OR_LEFTOVER_PRECONDITION"); return 2
        window, ok = wait_window(timeout_s=600 if index == 0 else 900,
                                 max_load=None if index == 0 else 2.0,
                                 min_wait_s=0 if index == 0 else 120)
        (out / f"window_{arm}.json").write_text(json.dumps(window, indent=1, sort_keys=True))
        event(log, kind="live_window", arm=arm, ok=ok, exclusive=window["exclusive"],
              loadavg=window["loadavg_1min"], gpu_mib=window["gpu_memory_used_mib"])
        if not ok:
            event(log, kind="stop", arm=arm, reason="LIVE_WINDOW_NOT_EXCLUSIVE"); return 2
        token = reg["arms"]["C2"]["execute_token"]
        event(log, kind="launch_start", arm=arm, budget=reg["transmitted_budget"], token=token)
        with open(out / f"launch_{arm}.log", "x") as stream:
            rc = subprocess.run([sys.executable, "-B", "-m", HOST[arm], "launch",
                                 "--output", str(root), "--execute", token], cwd=WT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT).returncode
        event(log, kind="launch_end", arm=arm, returncode=rc)
        try:
            post = retention_postflight(root, out, arm)
            event(log, kind="retention_postflight", arm=arm,
                  manifest_matches=post["manifest_matches_child"],
                  gt=post.get("gt_references"))
        except Exception as exc:
            event(log, kind="retention_postflight_error", arm=arm,
                  error=f"{type(exc).__name__}: {exc}")
            post = None
        if arm == "agent":
            from rl_agent.splitfusion_run5_full_live_analysis_v1.integration import evaluate
            from rl_agent.splitfusion_run5_full_live_retention_v1_analysis.checks import check
            verdict = evaluate(root, launch_returncode=rc, launch_log=out / f"launch_{arm}.log")
            verdict["gc_trace"] = check(root)
            passed = verdict["verdict"] == "PASS_ACTOR_DRIVEN_INTEGRATION" and verdict["gc_trace"]["passed"]
        else:
            verdict = static_verdict(root, rc); passed = verdict["passed"]
        (out / f"VERDICT_{arm}.json").write_text(json.dumps(verdict, indent=1, sort_keys=True, default=str))
        left, after = leftovers(), live_window()
        (out / f"window_after_{arm}.json").write_text(json.dumps(after, indent=1, sort_keys=True))
        event(log, kind="verdict", arm=arm, passed=passed, leftovers=left,
              post_exclusive=after["exclusive"])
        results[arm] = dict(launch_returncode=rc, passed=passed)
        if not passed or left or post is None:
            event(log, kind="stop", arm=arm,
                  reason="INVALID_OR_CLEANUP_FAILURE_EVIDENCE_KEPT_NO_RETRY")
            return 3
    event(log, kind="pair_complete", results=results)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(run(parser.parse_args().name))
