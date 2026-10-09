"""Qualification of the outcome-aware assessment (offline; nothing live).

Positive reproduction on the four retained M1 captures (format ``m1``):
  agent 495/495 complete, 476 by 170 ms (deployed actor; actor gates required)
  a54 AE32/UINT8 q0 470/500, 140 by 170 ms (fixed; actor-only gates not applicable)
  mode 11 q6000 500/500, 489 by 170 ms (fixed)
  noAE/UINT8 q0 0/500 (fixed; ZERO_COMPLETION; frame 1338's failed compute span NOT counted; P7 NOT_EXERCISED)
All four must be VALID_MEASUREMENT with exactly these counts and launch rc explained (M1 records: agent 0, fixed 1).
Negative controls, each on a scratch copy of the ~15 MB required evidence of one capture with exactly one mutation,
must produce the stated class.  The copies are discarded after the run (scratch directory only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from . import assessment as A
from . import completion as CP

ROOT = Path(__file__).resolve().parents[2]
PROFILER = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
M1 = {
    "agent": dict(root=PROFILER / "20261007_m1_baselines_v1/agent", fixed=False, launch_rc=0,
                  expect=(495, 495, 476, None)),
    "a54": dict(root=PROFILER / "20261007_m1_baselines_v3/fixed_ae32_uint8_a54", fixed=True, launch_rc=1,
                expect=(500, 470, 140, None)),
    "m11_q6000": dict(root=PROFILER / "20261007_m1_baselines_v3/fixed_mode11_q6000", fixed=True, launch_rc=1,
                      expect=(500, 500, 489, None)),
    "noae": dict(root=PROFILER / "20261007_m1_baselines_v1/static_attempt_02", fixed=True, launch_rc=1,
                 expect=(500, 0, 0, "ZERO_COMPLETION")),
}
REQUIRED = ("registration.json", "gt_gc_trace", "capture/QUALIFICATION_RESULT.json", "capture/PROFILER_ANALYSIS.json")
CELL_REQUIRED = ("CELL_RESULT.json", "per_frame_metrics.csv", "run4_phase6/PHASE6_UE_EVIDENCE.json",
                 "run4_phase6/edge_report.json", "profiler/edge/edge.profile.jsonl",
                 "profiler/edge/numeric_runtime/EDGE_EARLY_GT.json", "numeric_gt/service/FRAMES.jsonl",
                 "m1_fixed_action")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_fixed(root):
    return json.loads((Path(root) / "registration.json").read_bytes()).get("external_fixed_action")


def run_one(name, spec, root=None):
    root = root or spec["root"]
    return A.assess(root, fmt="m1", expected_fixed=expected_fixed(spec["root"]) if spec["fixed"] else None,
                    launch_rc=spec["launch_rc"], budget=1000)


def copy_required(spec, scratch):
    src, dst = Path(spec["root"]), Path(scratch)
    cell = CP.find_cell(src)
    rel = cell.relative_to(src)
    for item in REQUIRED:
        s, d = src / item, dst / item
        d.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if s.is_dir() else shutil.copy2)(s, d)
    for item in CELL_REQUIRED:
        s, d = cell / item, dst / rel / item
        if not s.exists():
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if s.is_dir() else shutil.copy2)(s, d)
    return dst, dst / rel


def _edit_json(path, fn):
    value = json.loads(Path(path).read_bytes())
    fn(value)
    Path(path).write_text(json.dumps(value))


def _edit_jsonl(path, fn):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = fn(rows)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def negatives():
    """(name, base capture, mutation(root, cell), expected verdict)."""
    def span_rows(cell):
        return cell / "profiler/edge/edge.profile.jsonl"

    def other_mode_on_a_hold(root, cell):
        def fn(rows):
            for r in rows:
                if r.get("r") == "span" and r.get("stage") == "edge.compute" and r.get("ph") == "live" \
                        and r.get("reward_requested") is False:
                    r["q_e4"] = 100
                    break
            return rows
        _edit_jsonl(span_rows(cell), fn)

    def actor_called(root, cell):
        _edit_json(cell / "run4_phase6/PHASE6_UE_EVIDENCE.json", lambda v: v["counters"].__setitem__("actor_calls", 1))

    def failed_span_relabelled_ok(root, cell):            # noAE frame 1338: an ok span without materialization
        def fn(rows):
            for r in rows:
                if r.get("r") == "span" and r.get("stage") == "edge.compute" and r.get("frame_id") == 1338 \
                        and r.get("ph") == "live":
                    r["ok"] = True
            return rows
        _edit_jsonl(span_rows(cell), fn)

    def timeout_stamped_as_completion(root, cell):        # a54: materialization event on an incomplete decision
        def fn(rows):
            for r in rows:
                if r.get("decision") and r.get("outcome") == "UNSCORED_EDGE_TERMINAL_STALE_BEFORE_EDGE":
                    r["events"].append(dict(role="edge", name="prediction_materialized",
                                            raw_ns=int(r["action_open_raw_ns"]) + 200_000_000))
                    break
            return rows
        _edit_jsonl(cell / "numeric_gt/service/FRAMES.jsonl", fn)

    def host_not_clean(root, cell):
        _edit_json(root / "capture/QUALIFICATION_RESULT.json",
                   lambda v: v["gates"].__setitem__("G9_HOST_RESTORED_CLEAN", False))

    def foreign_route_stop(root, cell):
        _edit_json(cell / "CELL_RESULT.json",
                   lambda v: v["child"]["route_detail"].__setitem__("error", "RuntimeError: CARLA connection lost"))

    def p7_false_with_completions(root, cell):
        _edit_json(root / "capture/QUALIFICATION_RESULT.json",
                   lambda v: v["phase6_report_only"]["gates"].__setitem__("P7_FEEDBACK_OVER_DOWNLINK", False))

    def edge_queue_overflow(root, cell):
        _edit_json(cell / "profiler/edge/numeric_runtime/EDGE_EARLY_GT.json",
                   lambda v: v["state"]["evaluator_counters"].__setitem__("queue_overflow", 1))

    def map_publications_are_not_completions(root, cell):   # positive control on the zero-completion capture
        _edit_json(cell / "run4_phase6/edge_report.json", lambda v: v.__setitem__("publications", 25))

    def unexplained_agent_rc(root, cell):
        pass                                               # launch rc 1 with CAPTURE_VALID (set by the caller)

    def fallback_on_fixed_arm(root, cell):
        _edit_json(cell / "run4_phase6/PHASE6_UE_EVIDENCE.json", lambda v: v["counters"].__setitem__("fallbacks", 1))

    return [
        ("N1_other_mode_on_a_hold", "a54", other_mode_on_a_hold, "INVALID_IDENTITY", None),
        ("N2_actor_called_on_fixed_arm", "a54", actor_called, "INVALID_IDENTITY", None),
        ("N3_failed_span_relabelled_ok_not_counted", "noae", failed_span_relabelled_ok, "INFRASTRUCTURE_FAILURE", None),
        ("N4_timeout_stamped_as_completion_refused", "a54", timeout_stamped_as_completion, "INFRASTRUCTURE_FAILURE", None),
        ("N5_host_not_clean", "m11_q6000", host_not_clean, "INFRASTRUCTURE_FAILURE", None),
        ("N6_foreign_route_stop", "m11_q6000", foreign_route_stop, "INFRASTRUCTURE_FAILURE", None),
        ("N7_p7_false_with_completions", "a54", p7_false_with_completions, "INFRASTRUCTURE_FAILURE", None),
        ("N8_edge_queue_overflow", "agent", edge_queue_overflow, "INFRASTRUCTURE_FAILURE", None),
        ("N9_unexplained_agent_launch_rc", "agent", unexplained_agent_rc, "INFRASTRUCTURE_FAILURE", 1),
        ("N10_fallback_substituted_on_fixed_arm", "a54", fallback_on_fixed_arm, "INVALID_IDENTITY", None),
        ("C1_map_publications_never_count_as_completion", "noae", map_publications_are_not_completions,
         "VALID_MEASUREMENT", None),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    positives, checks = {}, {}
    for name, spec in M1.items():
        result = run_one(name, spec)
        pc = result["prediction_completion"]
        got = (pc["decisions"], pc["completed"], pc["completed_by_170"], result["subtype"])
        positives[name] = dict(verdict=result["verdict"], counts=got, expected=spec["expect"],
                               failed_compute_span_frames=pc["failed_compute_span_frames"],
                               no_completion_subcategories=pc["no_completion_subcategories"],
                               checks=result["checks"], identity=result["identity"])
        checks[f"P_{name}_valid_with_registered_counts"] = (result["verdict"] == "VALID_MEASUREMENT"
                                                           and got == spec["expect"])
    checks["P_noae_frame_1338_failed_span_not_counted"] = positives["noae"]["failed_compute_span_frames"] == [1338]
    checks["P_noae_p7_not_exercised"] = positives["noae"]["checks"]["P7_FEEDBACK_OVER_DOWNLINK"] == A.NX
    checks["P_fixed_actor_gates_not_applicable"] = all(
        positives[k]["checks"][g] == A.NA for k in ("a54", "m11_q6000", "noae") for g in A.ACTOR_ONLY)
    checks["P_agent_actor_gates_required_and_pass"] = all(positives["agent"]["checks"][g] == A.PASS
                                                          for g in A.ACTOR_ONLY)
    neg = {}
    args.scratch.mkdir(parents=True, exist_ok=True)
    for name, base, mutate, expected, rc in negatives():
        work = Path(tempfile.mkdtemp(prefix=name + "_", dir=args.scratch))
        try:
            root, cell = copy_required(M1[base], work / "arm")
            mutate(root, cell)
            spec = dict(M1[base], launch_rc=M1[base]["launch_rc"] if rc is None else rc)
            try:
                result = A.assess(root, fmt="m1", expected_fixed=expected_fixed(M1[base]["root"]) if spec["fixed"]
                                  else None, launch_rc=spec["launch_rc"], budget=1000)
                verdict = result["verdict"]
                failing = sorted([k for k, v in result["checks"].items() if v == A.FAIL]
                                 + [k for k, v in result["identity"].items() if v == A.FAIL])
            except Exception as exc:                       # a crash is NOT an acceptable classification
                verdict, failing = f"CRASH: {type(exc).__name__}: {exc}"[:200], []
        finally:
            shutil.rmtree(work, ignore_errors=True)
        neg[name] = dict(base=base, expected=expected, verdict=verdict, failing=failing)
        checks[name] = verdict == expected
    passed = all(checks.values())
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    document = dict(schema="splitfusion.static_modes_v13_offline_v1.assessment_qualification.v1",
                    status="PASS" if passed else "FAIL", checks=checks, positives=positives, negatives=neg,
                    inputs={name: str(spec["root"]) for name, spec in M1.items()}, sources=sources,
                    live_services_started=False)
    (args.out / "RESULT.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    (args.out / "OUTPUT_SHA256.json").write_text(json.dumps({"RESULT.json": sha(args.out / "RESULT.json")},
                                                            indent=1) + "\n")
    print(json.dumps(dict(status=document["status"], failed=[k for k, v in checks.items() if not v],
                          positives={k: (v["verdict"], v["counts"]) for k, v in positives.items()},
                          negatives={k: v["verdict"] for k, v in neg.items()}), indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
