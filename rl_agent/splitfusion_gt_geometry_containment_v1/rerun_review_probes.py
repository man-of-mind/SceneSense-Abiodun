"""Re-execute the independent review's probe functions, UNCHANGED, against the corrected candidate.

``experiments/splitfusion_gt_geometry_containment_review_v1/review.py`` is imported by path (its ``main`` is not
called, so its ``REVIEW_RESULT.json`` is never touched); ``entry_probe`` and ``plan_probe`` run exactly as written,
with the review's own check definitions, and the result is written to this package's evidence directory.

    cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
      PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_gt_geometry_containment_v1.rerun_review_probes \\
      --name <dir>
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "experiments/splitfusion_gt_geometry_containment_review_v1/review.py"
EVIDENCE = ROOT / "experiments/splitfusion_gt_geometry_containment_v1"


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    review_result = REVIEW.with_name("REVIEW_RESULT.json")
    before = hashlib.sha256(review_result.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("independent_review_v1", REVIEW)
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    entries = [rv.entry_probe("current_V16", rv.VW, rv.VC), rv.entry_probe("candidate", rv.CW, rv.CC)]
    rv.P.environment_ok()
    rv.AVO.install()
    try:
        corpus = rv.FixtureCorpus({7: ("TILT", "146", *rv.S1_TILT, "independent_review_synthetic_tilt")})
        plans = [rv.plan_probe(corpus, refused=refused, mutation=mutation)
                 for mutation in ("missing_action", "missing_frame_context", "action_frame_mismatch")
                 for refused in (False, True)]
    finally:
        rv.AVO.uninstall()
    checks = dict(candidate_preserves_accounting_v3=entries[1]["accounting_v3_active"],
                  candidate_registers_accounting_v3=entries[1]["registration_has_storage_accounting"],
                  malformed_plans_remain_fatal=all(r["engine_failed"] and not r["normal_geometry_refusal"] for r in plans),
                  control_malformed_plans_fatal=all(r["engine_failed"] and not r["normal_geometry_refusal"]
                                                    for r in plans if not r["refused_frame"]),
                  entry_probes_restored=all(r["restored"] for r in entries))
    result = dict(status="PROBES_PASS" if all(checks.values()) else "HOLD", checks=checks, entry_probes=entries,
                  malformed_plan_probes=plans, review_script_sha256=hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
                  review_result_sha256_unchanged=before == hashlib.sha256(review_result.read_bytes()).hexdigest(),
                  candidate_sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(Path(__file__).parent.glob("*.py"))},
                  note="check definitions copied verbatim from review.py main(); probes imported unchanged")
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"], checks=checks,
                          review_result_unchanged=result["review_result_sha256_unchanged"]), indent=1))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
