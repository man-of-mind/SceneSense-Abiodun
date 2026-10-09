"""Prepared serial benchmark command; never executed by correctness tests.

Five explicitly different scopes: original evaluate_prepared (restoration,
validation, numerical scoring and evaluation evidence); prediction_admission
(complete records, identities, native labels and digest); admitted_input_join
(identity reconciliation and in-memory numerical row conversion); admitted_score
(only numerical scoring); deferred_evidence (JSON reconstruction/hashes, no PNG).
Target fixture import is outside samples. These are NOT interchangeable scopes
or a whole-pipeline speedup comparison.
"""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from . import scorer as S
from .fixtures import load_v11, oracle_arguments
from .shared_checks import shared_fixture
from .shared_admission import admit_prediction, join_admitted_inputs


def run(output, repetitions):
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    if np.__version__ != "1.26.4" or not all(os.environ.get(k) == "1" for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("NumPy 1.26.4 and one numerical thread required")
    if not 1 <= repetitions <= 5:
        raise ValueError("bounded 1..5 complete repetitions required")
    if not output.resolve().is_relative_to(S.ROOT / "experiments/splitfusion_minimal_quality_scorer_v1"):
        raise ValueError("owned fresh output directory required")
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    scorer = S.NumericalScorer()
    cases, reader, _ = load_v11()
    shared = [shared_fixture(scorer, case) for case in cases]
    admitted = [join_admitted_inputs(scorer, target=t, prediction=p) for t, p, _ in shared]
    results = [scorer.score(frame) for frame in admitted]
    for case, result, (_, _, attachment) in zip(cases, results, shared):
        reference = E.evaluate_prepared(**oracle_arguments(case))
        if S.reconstruct_evidence(result, legacy_reference=attachment) != reference.evidence or result.measurement != reference.measurement:
            raise AssertionError("exact premeasurement check failed")
    samples = []
    for repetition in range(repetitions):
        for case, frame, result, (target, prediction, attachment) in zip(cases, admitted, results, shared):
            calls = [("original_evaluate_prepared", lambda: E.evaluate_prepared(**oracle_arguments(case))),
                     ("admitted_score", lambda: scorer.score(frame)),
                     ("prediction_admission", lambda: admit_prediction(scorer, identity=case["prediction_identity"],
                         labels=case["predicted_labels"], records=case["predictions"], prediction_contract=case["prediction_contract"])),
                     ("admitted_input_join", lambda: join_admitted_inputs(scorer, target=target, prediction=prediction)),
                     ("deferred_evidence", lambda: S.reconstruct_evidence(result, legacy_reference=attachment))]
            if repetition % 2:
                calls.reverse()
            for name, call in calls:
                start_cpu = time.thread_time_ns()
                start_raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                value = call()
                end_raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                end_cpu = time.thread_time_ns()
                samples.append(dict(frame_id=frame.frame_id, repetition=repetition, stage=name,
                    start_raw_ns=start_raw, wall_ns=end_raw-start_raw, thread_cpu_ns=end_cpu-start_cpu))
                # Results checked outside samples. Source hashes and admission
                # are intentionally not moved inside the minimal scorer.
                if name == "admitted_score" and value.measurement != result.measurement:
                    raise AssertionError("benchmark result changed")
    document = dict(scope=__doc__, live_qualified=False, source_seal=reader.seal,
        repetitions=repetitions, samples=samples, numpy=np.__version__, opencv=cv2.__version__,
        affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid(),
        limits="Requires externally confirmed serial execution. No causal downlink, deadline or whole-pipeline claim.")
    (output / "BENCHMARK.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--coordinator-confirmed-serial", action="store_true", required=True)
    args = parser.parse_args()
    run(args.output, args.repetitions)
