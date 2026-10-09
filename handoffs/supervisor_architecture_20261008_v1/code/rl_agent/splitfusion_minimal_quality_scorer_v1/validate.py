"""CPU correctness only; no performance timing or runtime launches."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from . import scorer as S
from .fixtures import admission_arguments, oracle_arguments, load_v11, load_older_sample, V12_RESULT
from .offline_admission import admit_oracle_fixture, VALIDATOR_PINS


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def forbidden(*args, **kwargs):
    raise AssertionError("artifact/restoration/read/hash/matching crossed a forbidden stage")


def guarded_score(scorer, frame):
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    counts = dict(detection=0, person=0)
    def wrapper(function, name):
        def call(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)
        return call
    with ExitStack() as stack:
        # Process-local only; other sessions have separate interpreter globals.
        for target in ("builtins.open", "io.open", "pathlib.Path.read_bytes", "pathlib.Path.read_text",
                       "hashlib.sha256", "json.dumps", "json.loads", "cv2.imread", "cv2.imdecode", "cv2.imencode"):
            stack.enter_context(patch(target, forbidden))
        for obj, name in ((E, "restore_prepared_target"), (E, "verify_sources"),
                          (S, "reconstruct_evidence"), (S, "archive_payloads")):
            stack.enter_context(patch.object(obj, name, forbidden))
        stack.enter_context(patch.dict(scorer._detection,
            match_frame=wrapper(scorer._detection["match_frame"], "detection"),
            load_gt=forbidden, load_predictions=forbidden))
        stack.enter_context(patch.dict(scorer._person,
            greedy_match=wrapper(scorer._person["greedy_match"], "person")))
        result = scorer.score(frame)
    require(counts == dict(detection=1, person=3), "matching repeated or omitted")
    return result, counts


def guarded_evidence(scorer, result):
    with patch.dict(scorer._detection, match_frame=forbidden, load_gt=forbidden, load_predictions=forbidden), \
         patch.dict(scorer._person, greedy_match=forbidden):
        return S.reconstruct_evidence(result), S.archive_payloads(result)


def quality_checks(case, candidate, reference, spec, action_contract):
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import ExecutedActionIdentity
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as C
    cq, rq = Q.evaluate_exact_quality(spec, candidate), Q.evaluate_exact_quality(spec, reference)
    require(cq == rq, "complete derived exact quality differs")
    q_perc = None if cq is None else float(cq.q_perc)
    require(q_perc == case["feedback"]["q_perc"], "retained UE quality differs")
    action = case["prediction_identity"]["action_identity"]
    envelope = SimpleNamespace(**action, capture_timestamp_ns=case["prediction_identity"]["capture_timestamp_ns"])
    cf, cr = W.quality_feedback(spec, candidate, envelope)
    rf, rr = W.quality_feedback(spec, reference, envelope)
    require(cf == rf and cr == rr, "feedback kind/reason differs")
    executable = action_contract.resolve(action["mode_id"], action["q_e4"] / 10000.)
    executed = ExecutedActionIdentity.from_executable_action(executable, action_contract)
    identity = C.DecisionIdentityV1(action["session_uuid"], "offline-parity-ue", action["decision_seq"])
    resolutions = []
    for elapsed in (169_999_999, 170_000_000, 170_000_001):
        values = []
        for quality in (cq, rq):
            kind = C.RewardEventKind.EVALUATOR_FAULT if quality is None else C.RewardEventKind.DELIVERED_SUCCESS
            event = C.RewardEventV1(identity, executed, kind, 1, 1 + elapsed,
                "OFFLINE_BOUNDARY_TEST", "exact-scorer-correctness",
                None if quality is None else float(quality.q_perc))
            values.append(C.resolve_reward(event).to_canonical_dict())
        require(values[0] == values[1], "reward arithmetic differs")
        resolutions.append(dict(elapsed_ns=elapsed, terminal=values[0]["terminal"], reward=values[0]["reward"]))
    return dict(q_perc=q_perc, undefined=cq is None, feedback_kind=cf.kind, reason=cr.value,
                boundary_rewards=resolutions)


def run(output):
    require(np.__version__ == "1.26.4", "reference NumPy 1.26.4 required")
    require(all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "bounded CPU numerical threads required")
    cv2.setNumThreads(1)
    output = output.resolve()
    owned = S.ROOT / "experiments/splitfusion_minimal_quality_scorer_v1"
    require(output.is_relative_to(owned), "output must stay in assigned evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as A
    from .shared_checks import check_shared, check_shared_adversarial
    shared_paths = [S.ROOT / "rl_agent/splitfusion_numeric_gt_integration_v1" / name
        for name in ("contract.py", "evidence.py", "oracle_bridge.py", "sources.py", "INTERFACE.json")]
    shared_pins = {str(path.relative_to(S.ROOT)): sha(path.read_bytes()) for path in shared_paths}
    scorer = S.NumericalScorer()
    cases, reader, _ = load_v11()
    print("loaded 49 sealed fixtures; CPU-only exact comparisons", flush=True)
    spec, action_contract = W.load_run4_quality_spec(S.ROOT), A.load_contract()
    fd_count_before = len(os.listdir("/proc/self/fd"))
    rows = []
    for i, case in enumerate(cases):
        admitted = admit_oracle_fixture(scorer, **admission_arguments(case))
        result, counts = guarded_score(scorer, admitted)
        oracle = E.evaluate_prepared(**oracle_arguments(case))
        evidence, payloads = guarded_evidence(scorer, result)
        require(result.measurement == oracle.measurement, "measurement differs")
        require(evidence == oracle.evidence == case["saved_evidence"], "reconstructed evidence differs")
        require(S.canonical(evidence) == S.canonical(oracle.evidence) == S.canonical(case["saved_evidence"]),
                "canonical evidence bytes differ")
        require(payloads == E.archive_payloads(oracle), "deferred PNG/archive bytes differ")
        quality = quality_checks(case, result.measurement, oracle.measurement, spec, action_contract)
        shared = check_shared(scorer, case, oracle)
        rows.append(dict(frame_id=admitted.frame_id, action_identity=case["prediction_identity"]["action_identity"],
            capture_identity=case["prediction_identity"]["gt_identity"],
            measurement=result.measurement, associations=asdict(result.associations), **quality,
            original_target_content_sha256=admitted.original_content_sha256,
            evidence_sha256=sha(S.canonical(evidence)), saved_evidence_sha256=case["saved_evidence_sha256"],
            measurement_sha256=sha(S.canonical(result.measurement)),
            deferred_archive_sha256={name: sha(value) for name, value in payloads.items()},
            matching_call_counts=counts, exact_measurement=True, exact_evidence=True, exact_archive=True,
            shared_admission=shared))
        if (i + 1) % 10 == 0:
            print("exact frames", i + 1, flush=True)
    require(sum(row["undefined"] for row in rows) == 8, "undefined population differs")
    from .adversarial import check_adversarial
    adversarial = check_adversarial(scorer, cases[0])
    shared_adversarial = check_shared_adversarial(scorer, cases[0])
    older_cases, older_reader, _ = load_older_sample()
    older_rows = []
    for case in older_cases:
        frame = admit_oracle_fixture(scorer, **admission_arguments(case))
        value, _ = guarded_score(scorer, frame)
        oracle = E.evaluate_prepared(**oracle_arguments(case))
        evidence, payloads = guarded_evidence(scorer, value)
        require(value.measurement == oracle.measurement and evidence == oracle.evidence == case["saved_evidence"]
                and payloads == E.archive_payloads(oracle), "older retained parity failed")
        quality = quality_checks(case, value.measurement, oracle.measurement, spec, action_contract)
        older_rows.append(dict(frame_id=frame.frame_id, exact=True, **quality,
                              shared_admission=check_shared(scorer, case, oracle)))
    require(len(os.listdir("/proc/self/fd")) == fd_count_before, "open descriptor count changed")
    scorer._verified_sources.verify_unchanged()
    for path, digest in shared_pins.items():
        require(sha((S.ROOT / path).read_bytes()) == digest, "shared interface changed during correctness checks")
    source_inventory = {str(path.relative_to(S.ROOT)): sha(path.read_bytes())
        for path in sorted((S.ROOT / "rl_agent/splitfusion_minimal_quality_scorer_v1").glob("*.py"))}
    result = dict(status="ALL49_EXACT_OFFLINE_COMPONENT_NOT_LIVE_QUALIFIED", unique_frames=49,
        undefined_quality_frames=8, defined_quality_frames=41, frame_results=rows,
        adversarial=adversarial, shared_adversarial=shared_adversarial,
        source_inventory=source_inventory, validator_pins=VALIDATOR_PINS,
        shared_interface_source_pins=shared_pins, older_frame_results=older_rows,
        older_reference_seal_sha256=older_reader.seal, older_consumed_files=older_reader.consumed,
        no_descriptor_growth=True, open_descriptors_before_after=fd_count_before,
        original_source_pins=json.loads(scorer._source_pins),
        reference_seal_sha256=reader.seal, consumed_reference_files=reader.consumed,
        v12_reference_result_sha256=sha(V12_RESULT.read_bytes()),
        numpy_version=np.__version__, opencv_version=cv2.__version__, performance_measured=False,
        scope="Offline component only; coordinator admission, runtime pose reconciliation, history and archive lifecycle pending")
    (output / "RESULT.json").write_bytes(json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    print(result["status"], "adversarial checks", len(adversarial), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output)
    except Exception:
        import traceback
        owned = S.ROOT / "experiments/splitfusion_minimal_quality_scorer_v1"
        if args.output.resolve().is_relative_to(owned) and args.output.is_dir():
            with (args.output / "FAILURE.txt").open("x") as stream:
                stream.write(traceback.format_exc())
        raise
