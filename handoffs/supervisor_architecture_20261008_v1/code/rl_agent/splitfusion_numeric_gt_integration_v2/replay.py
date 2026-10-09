"""Sealed offline integration gates; does not start any live service."""
import argparse
import json
from pathlib import Path
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v1.producer_scorer_replay import (
    ROOT, PRODUCER_ROOT, SCORER_ROOT, verify_manifest, source_inventory, sha, write)
from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K, originals as O, source as NS
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import verify_environment, quality_bridge
from . import composition as X, fixtures as F, scoring as Q
from .ownership import ProducerOwner
from .join import BoundedTargetPredictionJoin

OUTPUT_ROOT = ROOT / "experiments/splitfusion_numeric_gt_integration_v2"
PACKAGE = Path(__file__).resolve().parent


def inventory(startup):
    pins = source_inventory(startup)
    for package in (PACKAGE, ROOT / "rl_agent/splitfusion_live_gt_optimization_v6",
            ROOT / "rl_agent/splitfusion_live_parity_runtime_v1",
            ROOT / "rl_agent/splitfusion_live_dispatch_v1", ROOT / "rl_agent/splitfusion_direct_edge_map_v1"):
        for path in package.glob("*.py"):
            pins[str(path.relative_to(ROOT))] = sha(path)
    return dict(sorted(pins.items()))


def create_output(output):
    output = Path(output).absolute()
    C.require(not output.exists() and output.resolve().is_relative_to(OUTPUT_ROOT)
        and not any(p.is_symlink() for p in (output, *output.parents)), "fresh owned result directory required")
    output.mkdir(parents=True, exist_ok=False)
    return output


def load_inputs(frames):
    environment = verify_environment()
    seals = {**verify_manifest(PRODUCER_ROOT), **verify_manifest(SCORER_ROOT)}
    startup = O.startup_binding()
    C.require(startup == json.loads((PRODUCER_ROOT / "full49_parity/PLAN.json").read_bytes())["startup_binding"],
              "producer startup differs from sealed proof")
    inputs = K.verify_inputs()
    cases = K.load_cases(limit=frames)
    predictions = K.load_predictions(cases)
    contexts = F.load_pose_contexts(cases, predictions)
    prediction_contract = K.read(K.CELL / "profiler/edge/prediction_archive/MANIFEST.json")["prediction_contract"]
    return dict(environment=environment, seals=seals, startup=startup, inputs=inputs, cases=cases,
                predictions=predictions, contexts=contexts, prediction_contract=prediction_contract)


def expectation(data, case, binding):
    prediction = data["predictions"][case.frame_id]
    context = data["contexts"][case.frame_id]["context"]
    return dict(expected_capture_identity=case.kwargs["identity"],
        expected_action_identity=prediction["identity"]["action_identity"], frame_context=context,
        plan_sha256=C.sha256(F.plan_for(case, prediction, context)), run_binding=binding,
        producer_binding=data["startup"]["binding_sha256"], prediction_contract=data["prediction_contract"])


def check_unchanged(data, sources):
    C.require(inventory(data["startup"]) == sources, "integration source changed during run")
    C.require({**verify_manifest(PRODUCER_ROOT), **verify_manifest(SCORER_ROOT)} == data["seals"],
              "sealed component changed during run")
    C.require(K.verify_inputs() == data["inputs"], "retained inputs changed during run")


def run(output, frames):
    C.require(frames in (2, 49), "only smoke2 or full49 chronological cohort allowed")
    data = load_inputs(frames)
    sources = inventory(data["startup"])
    output = create_output(output)
    binding = C.sha256(C.canonical(dict(scope="NUMERIC_GT_V2_OFFLINE_INTEGRATION", sources=sources,
                                       retained_input_seal=K.INPUT_SEAL_SHA)))
    write(output / "PLAN.json", dict(schema="splitfusion.numeric_gt.v2.offline_plan", binding=binding,
        source_sha256=sources, component_evidence_sha256=data["seals"], input_sha256=data["inputs"],
        unique_frames=frames, environment=data["environment"],
        actual_archived_ue_edge_context=True, live_source_reconstruction=False,
        acknowledgement_scope="OFFLINE_HARNESS_NOT_SOCKET_SEND_OR_UE_RECEIPT",
        order="NUMERICAL_READY_AND_SCORE_BEFORE_FULL_AVO_AND_PNG_ARCHIVE",
        performance_claim=False, live_authorization=False))
    rows = []
    try:
        suite = unittest.defaultTestLoader.discover(str(PACKAGE), pattern="test_*.py", top_level_dir=str(ROOT))
        tests = unittest.TextTestRunner(verbosity=1).run(suite)
        C.require(tests.wasSuccessful(), "v2 adversarial tests failed")
        scorer = Q.NumericalScorer()
        join = BoundedTargetPredictionJoin(scorer, max_jobs=2, max_bytes=2 * X.RESERVATION)
        spec, quality = quality_bridge()
        producer = NS.NumericSourceProducer(**K.producer_arguments(data["cases"][0]))
        with ProducerOwner(producer, max_retained_jobs=2, max_retained_bytes=2 * X.RESERVATION) as owner:
            for index, case in enumerate(data["cases"]):
                value = X.candidate(case, data["predictions"][case.frame_id], owner=owner, scorer=scorer, join=join,
                    expected=expectation(data, case, binding), spec=spec, quality=quality, guarded=True, index=index)
                checks = X.check_exact(value, case, data["predictions"][case.frame_id])
                rows.append(dict(frame_id=case.frame_id, q_perc=value["q_perc"], exact=checks,
                    ready_semantic_sha256=value["ready_semantic_sha256"], packet_bytes=value["packet_bytes"],
                    history_before_sha256=value["history_before_sha256"], history_after_sha256=value["history_after_sha256"],
                    archive_absent_before_scoring=value["archive_absent_before_scoring"],
                    actual_context_camera_delta=data["contexts"][case.frame_id]["camera_delta"],
                    matching_calls=value["matching_calls"], owner_snapshot=value["owner_snapshot"],
                    join_snapshot=value["join_snapshot"]))
                if (index + 1) % 10 == 0:
                    print(json.dumps(dict(composed=index + 1, total=frames)), flush=True)
        join.close()
        undefined = [row["frame_id"] for row in rows if row["q_perc"] is None]
        C.require(frames != 49 or len(undefined) == 8, "undefined-quality population differs")
        check_unchanged(data, sources)
        record = dict(status="ALL49_ARCHIVE_FREE_NUMERIC_COMPOSITION_EXACT" if frames == 49 else "SMOKE2_COMPONENT_CHECK_ONLY",
            tests_run=tests.testsRun, tests_failed=len(tests.failures) + len(tests.errors),
            unique_frames=frames, undefined_quality_frames=undefined, rows=rows,
            owner_closed=owner.snapshot(), source_files=len(sources), input_files=len(data["inputs"]),
            numerical_quality_and_archive_exact=True, full_AVO_and_PNG_not_needed_before_score=True,
            real_producer_commit_used=True, actual_archived_context_reconciled=True,
            live_dispatch_integrated=False, publisher_authenticated=False, OS_IPC_measured=False,
            performance_measured=False, live_deadline_claim=False)
        write(output / "RESULT.json", record)
        print(json.dumps({k: record[k] for k in ("status", "unique_frames", "tests_run", "undefined_quality_frames")}))
    except BaseException as error:
        write(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}", completed=len(rows), live_run=False))
        raise
    finally:
        write(output / "OUTPUT_SHA256.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(2, 49), default=49)
    args = parser.parse_args()
    run(args.output, args.frames)
