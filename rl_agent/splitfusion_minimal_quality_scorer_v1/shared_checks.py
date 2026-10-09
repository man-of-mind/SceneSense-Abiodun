"""Composition checks using independent oracle fixtures and published DTO."""
from contextlib import ExitStack
from dataclasses import replace
import hashlib
from unittest.mock import patch

from . import scorer as S
from .shared_admission import admit_prediction, join_admitted_inputs


def shared_fixture(scorer, case):
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C, evidence as V
    from rl_agent.splitfusion_numeric_gt_integration_v1.oracle_bridge import from_legacy
    source = V.admit_source_evidence(input_metadata=case["input_metadata"], avo_document=case["avo_document"],
        expected_capture_identity=case["prediction_identity"]["gt_identity"])
    action = case["prediction_identity"]["action_identity"]
    policy = {key: action[key] for key in C.POLICY_FIELDS}
    # This binds a replay identity, not a historical live plan or tracker replay.
    plan = C.sha256(C.canonical(dict(scope="OFFLINE_SCORER_REPLAY_NOT_LIVE_PLAN", action=action,
                                   capture=case["prediction_identity"]["gt_identity"])))
    target = from_legacy(prepared=case["prepared"], source_evidence=source, policy_identity=policy,
        binding=case["run_binding"], plan_sha256=plan,
        history_before_sha256=case["causal_history"]["previous_sha256"],
        history_after_sha256=case["causal_history"]["chain_sha256"])
    prediction = admit_prediction(scorer, identity=case["prediction_identity"], labels=case["predicted_labels"],
        records=case["predictions"], prediction_contract=case["prediction_contract"])
    reference = V.own_legacy_reference(target, source, target_tables=case["prepared"].target["tables"],
        old_prepared_sha256=case["prepared"].content_sha256)
    return target, prediction, reference


def check_shared(scorer, case, oracle):
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from .validate import guarded_score, forbidden, require
    target, prediction, reference = shared_fixture(scorer, case)
    require(target.segmentation_bytes == case["prepared"].target["segmentation"].tobytes(order="C")
        and target.object_ignore_bytes == case["prepared"].target["object_ignore"].tobytes(order="C")
        and C.thaw(target.metadata["object_boxes"]) == case["prepared"].target["tables"]["object_boxes.csv"]
        and C.thaw(target.metadata["avo"]) == case["avo_document"]["avo"], "shared numerical inputs changed")
    with ExitStack() as stack:
        for name in ("builtins.open", "io.open", "pathlib.Path.read_bytes", "pathlib.Path.read_text",
                     "hashlib.sha256", "json.dumps", "json.loads", "cv2.imread", "cv2.imdecode", "cv2.imencode"):
            stack.enter_context(patch(name, forbidden))
        for obj, name in ((E, "restore_prepared_target"), (E, "verify_sources"), (C, "thaw")):
            stack.enter_context(patch.object(obj, name, forbidden))
        frame = join_admitted_inputs(scorer, target=target, prediction=prediction)
    result, counts = guarded_score(scorer, frame)
    require(result.measurement == oracle.measurement, "shared DTO measurement differs")
    try:
        S.reconstruct_evidence(result)
    except ValueError:
        pass
    else:
        raise AssertionError("unavailable legacy evidence silently defaulted")
    with patch.dict(scorer._detection, match_frame=forbidden, load_gt=forbidden, load_predictions=forbidden), \
         patch.dict(scorer._person, greedy_match=forbidden):
        evidence = S.reconstruct_evidence(result, legacy_reference=reference)
        payloads = S.archive_payloads(result, legacy_reference=reference)
    require(evidence == oracle.evidence and payloads == E.archive_payloads(oracle), "shared DTO evidence/archive differs")
    for bad_reference in (replace(reference, semantic_sha256="0" * 64), None):
        try:
            S.reconstruct_evidence(result, legacy_reference=bad_reference)
        except ValueError:
            pass
        else:
            raise AssertionError("foreign/missing evidence reference accepted")
    return dict(exact_measurement=True, exact_evidence=True, exact_archive=True,
        semantic_sha256=target.semantic_sha256, original_prepared_sha256=reference.old_prepared_sha256,
        distinct_digest_domains=target.semantic_sha256 != reference.old_prepared_sha256,
        join_file_hash_png_json_thaw_calls=0, matching_call_counts=counts,
        late_artifact_reference_required=True, captured_history_receipts_only=True)


def check_shared_adversarial(scorer, case):
    import copy
    import numpy as np
    from .validate import require
    from rl_agent.splitfusion_live_dispatch_v1.registry import DispatchContractError
    checks = []
    def rejects(name, change):
        values = dict(identity=copy.deepcopy(case["prediction_identity"]), labels=case["predicted_labels"].copy(),
                      records=copy.deepcopy(case["predictions"]), prediction_contract=copy.deepcopy(case["prediction_contract"]))
        change(values)
        if name != "corrupt_prediction_digest":
            try:
                values["identity"]["prediction_digest"] = hashlib.sha256(values["labels"].tobytes(order="C")
                    + S.canonical(values["records"])).hexdigest()
            except (ValueError, TypeError):
                pass
        try:
            admit_prediction(scorer, **values)
        except (ValueError, TypeError, KeyError, OverflowError, DispatchContractError):
            checks.append(dict(name=name, outcome="REJECTED"))
        else:
            raise AssertionError("shared prediction boundary accepted: " + name)
    for field in scorer._detection["PREDICTION_FIELDS"]:
        def missing(values, field=field):
            values["records"][0]["score"] = .01
            values["records"][0].pop(field)
        rejects("missing_below_threshold:" + field, missing)
        rejects("nonfinite:" + field, lambda values, field=field: values["records"][0].__setitem__(field, "nan"))
    for value in (-.1, 1.1):
        rejects("score_bound:" + str(value), lambda values, value=value: values["records"][0].__setitem__("score", value))
    rejects("corrupt_prediction_digest", lambda values: values["identity"].__setitem__("prediction_digest", "0" * 64))
    rejects("native_model_grid_mismatch", lambda values: values.__setitem__("labels", np.zeros((432, 768), np.uint8)))
    rejects("native_dtype", lambda values: values.__setitem__("labels", values["labels"].astype(np.int64)))
    rejects("native_illegal_label", lambda values: values["labels"].fill(255))
    rejects("foreign_record_frame", lambda values: values["records"][0].__setitem__("frame_id", 99))
    rejects("record_count", lambda values: values.__setitem__("records", values["records"][:1] * 513))
    rejects("unknown_class", lambda values: values["records"][0].__setitem__("class_name", "unknown"))
    rejects("unused_nested_nonfinite", lambda values: values["records"][0].__setitem__("unused", [float("inf")]))
    rejects("native_calibration_hash", lambda values: values["prediction_contract"].__setitem__("camera_model_sha256", "0" * 64))
    rejects("missing_session", lambda values: values["identity"]["action_identity"].__setitem__("session_uuid", ""))
    target, prediction, reference = shared_fixture(scorer, case)
    frame = join_admitted_inputs(scorer, target=target, prediction=prediction)
    for mutation in (lambda: target.segmentation.setflags(write=True),
                     lambda: target.metadata["avo"]["qualified"].__setitem__(0, {}),
                     lambda: prediction.records[0].__setitem__("score", 0.),
                     lambda: replace(prediction, labels=b""),
                     lambda: join_admitted_inputs(scorer, target={"already_validated": True}, prediction=prediction)):
        try:
            mutation()
        except (AttributeError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("shared admitted alias/constructor/type gate failed")
    identity = copy.deepcopy(case["prediction_identity"])
    identity["gt_identity"]["run_id"] = "foreign_but_well_formed"
    foreign = admit_prediction(scorer, identity=identity, labels=case["predicted_labels"],
                               records=case["predictions"], prediction_contract=case["prediction_contract"])
    try:
        join_admitted_inputs(scorer, target=target, prediction=foreign)
    except ValueError:
        pass
    else:
        raise AssertionError("foreign complete identity accepted")
    checks.append(dict(name="shared_mutation_constructor_identity_gates", outcome="REJECTED"))
    result = scorer.score(frame)
    with patch("cv2.imencode", return_value=(False, None)):
        try:
            S.archive_payloads(result, legacy_reference=reference)
        except ValueError:
            pass
        else:
            raise AssertionError("archive encoding failure suppressed")
    require(scorer.score(frame).measurement == result.measurement, "deferred fault changed score")
    with patch.dict(scorer._detection, match_frame=lambda *args: (_ for _ in ()).throw(RuntimeError("test scoring fault"))):
        try:
            scorer.score(frame)
        except RuntimeError:
            pass
        else:
            raise AssertionError("scoring failure manufactured quality")
    checks.append(dict(name="scoring_and_archive_failures_remain_explicit", outcome="RAISED"))
    return checks
