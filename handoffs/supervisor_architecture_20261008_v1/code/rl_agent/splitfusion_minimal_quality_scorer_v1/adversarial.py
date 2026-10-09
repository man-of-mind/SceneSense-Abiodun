"""Deterministic in-memory boundary fixtures, independently scored by oracle.

Synthetic targets are derived test inputs, not new scientific ground truth.
No source archive is altered and no derived fixture is eligible for live use.
"""
from __future__ import annotations

import copy
from dataclasses import replace, FrozenInstanceError
import hashlib
import json
import math

import cv2
import numpy as np

from . import scorer as S
from .fixtures import admission_arguments, oracle_arguments
from .offline_admission import admit_oracle_fixture


def check_adversarial(scorer, original):
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    from rl_agent.splitfusion_live_parity_repair_v2.prediction_archive import PredictionContract
    from .validate import guarded_score, guarded_evidence, require
    checks = []

    def seal_prediction(case):
        case["prediction_identity"]["prediction_digest"] = hashlib.sha256(
            case["predicted_labels"].tobytes(order="C") + S.canonical(case["predictions"])).hexdigest()

    def seal_target(case):
        previous = case["prepared"]
        target = previous.target
        for name in ("segmentation", "object_ignore"):
            success, png = cv2.imencode(".png", target[name])
            require(success, "test PNG encoding failed")
            target[name + "_png"] = bytes(png)
        case["prepared"] = E.PreparedEvaluationTarget(previous.frame_id, previous.sample_id, target,
            E._target_digest(previous.frame_id, previous.sample_id, target))

    def compare(name, case):
        seal_target(case)
        seal_prediction(case)
        admitted = admit_oracle_fixture(scorer, **admission_arguments(case))
        result, _ = guarded_score(scorer, admitted)
        evidence, payloads = guarded_evidence(scorer, result)
        oracle = E.evaluate_prepared(**oracle_arguments(case))
        require(result.measurement == oracle.measurement and evidence == oracle.evidence
            and payloads == E.archive_payloads(oracle), "boundary parity failed: " + name)
        checks.append(dict(name=name, outcome="EXACT", measurement=result.measurement,
                           associations=repr(result.associations)))
        return result

    def base():
        case = copy.deepcopy(original)
        target = case["prepared"].target
        target["segmentation"] = np.zeros((432, 768), np.uint8)
        target["object_ignore"] = np.zeros((432, 768), np.uint8)
        target["tables"]["object_boxes.csv"] = []
        case["predictions"] = []
        case["avo_document"]["avo"]["qualified"] = []
        case["avo_document"]["avo"]["structural_ignored"] = []
        return case

    template = copy.deepcopy(original["prepared"].target["tables"]["object_boxes.csv"][0])
    def target(cls, x, identifier):
        return dict(template, label=cls, object_world_x=str(x), object_world_y="0.0",
                    source_identity="synthetic:" + identifier)
    def prediction(cls, x, score=.5, cx=-100, cy=-100):
        return dict(original["predictions"][0], class_name=cls, world_x=x, world_y=0.,
                    score=score, center_x_px=cx, center_y_px=cy)
    def actor(x, identifier, avo=None):
        row = dict(frame_id=original["prepared"].frame_id, gt_actor_id=identifier,
                   world_x=float(x), world_y=0., qualified=avo is not None)
        row.update(dict(avo=avo) if avo is not None else dict(reasons=["synthetic_structural_boundary"]))
        return row

    case = base()
    case["prepared"].target["tables"]["object_boxes.csv"] = [target("person", 50, "p"), target("vehicle", 0, "v")]
    case["predictions"] = [prediction("vehicle", 3., .20),
        prediction("vehicle", math.nextafter(3., math.inf), .20),
        prediction("vehicle", 1000., math.nextafter(.20, 0.))]
    result = compare("vehicle_inclusive_score_radius_and_full_table_index", case)
    require(result.measurement["loc_vehicle_tp"] == result.measurement["loc_vehicle_fp"] == 1
            and result.associations.vehicle_prediction_order[0][:2] == (0, 1), "vehicle boundary expectation")

    case = base()
    case["prepared"].target["tables"]["object_boxes.csv"] = [target("person", 0, "p"),
        target("vehicle", 0, "v0"), target("vehicle", 0, "v1")]
    case["predictions"] = [prediction("vehicle", 0), prediction("person", 0), prediction("vehicle", 0)]
    result = compare("stable_score_class_distance_prediction_target_ties", case)
    require([pair[:2] for pair in result.associations.vehicle_prediction_order] == [(0, 1), (2, 2)], "tie order")

    case = base()
    case["prepared"].target["tables"]["object_boxes.csv"] = [target("vehicle", x, str(x)) for x in (0, 10, 20)]
    case["predictions"] = [prediction("vehicle", 2.5, .9), prediction("vehicle", 10.125, .8), prediction("vehicle", 21, .7)]
    result = compare("different_distance_and_prediction_error_order", case)
    require(result.measurement["loc_vehicle_matched_xy_errors_m"] == [2.5, .125, 1.]
            and [x[2] for x in result.associations.detection_distance_order] == [.125, 1., 2.5], "error ordering")

    case = base()
    case["prepared"].target["object_ignore"][0, [0, 2]] = 255
    case["predictions"] = [prediction("vehicle", 1000, cx=x, cy=0) for x in (.5, 1.5, 2.5, -.5, -.5001, 767.5)]
    result = compare("ignore_bankers_rounding_and_outside_grid", case)
    require(result.measurement["loc_vehicle_ignored_predictions"] == 4 and result.measurement["loc_vehicle_fp"] == 2,
            "neutral rounding expectation")

    case = base()
    case["prepared"].target["object_ignore"][:] = 255
    case["prepared"].target["tables"]["object_boxes.csv"] = [target("vehicle", 0, "matched")]
    case["predictions"] = [prediction("vehicle", 0, cx=0, cy=0), prediction("vehicle", 10, cx=0, cy=0)]
    result = compare("match_precedes_ignore_neutralization", case)
    require(result.measurement["loc_vehicle_tp"] == result.measurement["loc_vehicle_ignored_predictions"] == 1, "match before ignore")

    case = base()
    case["avo_document"]["avo"]["qualified"] = [actor(0, "ignored", math.nextafter(.65, 0)), actor(3, "eligible", .65)]
    case["avo_document"]["avo"]["structural_ignored"] = [actor(20, "structural")]
    case["predictions"] = [prediction("person", 0, .25), prediction("person", 0, .25),
        prediction("person", 20, .25), prediction("person", 20, .25),
        prediction("person", 1000, math.nextafter(.25, 0))]
    result = compare("person_eligibility_then_avo_then_structural_one_to_one", case)
    m = result.measurement
    require((m["loc_person_tp"], m["loc_person_avo_ignored_predictions"],
             m["loc_person_structural_ignored_predictions"], m["loc_person_fp"]) == (1, 1, 1, 1), "person stages")
    require(result.associations.person_prediction_order == ((0, 0, 3.),), "person radius/eligible index")

    case = base()
    case["avo_document"]["avo"]["qualified"] = [actor(0, "a", 1.), actor(0, "b", .65)]
    case["predictions"] = [prediction("person", 0, .25), prediction("person", 0, .9)]
    result = compare("person_original_record_order_despite_different_scores", case)
    require([p[:2] for p in result.associations.person_prediction_order] == [(0, 0), (1, 1)], "person original order")

    for shape, all_ignore in (((216, 384), False), ((432, 768), True), ((1, 1), False), ((448, 768), False)):
        case = base()
        truth = np.indices((432, 768)).sum(axis=0).astype(np.uint8) % 3
        truth[::7, ::5] = 255
        if all_ignore:
            truth[:] = 255
        case["prepared"].target["segmentation"] = truth
        labels = (np.indices(shape).sum(axis=0) % 3).astype(np.uint8)
        case["predicted_labels"] = labels[:, ::-1]  # noncontiguous prediction ownership
        case["prediction_contract"] = PredictionContract.cpu_fixture(shape).document()
        compare(f"segmentation_nearest_ignore_noncontiguous_{shape}_{all_ignore}", case)
    compare("empty_predictions_empty_localization_undefined_support", base())

    def reject(name, change, reseal_prediction=True):
        case = copy.deepcopy(original)
        change(case)
        if reseal_prediction:
            try:
                seal_prediction(case)
            except (TypeError, ValueError):
                pass  # canonical nonfinite rejection itself must remain active
        try:
            admit_oracle_fixture(scorer, **admission_arguments(case))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            checks.append(dict(name=name, outcome="REJECTED", exception=type(exc).__name__))
        else:
            raise AssertionError("malformed fixture accepted: " + name)

    for field in scorer._detection["PREDICTION_FIELDS"]:
        def missing(case, field=field):
            case["predictions"][0]["score"] = .01
            case["predictions"][0].pop(field)
        reject("missing_full_field_even_below_threshold:" + field, missing)
        reject("nonfinite_full_field:" + field,
            lambda case, field=field: case["predictions"][0].__setitem__(field, "nan"))
    for score in (-.001, 1.001):
        reject("score_bound:" + str(score), lambda case, value=score: case["predictions"][0].__setitem__("score", value))
    reject("foreign_record_frame", lambda case: case["predictions"][0].__setitem__("frame_id", 999999))
    reject("unknown_record_class", lambda case: case["predictions"][0].__setitem__("class_name", "cyclist"))
    reject("prediction_record_count", lambda case: case.__setitem__("predictions", [case["predictions"][0]] * 513))
    reject("prediction_digest", lambda case: case["prediction_identity"].__setitem__("prediction_digest", "0" * 64), False)
    reject("nonfinite_unused_nested_record_value", lambda case: case["predictions"][0].__setitem__("unused", [float("inf")]))
    reject("foreign_capture_identity", lambda case: case["prediction_identity"]["gt_identity"].__setitem__("run_id", "foreign"))
    reject("capture_timestamp", lambda case: case["prediction_identity"].__setitem__("capture_timestamp_ns", 3))
    reject("action_session_missing", lambda case: case["prediction_identity"]["action_identity"].__setitem__("session_uuid", ""))
    reject("action_frame", lambda case: case["prediction_identity"]["action_identity"].__setitem__("frame_id", 42))
    reject("hold_not_reward_ticket", lambda case: case["prediction_identity"]["action_identity"].update(frame_kind="POLICY_HOLD", reward_requested=False))
    reject("pose_mismatch", lambda case: case["avo_document"]["camera_inverse"][0].__setitem__(0, 99.))
    reject("run_binding_missing", lambda case: case.__setitem__("run_binding", ""))
    reject("input_metadata_provenance", lambda case: case["input_metadata"]["provenance"].__setitem__("input_content_sha256", "0" * 64))
    reject("target_digest", lambda case: case.__setitem__("prepared", replace(case["prepared"], content_sha256="0" * 64)))
    reject("target_png_mismatch", lambda case: case["prepared"].target.__setitem__("segmentation_png", b"not-png"))
    reject("missing_complete_declaration", lambda case: case["prepared"].target["provenance"]["complete"].clear())
    reject("foreign_sample_table", lambda case: case["prepared"].target["tables"]["object_boxes.csv"][0].__setitem__("sample_id", "foreign"))
    reject("gt_shape", lambda case: case["prepared"].target.__setitem__("segmentation", np.zeros((1, 1), np.uint8)))
    reject("gt_dtype", lambda case: case["prepared"].target.__setitem__("segmentation", np.zeros((432, 768), np.int64)))
    for shape in ((0, 1), (449, 1), (1, 769)):
        reject("prediction_bounds:" + str(shape), lambda case, shape=shape: case.__setitem__("predicted_labels", np.zeros(shape, np.uint8)))
    reject("prediction_illegal_label", lambda case: case.__setitem__("predicted_labels", np.full((2, 2), 255, np.uint8)))
    reject("prediction_dtype", lambda case: case.__setitem__("predicted_labels", np.zeros((2, 2), np.float32)))
    reject("avo_threshold", lambda case: case["avo_document"]["avo"].__setitem__("threshold", .5))
    reject("avo_missing_structural", lambda case: case["avo_document"]["avo"].pop("structural_ignored"))
    reject("avo_duplicate_actor", lambda case: case["avo_document"]["avo"]["structural_ignored"].append(
        copy.deepcopy(case["avo_document"]["avo"]["structural_ignored"][0])))
    reject("avo_nonfinite_unused_actor", lambda case: case["avo_document"]["avo"]["structural_ignored"][0].__setitem__("world_x", float("inf")))
    reject("avo_foreign_frame", lambda case: case["avo_document"]["avo"]["structural_ignored"][0].__setitem__("frame_id", 42))
    reject("avo_out_of_range", lambda case: case["avo_document"]["avo"].__setitem__("qualified", [actor(0, "bad", 1.01)]))

    case = copy.deepcopy(original)
    frame = admit_oracle_fixture(scorer, **admission_arguments(case))
    before = scorer.score(frame)
    case["predictions"][0]["world_x"] += 10000
    case["prepared"].target["tables"]["object_boxes.csv"].clear()
    case["avo_document"]["avo"]["structural_ignored"].clear()
    case["predicted_labels"].flags.writeable = True
    case["predicted_labels"][:] = 0
    require(scorer.score(frame) == before, "caller mutation changed admitted score")
    for mutation in (lambda: frame.label_view().setflags(write=True),
                     lambda: frame.predictions[0].__setitem__("score", 0.),
                     lambda: setattr(frame, "frame_id", 99),
                     lambda: replace(frame, frame_id=99),
                     lambda: scorer.score({"already_validated": True}),
                     lambda: S.NumericalScorer().score(frame)):
        try:
            mutation()
        except (AttributeError, ValueError, TypeError, FrozenInstanceError):
            pass
        else:
            raise AssertionError("immutable/private-type gate failed")
    exported = before.measurement
    exported["loc_vehicle_matched_xy_errors_m"].append(999.)
    evidence = S.reconstruct_evidence(before)
    evidence["person_avo"]["structural_ignored"].clear()
    require(scorer.score(frame) == before and S.reconstruct_evidence(before) == S.reconstruct_evidence(scorer.score(frame)),
            "returned measurement/evidence mutated retained input")
    checks.append(dict(name="bytes_rows_nested_metadata_caller_return_and_constructor_mutation", outcome="REJECTED_OR_DETACHED"))
    return checks
