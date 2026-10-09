"""V2 numerical/deferred boundaries; no service, training or benchmark."""
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C, evidence as V
from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, target_transport as T
from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as SP
from rl_agent.splitfusion_direct_edge_map_v1.protocol import DirectMapProtocolError
from . import ready as R, scoring as G


def forbidden(*args, **kwargs):
    raise AssertionError("forbidden on numerical/deferred seam")


def fixture(scorer):
    """Explicit synthetic native-calibrated fixture, not a producer proof."""
    from .test_ready import fixture as ready_fixture
    payload, expected = ready_fixture()
    target = R.admit_payload(payload, **expected)
    capture, action = C.thaw(target.metadata["capture_identity"]), C.thaw(target.metadata["action_identity"])
    records = [dict.fromkeys(scorer._detection["PREDICTION_FIELDS"], 0.0)]
    records[0].update(class_name="person", frame_id=capture["frame_id"], score=.8,
        world_x=1., world_y=2., yaw_cos=1., size_x=1., size_y=1., size_z=1., bbox_x1=2., bbox_y1=2.)
    labels = np.zeros((720, 1280), np.uint8)
    identity = dict(gt_identity=capture, action_identity=action,
        capture_timestamp_ns=capture["capture_timestamp_ns"],
        prediction_digest=hashlib.sha256(labels.tobytes() + S.canonical(records)).hexdigest())
    prediction_kwargs = dict(identity=identity, labels=labels, records=records,
        prediction_contract=C.thaw(target.metadata["prediction_contract"]))
    prediction = G.admit_prediction(scorer, **prediction_kwargs)
    inputs = dict(frame=C.thaw(target.metadata["frame"]), split="val", actor_boxes=[], actor_visibility=[],
        static_candidates=[], quarantine=dict.fromkeys(T.targets.QUARANTINE_KEYS, []),
        provenance=C.thaw(target.metadata["provenance"]))
    document = dict(schema=SP.LG.SCHEMA, mask_contract=SP.LG.MASK_CONTRACT, record_kind="AVO_SCORED_FRAME",
        identity=capture, source_frame_id=capture["frame_id"], native_shape=[720, 1280], ready_wall_ns=123456,
        camera_matrix=C.thaw(target.metadata["camera_matrix"]),
        camera_inverse=C.thaw(target.metadata["camera_inverse"]), intrinsics=C.thaw(target.metadata["intrinsics"]),
        avo=C.thaw(target.metadata["avo"]))
    source_kwargs = dict(input_metadata=inputs, avo_document=document, expected_capture_identity=capture)
    source = V.admit_source_evidence(**source_kwargs)
    seg_ok, seg = cv2.imencode(".png", target.segmentation)
    obj_ok, obj = cv2.imencode(".png", target.object_ignore)
    assert seg_ok and obj_ok
    tables = {"object_boxes.csv": C.thaw(target.metadata["object_boxes"]),
        "object_ignore_regions.csv": C.thaw(target.metadata["object_ignore_regions"]),
        "target_manifest.csv": [dict(sample_id=target.metadata["sample_id"],
            segmentation_mask_path="segmentation.png", segmentation_mask_sha256=hashlib.sha256(bytes(seg)).hexdigest(),
            object_ignore_mask_path="object_ignore.png", object_ignore_mask_sha256=hashlib.sha256(bytes(obj)).hexdigest())]}
    legacy = dict(contract=C.TARGET_CONTRACT, provenance=C.thaw(target.metadata["provenance"]), live_qualified=False,
        segmentation=target.segmentation, object_ignore=target.object_ignore,
        segmentation_png=bytes(seg), object_ignore_png=bytes(obj), tables=tables)
    digest = E._target_digest(capture["frame_id"], target.metadata["sample_id"], legacy)
    prepared = E.restore_prepared_target(frame_id=capture["frame_id"], sample_id=target.metadata["sample_id"],
        target=legacy, content_sha256=digest)
    reference_kwargs = dict(source_evidence=source, target_tables=tables, old_prepared_sha256=digest,
        prepared_target=prepared)
    return dict(target=target, prediction=prediction, prediction_kwargs=prediction_kwargs,
        source_kwargs=source_kwargs, reference_kwargs=reference_kwargs, prepared=prepared,
        expected=expected, payload=payload)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = G.NumericalScorer()
        cls.case = fixture(cls.scorer)

    def join(self, **changes):
        values = dict(target=self.case["target"], prediction=self.case["prediction"])
        values.update(changes)
        return G.join_admitted_inputs(self.scorer, **values)

    def attach(self, **changes):
        values = dict(self.case["reference_kwargs"]); values.update(changes)
        return G.attach_archive(self.case["target"], **values)

    def test_join_score_needs_no_archive_json_hash_io_png(self):
        with ExitStack() as stack:
            for name in ("builtins.open", "io.open", "pathlib.Path.read_bytes", "pathlib.Path.read_text",
                         "hashlib.sha256", "json.dumps", "json.loads", "cv2.imread", "cv2.imdecode", "cv2.imencode"):
                stack.enter_context(patch(name, forbidden))
            for obj, name in ((E, "restore_prepared_target"), (E, "verify_sources"), (C, "thaw"),
                              (G, "attach_archive"), (V, "admit_source_evidence")):
                stack.enter_context(patch.object(obj, name, forbidden))
            frame = self.join()
            result = self.scorer.score(frame)
        self.assertEqual(result.measurement["frame_id"], self.case["target"].metadata["capture_identity"]["frame_id"])
        self.assertIsNone(frame.original_content_sha256)
        self.assertEqual(frame.evidence_inputs, b"")

    def test_original_measurement_evidence_png_and_saved_associations_exact(self):
        case = self.case
        oracle = E.evaluate_prepared(identity=case["prediction_kwargs"]["identity"]["action_identity"],
            prepared=case["prepared"], predicted_labels=case["prediction_kwargs"]["labels"],
            predictions=case["prediction_kwargs"]["records"], avo=case["source_kwargs"]["avo_document"]["avo"])
        result = self.scorer.score(self.join())
        self.assertEqual(result.measurement, oracle.measurement)
        reference = self.attach()
        with patch.dict(self.scorer._detection, match_frame=forbidden, load_gt=forbidden, load_predictions=forbidden), \
             patch.dict(self.scorer._person, greedy_match=forbidden):
            evidence = G.reconstruct_evidence(result, archive_reference=reference)
            payloads = G.archive_payloads(result, archive_reference=reference)
        self.assertEqual(evidence, oracle.evidence)
        self.assertEqual(payloads, E.archive_payloads(oracle))
        copied = G.bind_archive_result(result, archive_reference=reference)
        self.assertIs(copied.associations, result.associations)
        self.assertIs(copied._measurement, result._measurement)
        self.assertIsNone(copied._frame.shared_context)
        self.assertIsNotNone(result._frame.shared_context)
        self.assertIsNone(result._frame.original_content_sha256)

    def test_all_action_identity_fields_not_just_policy_counters_match(self):
        action = self.case["prediction_kwargs"]["identity"]["action_identity"]
        replacements = {"session_uuid": "foreign", "decision_seq": 99, "tensor_seq": 99, "ticket_seq": 99,
            "frame_id": action["frame_id"] + 1, "mode_id": (action["mode_id"] + 1) % 12,
            "q_e4": (action["q_e4"] + 1) % 9801, "anchor_action_id": 71 if action["anchor_action_id"] != 71 else 0,
            "controller_lineage_sha256": "4" * 64, "execution_bundle_sha256": "5" * 64,
            "frame_kind": "POLICY_HOLD", "reward_requested": False}
        self.assertEqual(set(action), set(replacements))
        for field, value in replacements.items():
            with self.subTest(field=field):
                kw = copy.deepcopy(self.case["prediction_kwargs"])
                kw["identity"]["action_identity"][field] = value
                with self.assertRaises((ValueError, TypeError, DirectMapProtocolError)):
                    prediction = G.admit_prediction(self.scorer, **kw)
                    self.join(prediction=prediction)

    def test_foreign_capture_and_native_contract_refused(self):
        kw = copy.deepcopy(self.case["prediction_kwargs"])
        kw["identity"]["gt_identity"]["run_id"] = "foreign"
        prediction = G.admit_prediction(self.scorer, **kw)
        with self.assertRaisesRegex(ValueError, "capture/action"):
            self.join(prediction=prediction)
        from rl_agent.splitfusion_live_parity_repair_v2.prediction_archive import PredictionContract
        kw = copy.deepcopy(self.case["prediction_kwargs"])
        kw["prediction_contract"] = PredictionContract.cpu_fixture((720, 1280)).document()
        prediction = G.admit_prediction(self.scorer, **kw)
        with self.assertRaisesRegex(ValueError, "calibration/grid"):
            self.join(prediction=prediction)

    def test_wrong_owner_and_unadmitted_objects_refused(self):
        another = G.NumericalScorer()
        foreign = G.admit_prediction(another, **self.case["prediction_kwargs"])
        for changes in ({"prediction": foreign}, {"prediction": {"already_validated": True}},
                        {"target": {"already_validated": True}}):
            with self.assertRaises(ValueError):
                self.join(**changes)

    def test_original_full_prediction_validation_below_threshold_preserved(self):
        for field in self.scorer._detection["PREDICTION_FIELDS"]:
            with self.subTest(field=field):
                kw = copy.deepcopy(self.case["prediction_kwargs"])
                kw["records"][0]["score"] = .01
                del kw["records"][0][field]
                kw["identity"]["prediction_digest"] = hashlib.sha256(kw["labels"].tobytes()
                    + S.canonical(kw["records"])).hexdigest()
                with self.assertRaises((ValueError, KeyError)):
                    G.admit_prediction(self.scorer, **kw)

    def test_source_frame_provenance_numeric_avo_pose_intrinsics_match(self):
        changes = (
            lambda kw: kw["input_metadata"]["frame"].update(timestamp=99.),
            lambda kw: kw["input_metadata"]["provenance"]["evidence_sha256"].update(owned_source="3" * 64),
            lambda kw: kw["avo_document"]["avo"].update(timestamp=99.),
            lambda kw: kw["avo_document"]["camera_matrix"][0].__setitem__(3, 99.),
            lambda kw: kw["avo_document"]["camera_inverse"][0].__setitem__(3, 99.),
            lambda kw: kw["avo_document"]["intrinsics"][0].__setitem__(0, 99.),
            lambda kw: kw["avo_document"].pop("camera_matrix"),
            lambda kw: kw["input_metadata"]["provenance"].update(input_content_sha256="3" * 64),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                kw = copy.deepcopy(self.case["source_kwargs"]); change(kw)
                with self.assertRaises(ValueError):
                    source = V.admit_source_evidence(**kw)
                    self.attach(source_evidence=source)

    def test_attachment_tables_manifest_and_digest_checked(self):
        mutations = (
            lambda t: t.pop("object_boxes.csv"),
            lambda t: t["object_boxes.csv"].append({"invented": "row"}),
            lambda t: t["target_manifest.csv"][0].update(sample_id="foreign"),
            lambda t: t["target_manifest.csv"][0].update(segmentation_mask_path="../escape.png"),
            lambda t: t["target_manifest.csv"][0].update(segmentation_mask_sha256="bad"),
            lambda t: t["target_manifest.csv"].append(dict(t["target_manifest.csv"][0])),
        )
        for change in mutations:
            tables = copy.deepcopy(self.case["reference_kwargs"]["target_tables"]); change(tables)
            with self.assertRaises(ValueError):
                self.attach(target_tables=tables)
        with self.assertRaises(ValueError):
            self.attach(old_prepared_sha256="bad")
        with self.assertRaises(ValueError):
            self.attach(old_prepared_sha256="0" * 64)
        prepared = self.case["prepared"]
        corrupt = replace(prepared, target=dict(prepared.target, segmentation_png=b"corrupt PNG"))
        with self.assertRaises(ValueError):
            self.attach(prepared_target=corrupt)
        changed = prepared.target["segmentation"].copy()
        changed.flat[0] = 1 if changed.flat[0] != 1 else 2
        corrupt = replace(prepared, target=dict(prepared.target, segmentation=changed))
        with self.assertRaises(ValueError):
            self.attach(prepared_target=corrupt)
        with self.assertRaises(ValueError):
            self.attach(prepared_target={"already_validated": True})
        with patch.object(G, "MAX_ARCHIVE_BINDING_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "bound"):
                self.attach()

    def test_reference_private_owned_immutable_and_foreign_target_refused(self):
        reference = self.attach()
        with self.assertRaises(ValueError):
            G.ArchiveReference(already_validated=True)
        with self.assertRaises((TypeError, ValueError)):
            replace(reference, old_prepared_sha256="0" * 64)
        with self.assertRaises(TypeError):
            reference.target_tables["target_manifest.csv"][0]["sample_id"] = "changed"
        with self.assertRaises(FrozenInstanceError):
            reference.semantic_sha256 = "0" * 64
        result = self.scorer.score(self.join())
        with self.assertRaises(ValueError):
            G.reconstruct_evidence(result, archive_reference=None)
        expected = dict(self.case["expected"], plan_sha256="6" * 64)
        target = R.admit_payload(self.case["payload"], **expected)
        foreign_result = self.scorer.score(self.join(target=target))
        with self.assertRaisesRegex(ValueError, "different ready"):
            G.archive_payloads(foreign_result, archive_reference=reference)

    def test_archive_encoder_failure_does_not_change_score(self):
        result = self.scorer.score(self.join())
        before = result.measurement
        reference = self.attach()
        with patch("cv2.imencode", return_value=(False, None)):
            with self.assertRaisesRegex(ValueError, "PNG encoding"):
                G.archive_payloads(result, archive_reference=reference)
        self.assertEqual(result.measurement, before)
        self.assertIsNone(result._frame.original_content_sha256)


if __name__ == "__main__":
    unittest.main()
