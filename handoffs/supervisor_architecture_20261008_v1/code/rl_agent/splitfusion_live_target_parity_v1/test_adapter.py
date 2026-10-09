from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from . import adapter as A
from .fixtures import FIXTURES, load_retained_fixture


class RetainedParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {sid: load_retained_fixture(sid) for sid in FIXTURES}

    def test_original_complete_v010_matches_every_retained_byte_and_row(self):
        reasons, positives = set(), set()
        for sid, (inputs, references) in self.fixtures.items():
            with self.subTest(sample_id=sid):
                result = A.build_complete_v010(inputs)
                self.assertEqual(result["segmentation_png"], references["segmentation"].read_bytes())
                self.assertEqual(result["object_ignore_png"], references["object_ignore"].read_bytes())
                for name in ("object_boxes.csv", "object_ignore_regions.csv"):
                    self.assertEqual(result["tables"][name], references[name])
                self.assertFalse(result["live_qualified"])
                reasons.update(row["reason"] for row in result["tables"]["object_ignore_regions.csv"])
                positives.update((row["label"], row["source_kind"]) for row in result["tables"]["object_boxes.csv"])
        self.assertEqual(positives, {("person", "actor"), ("vehicle", "actor"), ("vehicle", "environment_static")})
        self.assertTrue({"registered_manual_panel_quarantine", "registered_identity_or_frame_conflict",
                         "zero_semantic_depth_only_quarantine", "unresolved_semantic_vehicle_component",
                         "v010_unobservable_or_outside_geometry_gate"}.issubset(reasons))

    def test_v3_person_paint_exact_but_never_labelled_v010(self):
        _builder, visibility, _recon = A._original_modules(A.ROOT)
        for inputs, _ in self.fixtures.values():
            result = A.original_v3_person_mask(vehicle_mask=(inputs.original_v3_mask == 1).astype(np.uint8),
                depth_m=visibility.decode_depth_bgra(inputs.depth_bgra),
                person_visibility=[r for r in inputs.actor_visibility if r["label"] == "person"])
            np.testing.assert_array_equal(result["mask"], inputs.original_v3_mask)
            self.assertIn("NOT_V010", result["contract"])
            self.assertFalse(result["live_qualified"])

    def test_each_missing_completeness_declaration_is_fatal(self):
        inputs = next(iter(self.fixtures.values()))[0]
        for key in A.COMPLETENESS_KEYS:
            complete = dict(inputs.provenance["complete"])
            complete.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(A.TargetInputError, "incomplete"):
                A.build_complete_v010(replace(inputs, provenance=dict(inputs.provenance, complete=complete)))

    def test_quarantine_cannot_default_to_empty(self):
        inputs = next(iter(self.fixtures.values()))[0]
        with self.assertRaisesRegex(A.TargetInputError, "quarantine"):
            A.build_complete_v010(replace(inputs, quarantine={}))

    def test_modified_arrays_cannot_reuse_bound_provenance(self):
        inputs = next(iter(self.fixtures.values()))[0]
        mask = inputs.original_v3_mask.copy()
        mask[0, 0] = 1 - int(mask[0, 0] == 1)
        with self.assertRaisesRegex(A.TargetInputError, "content digest"):
            A.build_complete_v010(replace(inputs, original_v3_mask=mask))

    def test_foreign_frame_rows_refused(self):
        inputs = next(iter(self.fixtures.values()))[0]
        static = list(inputs.static_candidates)
        static[0] = dict(static[0], sample_id="foreign")
        with self.assertRaisesRegex(A.TargetInputError, "foreign-frame"):
            A.build_complete_v010(replace(inputs, static_candidates=static))

    def test_shape_refused_before_original_builder(self):
        inputs = next(iter(self.fixtures.values()))[0]
        with self.assertRaisesRegex(A.TargetInputError, "native uint8"):
            A.build_complete_v010(replace(inputs, depth_bgra=inputs.depth_bgra[:, :, :3]))

    def test_source_pin_fail_closed(self):
        with patch.object(A, "sha256", return_value="0" * 64), self.assertRaisesRegex(A.TargetInputError, "source changed"):
            A.verify_sources()

    def test_current_avo_input_is_not_complete_v010(self):
        result = A.audit_saved_live_bundle({"payloads": {}, "raw_pedestrian_rows": [], "intrinsics": []})
        self.assertEqual(result["status"], "EXACT_V010_INPUTS_UNAVAILABLE")
        self.assertFalse(result["can_score_corrected_v010"])
        self.assertEqual(len(result["missing"]), 5)

    def test_scorer_resizes_prediction_to_target_not_the_reverse(self):
        inputs = next(iter(self.fixtures.values()))[0]
        built = A.build_complete_v010(inputs)
        prediction = np.zeros((217, 389), dtype=np.uint8)
        prediction[20:100, 70:230] = 1
        prediction[125:200, 240:350] = 2
        result = A.score_complete_v010(built, prediction)
        resized = cv2.resize(prediction, (768, 432), interpolation=cv2.INTER_NEAREST)
        target = built["segmentation"]
        valid = target != 255
        expected = np.bincount(target[valid].astype(np.int64) * 3 + resized[valid], minlength=9).reshape(3, 3)
        np.testing.assert_array_equal(result["original_score"]["confusion_matrix"], expected)
        self.assertEqual(result["original_score"]["ignored_pixels"], int(np.count_nonzero(~valid)))
        self.assertEqual(int(expected.sum()), int(np.count_nonzero(valid)))
        self.assertFalse(result["live_qualified"])

    def test_ignore_pixels_never_count_as_background_or_false_positives(self):
        inputs = next(iter(self.fixtures.values()))[0]
        built = A.build_complete_v010(inputs)
        target = built["segmentation"]
        self.assertGreater(np.count_nonzero(target == 255), 0)
        first = np.where(target == 255, 0, target).astype(np.uint8)
        second = first.copy()
        second[target == 255] = 2
        a, b = A.score_complete_v010(built, first), A.score_complete_v010(built, second)
        self.assertEqual(a, b)
        self.assertEqual(a["original_score"]["person_box_mask_iou"], 1.0)
        self.assertEqual(a["original_score"]["vehicle_iou"], 1.0)

    def test_v3_only_mask_cannot_enter_complete_scorer(self):
        with self.assertRaisesRegex(A.TargetInputError, "complete v010"):
            A.score_complete_v010({"contract": "ORIGINAL_V3_DEPTH_INTERVAL_PERSON_PAINT_ONLY_NOT_V010"}, np.zeros((1, 1), np.uint8))

    def test_schema_requires_predictions_and_explicit_quarantines(self):
        schema = A.required_collection_inputs()
        self.assertIn("predicted_labels", schema["arrays"])
        self.assertIn("never default to empty", schema["quarantine"]["missing_or_unadjudicated"])
        self.assertEqual(schema["target_grid"], [432, 768])


if __name__ == "__main__":
    unittest.main()
