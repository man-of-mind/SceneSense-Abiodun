"""CPU retained-fixture, original-function and adversarial evaluator tests."""
from dataclasses import replace
import ast
import copy
import csv
import hashlib
import json
import math
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as target
from rl_agent.splitfusion_live_target_parity_v1.fixtures import FIXTURES, FINAL, load_retained_fixture
from rl_agent.splitfusion_run5_live_avo_v1 import quality_adapter as people
from . import evaluation as E


class CompleteEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.prepared, cls.references = {}, {}, {}
        for sid in FIXTURES:
            inputs, references = load_retained_fixture(sid)
            cls.inputs[sid], cls.references[sid] = inputs, references
            cls.prepared[sid] = E.prepare_target(inputs)
        source = E.ROOT / "experiments/splitfusion_fcos_service_candidate_v1/predictions/detections.csv"
        if target.sha256(source) != "a682a1fc5eabb2e59e07449a8c6b5fc604077b40ef094b57dc30c5a18d7ec260":
            raise RuntimeError("retained reference predictions changed")
        cls.records = {sid: [] for sid in FIXTURES}
        with source.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["sample_id"] in FIXTURES:
                    cls.records[row["sample_id"]].append(dict(row))
        table = E.ROOT / "experiments/actor_volume_observability_model_comparison_v1/20260901_repaired_tolerance_cpu_once/actor_volume_observability_table.csv"
        if target.sha256(table) != "abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386":
            raise RuntimeError("retained AVO table changed")
        cls.avo = {sid: {"frame_id": cls.prepared[sid].frame_id,
                        "qualified": [], "structural_ignored": []} for sid in FIXTURES}
        with table.open(newline="") as stream:
            for row in csv.DictReader(stream):
                sid = row["sample_id"]
                if sid in FIXTURES:
                    cls.avo[sid]["qualified"].append({
                        "frame_id": cls.prepared[sid].frame_id, "gt_actor_id": row["gt_actor_id"],
                        "world_x": float(row["object_world_x"]), "world_y": float(row["object_world_y"]),
                        "avo": float(row["actor_volume_observability"]), "qualified": True,
                        "distance_m": float(row["distance_m"]), "distance_bin": row["distance_bin"],
                        "no_support": row["no_support"].lower() == "true"})
        for sid, inputs in cls.inputs.items():
            qualified_ids = {r["gt_actor_id"] for r in cls.avo[sid]["qualified"]}
            for row in inputs.actor_boxes:
                if row["label"] == "person" and row["gt_actor_id"] not in qualified_ids:
                    cls.avo[sid]["structural_ignored"].append({"frame_id": cls.prepared[sid].frame_id,
                        "gt_actor_id": row["gt_actor_id"], "qualified": False,
                        "world_x": float(row["object_world_x"]), "world_y": float(row["object_world_y"])})
        from pole_lraspp_multimodal_fusion.object_head_pilot_v1.route_b_v3_1_targeted_refinement_v1 import audit_v1
        cls.original = audit_v1
        cls.reference_gt, _ = audit_v1.load_gt(E.ROOT / FINAL.split("/contracts/")[0], "v010")

    def inputs_for(self, sid=None):
        sid = sid or next(iter(FIXTURES))
        labels = np.zeros((720, 1280), np.uint8)
        labels[100:500, 250:600] = 1
        labels[150:400, 700:850] = 2
        return dict(identity={"frame_id": self.prepared[sid].frame_id,
                              "session_uuid": "00000000-0000-4000-8000-000000000001",
                              "decision_seq": 0, "tensor_seq": 1, "ticket_seq": 0},
                    prepared=self.prepared[sid], predicted_labels=labels,
                    predictions=self.records[sid], avo=self.avo[sid])

    def test_retained_complete_targets_and_original_vehicle_scorer_match(self):
        for sid in FIXTURES:
            with self.subTest(sample=sid):
                prepared = self.prepared[sid]
                self.assertEqual(prepared.target["segmentation_png"], self.references[sid]["segmentation"].read_bytes())
                self.assertEqual(prepared.target["object_ignore_png"], self.references[sid]["object_ignore"].read_bytes())
                result = E.evaluate_prepared(**self.inputs_for(sid))
                rows = [{"class_name": r["class_name"], **{k: float(r[k]) for k in self.original.PREDICTION_FIELDS}}
                        for r in self.records[sid]]
                rows.sort(key=lambda r: (-r["score"], r["class_name"]))
                expected = self.original.score_arm(experiment=E.ROOT / FINAL.split("/contracts/")[0],
                    contract="v010", frame_ids=[sid], predictions={sid: rows},
                    gt={sid: self.reference_gt[sid]}, threshold=.20,
                    ignore_cache={sid: cv2.imread(str(self.references[sid]["object_ignore"]), cv2.IMREAD_UNCHANGED)})
                self.assertEqual(result.evidence["scoring"]["vehicle_detection"], expected["classes"]["vehicle"])
                self.assertEqual(result.measurement["loc_vehicle_tp"], len(result.evidence["matched_pairs"]["vehicle"]))

    def test_segmentation_confusion_exact_prediction_resize_ignore255(self):
        args = self.inputs_for()
        result = E.evaluate_prepared(**args)
        target_mask = args["prepared"].target["segmentation"]
        resized = cv2.resize(args["predicted_labels"], (768, 432), interpolation=cv2.INTER_NEAREST)
        valid = target_mask != 255
        expected = np.bincount(target_mask[valid].astype(np.int64) * 3 + resized[valid], minlength=9).reshape(3, 3)
        self.assertEqual(result.evidence["scoring"]["segmentation"]["confusion_matrix"], expected.tolist())
        self.assertGreater(int((~valid).sum()), 0)

    def test_person_avo_is_exact_existing_function(self):
        from data_collection.route_b_publication_actor_volume_observability_model_comparison_v1 import run_comparison as original
        for sid in FIXTURES:
            args = self.inputs_for(sid)
            result = E.evaluate_prepared(**args)
            expected = people.person_avo_measurement(args["predictions"], args["avo"])
            self.assertTrue(all(result.measurement[k] == v for k, v in expected.items()))
            # Independent original full AVO scorer over retained geometry/predictions.
            pred = [{"score": float(r["score"]), "world_x": float(r["world_x"]), "world_y": float(r["world_y"])}
                    for r in args["predictions"] if r["class_name"] == "person"]
            scored = original.score_person_view(frame_ids=[sid], predictions={sid: pred},
                qualified_gt={sid: args["avo"]["qualified"]},
                structural_ignored_gt={sid: args["avo"]["structural_ignored"]},
                episode_by_sample={sid: self.inputs[sid].frame["experiment_id"]},
                avo_threshold=.65, detection_threshold=.25)["overall"]
            for k in ("tp", "fp", "fn", "ignored_predictions"):
                self.assertEqual(result.measurement[f"loc_person_{k}"], scored[k])

    def test_vehicle_threshold_inclusive_and_low_scores_not_false_positives(self):
        args = self.inputs_for()
        row = dict(args["predictions"][0], class_name="vehicle", world_x=10000., world_y=10000.,
                   center_x_px=-100., center_y_px=-100.)
        a = E.evaluate_prepared(**dict(args, predictions=[dict(row, score=.199999)]))
        b = E.evaluate_prepared(**dict(args, predictions=[dict(row, score=.20)]))
        self.assertEqual(a.measurement["loc_vehicle_fp"], 0)
        self.assertEqual(b.measurement["loc_vehicle_fp"], 1)

    def test_vehicle_ignore_is_neutral_not_false_positive(self):
        args = self.inputs_for()
        yy, xx = np.argwhere(args["prepared"].target["object_ignore"] != 0)[0]
        row = dict(args["predictions"][0], class_name="vehicle", world_x=10000., world_y=10000.,
                   center_x_px=float(xx), center_y_px=float(yy), score=.9)
        result = E.evaluate_prepared(**dict(args, predictions=[row]))
        self.assertEqual(result.measurement["loc_vehicle_fp"], 0)
        self.assertEqual(result.measurement["loc_vehicle_ignored_predictions"], 1)

    def test_missing_inputs_foreign_identity_and_malformed_prediction_refused(self):
        args = self.inputs_for()
        for identity in ({}, dict(args["identity"], frame_id=99)):
            with self.assertRaises(E.EvaluationInputError):
                E.evaluate_prepared(**dict(args, identity=identity))
        with self.assertRaises(E.EvaluationInputError):
            E.evaluate_prepared(**dict(args, avo={"frame_id": args["identity"]["frame_id"]}))
        row = dict(args["predictions"][0]); row.pop("center_x_px")
        with self.assertRaises(E.EvaluationInputError):
            E.evaluate_prepared(**dict(args, predictions=[row]))
        incomplete = replace(self.inputs[next(iter(FIXTURES))], quarantine={})
        with self.assertRaises(target.TargetInputError):
            E.prepare_target(incomplete)

    def test_tampered_target_refused(self):
        prepared = next(iter(self.prepared.values()))
        changed = dict(prepared.target)
        changed["segmentation"] = changed["segmentation"].copy()
        changed["segmentation"][0, 0] ^= 1
        with self.assertRaises(E.EvaluationInputError):
            E.restore_prepared_target(frame_id=prepared.frame_id, sample_id=prepared.sample_id,
                target=changed, content_sha256=prepared.content_sha256)

    def test_foreign_table_frame_refused_even_with_recomputed_content_digest(self):
        prepared = next(iter(self.prepared.values()))
        for name, rows in prepared.target["tables"].items():
            if not rows or name == "target_manifest.csv":
                continue
            changed = dict(prepared.target, tables=copy.deepcopy(prepared.target["tables"]))
            changed["tables"][name][0]["frame_id"] = str(prepared.frame_id + 1)
            with self.subTest(table=name), self.assertRaisesRegex(E.EvaluationInputError, "foreign frame"):
                E.restore_prepared_target(frame_id=prepared.frame_id, sample_id=prepared.sample_id,
                    target=changed, content_sha256=E._target_digest(prepared.frame_id, prepared.sample_id, changed))

    def test_actual_v3_live_serializer_and_snapshot_feed_complete_evaluator(self):
        # No network, CUDA, model construction or inference. The real record
        # producer and serializer consume retained model-output values on CPU.
        import torch
        from rl_agent.splitfusion_edge_optimization_v1.detached_tail_v3 import DetachedOptimizedTailAdapterV3
        from rl_agent.splitfusion_edge_optimization_v1.test_detached_tail import work, _Ledger
        infer_path = E.ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_v1/infer.py"
        self.assertEqual(target.sha256(infer_path), "b34308f9ad6fa6524511c2d5e43fab335d5d4c1024914e7c2f5719a5950b3194")
        source = ast.parse(infer_path.read_text())
        record = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "record")
        fields = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "FIELDS" for t in n.targets))
        ns = {"torch": torch, "math": math, "Any": object,
              "CLASS_NAMES": ("vehicle", "person"), "LEVELS": ("p2", "p3", "p4", "p5", "p6", "p7")}
        exec(compile(ast.Module(body=[record], type_ignores=[]), str(infer_path), "exec"), ns)
        args = self.inputs_for()
        rows = args["predictions"]
        def values(names, dtype=torch.float64):
            return torch.tensor([[float(r[n]) for n in names] for r in rows], dtype=dtype)
        tensors = {"boxes": values(("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")),
            "labels_internal": values(("internal_class",), torch.long).flatten(),
            "local_xyz": values(("local_x", "local_y", "local_z")),
            "world_xyz": values(("world_x", "world_y", "world_z")),
            "dimensions": values(("size_x", "size_y", "size_z")),
            "yaw": values(("yaw_sin", "yaw_cos")),
            "physical_uv": values(("physical_ray_x_px", "physical_ray_y_px")),
            "candidate_identity": torch.tensor([[int(v) for v in r["candidate_identity"].split(":")] for r in rows]),
            "scores": values(("score",)).flatten(), "depth": values(("actor_forward_depth_m",)).flatten(),
            "depth_bin": values(("depth_bin",), torch.long).flatten(),
            "depth_residual": values(("depth_residual",)).flatten()}
        basework = work()
        indices = values(("prediction_index",), torch.long).flatten()
        product = replace(basework, perception=tensors, perception_cpu=tensors,
            original_indices=indices, original_indices_cpu=indices,
            frame_context=replace(basework.frame_context, frame_id=args["identity"]["frame_id"]))
        adapter = object.__new__(DetachedOptimizedTailAdapterV3)
        adapter._publication_call_lock = threading.Lock()
        adapter._ledger = _Ledger()
        adapter._base = SimpleNamespace(infer=SimpleNamespace(record=ns["record"]))
        adapter._record_fields = fields
        snapshot = adapter.snapshot(adapter.serialize_product(product))
        self.assertEqual(len(snapshot.records), len(rows))
        required = set(E._detection_code()[1]["PREDICTION_FIELDS"]) | {"class_name", "frame_id"}
        self.assertTrue(all(required <= set(r) for r in snapshot.records))
        actual = E.evaluate_prepared(**dict(args, predictions=snapshot.records))
        expected = E.evaluate_prepared(**args)
        self.assertEqual(actual.measurement, expected.measurement)
        self.assertEqual(actual.evidence["matched_pairs"], expected.evidence["matched_pairs"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_complete_prediction_evidence_owns_data_and_encoding_is_deferred(self):
        args = self.inputs_for()
        with patch.object(E.cv2, "imencode", side_effect=AssertionError("archive must occur after feedback")):
            result = E.evaluate_prepared(**args)
        self.assertEqual(result.evidence["prediction_records"], args["predictions"])
        args["predicted_labels"][:] = 2
        self.assertFalse(result.predicted_labels.flags.writeable)
        self.assertFalse(np.array_equal(result.predicted_labels, args["predicted_labels"]))
        payloads = E.archive_payloads(result)
        record = json.loads(payloads["evaluation.json"])
        self.assertEqual(record["predicted_labels"]["png_sha256"], hashlib.sha256(payloads["predicted_labels.png"]).hexdigest())
        np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(payloads["predicted_labels.png"], np.uint8), cv2.IMREAD_UNCHANGED), result.predicted_labels)

    def test_measurement_composes_with_authoritative_quality_without_formula_change(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
        spec = W.load_run4_quality_spec(E.ROOT)
        args = self.inputs_for()
        result = E.evaluate_prepared(**args)
        value = Q.evaluate_exact_quality(spec, result.measurement)
        self.assertIsNotNone(value)
        reference = dict(result.measurement)
        reference.update(people.person_avo_measurement(args["predictions"], args["avo"]))
        self.assertEqual(value, Q.evaluate_exact_quality(spec, reference))


if __name__ == "__main__":
    unittest.main()
