"""CPU production-sized predictions through real ownership, evaluator and reader."""
import ast
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import FunctionType, SimpleNamespace
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import prediction_archive as P, readiness as R
from rl_agent.splitfusion_live_parity_repair_v1 import prediction_archive as OLD, readiness as OLD_R
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation, target_transport, runtime as old_runtime
from rl_agent.splitfusion_live_parity_repair_v1 import test_readiness as retained_tests
from rl_agent.splitfusion_live_parity_repair_v1.test_prediction_archive import ticket

ROOT = Path(__file__).resolve().parents[2]


def production_labels():
    """Execute real production interpolation/argmax/cast; no model or CUDA."""
    from rl_agent.splitfusion_live_dispatch_v1.frame_context import StaticCameraRegistry
    contract = P.native_prediction_contract()
    spec = StaticCameraRegistry.audited().resolve(contract.camera_model_sha256, contract.camera_mount_sha256)
    path = ROOT / "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v3.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "89109f0b7a3382e69d690124a2a0456a0dcd42b7e9b5f7a96a75c57724236d87":
        raise ValueError("production native-label source changed")
    finish = next(node for node in ast.walk(ast.parse(path.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == "_finish")
    names = {"source_hw", "semantic_logits", "semantic_labels"}
    nodes = [node for node in finish.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    if len(nodes) != 3:
        raise ValueError("native production assignment API changed")
    logits = torch.zeros((1, 3, 16, 16))
    logits[:, 1, 2:8, 2:8] = 3.
    logits[:, 2, 10:14, 10:14] = 3.
    scope = {"static": {"spec": spec}, "outputs": {"semantic_logits": logits}, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    path = ROOT / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_edge_runtime_v2.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "8446eaa50ec881028ff66e0718060410edf1c066b1f1589fc985457eb1111cd6":
        raise ValueError("production edge cast source changed")
    compute = next(node for node in ast.walk(ast.parse(path.read_text()))
                   if isinstance(node, ast.FunctionDef) and node.name == "compute")
    cast = next(node for node in compute.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "labels" for target in node.targets))
    target = {"snapshot": SimpleNamespace(semantic_labels=scope["semantic_labels"]), "torch": torch}
    exec(compile(ast.Module(body=[cast], type_ignores=[]), str(path), "exec"), target)
    return target["labels"]


def native_ticket():
    return replace(ticket(), records=(), predicted_mask=production_labels())


def new_frame(value):
    frame = value.context.frame_id + 1
    return replace(value, envelope=replace(value.envelope, frame_id=frame),
                   context=replace(value.context, frame_id=frame), gt_identity=dict(value.gt_identity, frame_id=frame))


def rebound(function, **values):
    result = FunctionType(function.__code__, dict(function.__globals__, **values),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class NativeWriterTests(unittest.TestCase):
    def setUp(self):
        self.value = native_ticket()

    def test_real_production_shape_previously_rejected_now_owned_without_transform(self):
        self.assertEqual(self.value.predicted_mask.shape, (720, 1280))
        self.assertEqual(set(np.unique(self.value.predicted_mask)), {0, 1, 2})
        with self.assertRaisesRegex(ValueError, "invalid dense prediction"):
            OLD.PredictionArchive(max_frames=100).add(self.value)
        archive = P.PredictionArchive(max_frames=100)
        archive.add(self.value)
        owned = archive.owned_ticket(self.value)
        np.testing.assert_array_equal(owned.predicted_mask, self.value.predicted_mask)
        self.assertFalse(owned.predicted_mask.flags.writeable)
        before = owned.predicted_mask.copy()
        self.value.predicted_mask[:] = 2
        np.testing.assert_array_equal(owned.predicted_mask, before)
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_dtype_class_batch_transpose_and_model_grid_mismatches_refused(self):
        invalid = [self.value.predicted_mask.astype(np.int64), np.zeros((1, 720, 1280), np.uint8),
                   np.zeros((432, 768), np.uint8), np.zeros((448, 768), np.uint8),
                   np.zeros((1280, 720), np.uint8), np.zeros((0, 1280), np.uint8)]
        for label in (3, 255):
            value = self.value.predicted_mask.copy()
            value[0, 0] = label
            invalid.append(value)
        for labels in invalid:
            with self.subTest(shape=labels.shape, dtype=labels.dtype), self.assertRaisesRegex(ValueError, "invalid dense prediction"):
                P.PredictionArchive(max_frames=1).add(replace(self.value, predicted_mask=labels))

    def test_ticket_camera_and_frame_identity_mismatches_refused(self):
        for context in (replace(self.value.context, camera_model_sha256="a" * 64),
                        replace(self.value.context, camera_mount_sha256="b" * 64),
                        replace(self.value.context, frame_id=self.value.context.frame_id + 1)):
            with self.subTest(context=context), self.assertRaises(ValueError):
                P.PredictionArchive(max_frames=1).add(replace(self.value, context=context))

    def test_explicit_synthetic_contract_only_and_invalid_contracts_refused(self):
        small = replace(self.value, predicted_mask=np.zeros((3, 5), np.uint8))
        with self.assertRaises(ValueError):
            P.PredictionArchive(max_frames=1).add(small)
        contract = P.PredictionContract.cpu_fixture((3, 5))
        P.PredictionArchive(max_frames=1, prediction_contract=contract).add(small)
        self.assertEqual(contract.document()["kind"], "CPU_ONLY_SYNTHETIC_FIXTURE")
        with self.assertRaises(ValueError):
            P.PredictionContract("NATIVE_CALIBRATED", 3, 5,
                self.value.context.camera_model_sha256, self.value.context.camera_mount_sha256)
        for shape in ((0, 5), (3, 1281), (True, 5), [3, 5]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                P.PredictionContract.cpu_fixture(shape)

    def test_100_and_1000_frame_memory_bounds_are_explicit_and_lazy(self):
        for frames, budget, raw in ((100, 512_000_000, 92_160_000), (1000, 2_000_000_000, 921_600_000)):
            archive = P.PredictionArchive(max_frames=frames)
            self.assertEqual((archive._bytes, archive._items), (0, {}))
            self.assertEqual(archive.max_bytes, budget)
            document = P.budget_document(frames, archive.max_bytes, archive.prediction_contract)
            self.assertEqual(document["raw_label_bytes_at_frame_bound"], raw)
            self.assertLess(raw, budget)
        for options in ({"max_frames": 0}, {"max_frames": 1001}, {"max_frames": True},
                        {"max_frames": 100, "max_bytes": 2_000_000_001}, {"max_frames": 1, "max_bytes": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                P.PredictionArchive(**options)

    def test_capacity_and_duplicate_conflicts_never_replace_owned_prediction(self):
        archive = P.PredictionArchive(max_frames=1)
        key = archive.add(self.value)
        self.assertEqual(archive.add(self.value), key)
        initial_bytes = archive._bytes
        for value in (new_frame(self.value), replace(self.value, predicted_mask=np.zeros((720, 1280), np.uint8))):
            with self.assertRaises(ValueError):
                archive.add(value)
            self.assertEqual(archive._bytes, initial_bytes)
            np.testing.assert_array_equal(archive.owned_ticket(self.value).predicted_mask, self.value.predicted_mask)

    def test_byte_capacity_and_evaluation_capacity_are_explicit_atomic_failures(self):
        archive = P.PredictionArchive(max_frames=100, max_bytes=1)
        with self.assertRaisesRegex(ValueError, "capacity"):
            archive.add(self.value)
        self.assertEqual((archive._bytes, len(archive._items)), (0, 0))
        cost = self.value.predicted_mask.nbytes + len(P.canonical(list(self.value.records))) + len(P.canonical(P.PredictionArchive.identity(self.value)))
        archive = P.PredictionArchive(max_frames=100, max_bytes=cost)
        key = archive.add(self.value)
        with self.assertRaisesRegex(ValueError, "memory bound"):
            archive.evaluated(self.value, {"cannot_fit": True})
        self.assertEqual(archive._bytes, cost)
        self.assertIsNone(archive._items[key]["evaluation"])

    def test_preserved_inherited_ownership_and_identity_methods(self):
        for method in ("identity", "key", "owned_ticket", "evaluated"):
            self.assertIs(getattr(P.PredictionArchive, method), getattr(OLD.PredictionArchive, method))


class NativeReaderTests(retained_tests.ReadinessTests):
    """All 15 old corruption/source/identity cases, with actual native labels."""
    def setUp(self):
        retained_tests.ReadinessTests.setUp(self)
        self.ticket = replace(self.ticket, predicted_mask=production_labels())

    build = rebound(retained_tests.ReadinessTests.build, PredictionArchive=P.PredictionArchive)

    def test_v1_schema_and_small_fixture_manifest_refused_by_native_default(self):
        args = self.build()
        manifest = json.loads((self.archive_root / "MANIFEST.json").read_text())
        for update in ({"schema": OLD.SCHEMA},
                       {"prediction_contract": P.PredictionContract.cpu_fixture((3, 5)).document()}):
            changed = {**manifest, **update}
            (self.archive_root / "MANIFEST.json").write_bytes(P.canonical(changed))
            self.reseal()
            with self.subTest(update=update), self.assertRaises(R.ReadinessError):
                R.verify_offline_evidence(**args)

    def test_reader_rejects_resealed_shape_class_dtype_and_bound_corruption(self):
        args = self.build()
        path = self.archive_root / (self.key + ".labels.npy")
        for labels in (np.zeros((432, 768), np.uint8), np.zeros((720, 1280), np.int64),
                       np.full((720, 1280), 255, np.uint8)):
            buffer = io.BytesIO()
            np.save(buffer, labels, allow_pickle=False)
            path.write_bytes(buffer.getvalue())
            self.reseal(path.name)
            with self.subTest(shape=labels.shape, dtype=labels.dtype), self.assertRaisesRegex(R.ReadinessError, "invalid owned prediction labels"):
                R.verify_offline_evidence(**args)

    def test_memory_budget_and_owned_byte_accounting_are_verified_not_trusted(self):
        args = self.build()
        original = json.loads((self.archive_root / "MANIFEST.json").read_text())
        changes = [lambda value: value["memory_budget"].update(max_bytes=2_000_000_001),
                   lambda value: value["memory_budget"].update(max_frames=0),
                   lambda value: value["memory_budget"].update(raw_label_bytes_per_frame=1),
                   lambda value: value.update(owned_bytes=value["owned_bytes"] + 1)]
        for mutate in changes:
            value = json.loads(json.dumps(original))
            mutate(value)
            (self.archive_root / "MANIFEST.json").write_bytes(P.canonical(value))
            self.reseal()
            with self.subTest(mutate=mutate), self.assertRaises(R.ReadinessError):
                R.verify_offline_evidence(**args)

    def test_tiny_fixture_reader_needs_explicit_matching_contract(self):
        contract = P.PredictionContract.cpu_fixture((3, 5))
        self.ticket = replace(self.ticket, predicted_mask=np.zeros(contract.shape, np.uint8))
        factory = lambda **kw: P.PredictionArchive(**kw, prediction_contract=contract)
        args = rebound(retained_tests.ReadinessTests.build, PredictionArchive=factory)(self)
        with self.assertRaisesRegex(R.ReadinessError, "prediction contract"):
            R.verify_offline_evidence(**args)
        result = R.verify_offline_evidence(**args, expected_prediction_contract=contract)
        self.assertEqual(result["scored_predictions"], 1)
        self.assertEqual(result["prediction_contract"]["kind"], "CPU_ONLY_SYNTHETIC_FIXTURE")
        self.assertFalse(result["live_qualified"])

    def test_native_scoring_preserves_original_prediction_resize(self):
        result = evaluation.evaluate_prepared(identity=self.identity["action_identity"], prepared=self.prepared,
            predicted_labels=self.ticket.predicted_mask, predictions=[],
            avo={"frame_id": self.gt["frame_id"], "qualified": [], "structural_ignored": []})
        resized = cv2.resize(self.ticket.predicted_mask, (768, 432), interpolation=cv2.INTER_NEAREST)
        truth = self.prepared.target["segmentation"]
        valid = truth != 255
        expected = np.bincount(truth[valid].astype(np.int64) * 3 + resized[valid], minlength=9).reshape(3, 3)
        self.assertEqual(result.evidence["scoring"]["segmentation"]["confusion_matrix"], expected.tolist())
        np.testing.assert_array_equal(result.predicted_labels, self.ticket.predicted_mask)


# Clone only each retained test function's globals; never modify v1 test/module
# globals. The old source assertions execute unchanged against the new reader.
for _name, _function in vars(retained_tests.ReadinessTests).items():
    if _name.startswith("test_"):
        setattr(NativeReaderTests, _name, rebound(_function, R=R))


class RealEvaluatorTests(unittest.TestCase):
    """Real retained async evaluator with newly scoped archive class."""
    setUp = NativeReaderTests.setUp
    manifest_digest = retained_tests.ReadinessTests.manifest_digest

    def test_native_archive_evaluate_close_reader_full_chain_and_restoration(self):
        from . import runtime
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as edge
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as wire
        from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as ground_truth
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt
        written = target_transport.write_bundle(self.target_root, identity=self.gt, inputs=self.inputs, prepared=self.prepared)
        avo = dict(frame_id=self.gt["frame_id"], qualified=[], structural_ignored=[], threshold=.65)
        live_gt.sidecar_path(self.target_root, self.gt).write_bytes(P.canonical(dict(
            schema=live_gt.SCHEMA, mask_contract=live_gt.MASK_CONTRACT, identity=self.gt,
            source_frame_id=self.gt["frame_id"], avo=avo, ready_wall_ns=time.time_ns())))
        expected = evaluation.evaluate_prepared(identity=self.identity["action_identity"], prepared=self.prepared,
            predicted_labels=self.ticket.predicted_mask, predictions=[], avo=avo)
        spec = wire.load_run4_quality_spec(ROOT)
        feedback, reason = wire.quality_feedback(spec, expected.measurement, self.ticket.envelope)
        sent, old = [], old_runtime.PredictionArchive
        with runtime.install_prediction_contract(), old_runtime.install_edge_evaluation(
                archive_directory=self.archive_root, max_frames=100):
            worker = edge.Run4EvaluatorV2(spec=spec, send=sent.append,
                read_ground_truth=lambda **kwargs: ground_truth.read_ground_truth(self.target_root, **kwargs),
                match_distance_m=3., gt_timeout_s=1., probe_timeout_s=.001)
            worker.start()
            try:
                worker.submit(self.ticket)
            finally:
                result = worker.close()
        self.assertIs(old_runtime.PredictionArchive, old)
        self.assertFalse(result["worker_alive"])
        self.assertTrue(result["parity_archive_complete"])
        self.assertEqual(result["parity_archive_errors"], [])
        self.assertEqual((result["submitted"], result["emitted"]), (1, 1))
        self.assertEqual(sent, [wire.encode_feedback(feedback, reason)])
        self.assertEqual(result["parity_archive"]["prediction_contract"], P.native_prediction_contract().document())
        args = dict(target_directory=self.target_root,
            target_descriptors=[{"identity": self.gt, "manifest_sha256": written["manifest_sha256"]}],
            archive_directory=self.archive_root, expected_archive_manifest_sha256=self.manifest_digest(),
            evaluator_result=result, expected_ticket_identities=[self.identity])
        checked = R.verify_offline_evidence(**args)
        self.assertEqual(checked["scored_predictions"], 1)
        key = P.PredictionArchive.key(self.identity)
        labels = np.load(self.archive_root / (key + ".labels.npy"), allow_pickle=False)
        np.testing.assert_array_equal(labels, self.ticket.predicted_mask)
        self.assertEqual(labels.shape, (720, 1280))
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_queue_backpressure_retains_both_native_predictions_and_emits_fault(self):
        from . import runtime
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as edge
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as wire
        from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as ground_truth
        second = new_frame(self.ticket)
        sent = []
        with runtime.install_prediction_contract(), old_runtime.install_edge_evaluation(
                archive_directory=self.archive_root, max_frames=100):
            worker = edge.Run4EvaluatorV2(spec=wire.load_run4_quality_spec(ROOT), send=sent.append,
                read_ground_truth=lambda **kwargs: ground_truth.read_ground_truth(self.target_root, **kwargs),
                match_distance_m=3., gt_timeout_s=.005, probe_timeout_s=.001, queue_depth=1)
            # Deterministically fill the real queue before starting its worker.
            worker.submit(self.ticket)
            worker.submit(second)
            self.assertEqual(worker.counters["queue_overflow"], 1)
            self.assertEqual(len(sent), 1)
            worker.start()
            result = worker.close()
        self.assertFalse(result["worker_alive"])
        self.assertEqual((result["submitted"], result["emitted"], result["queue_overflow"]), (1, 2, 1))
        self.assertEqual(result["parity_archive"]["frames"], 2)
        self.assertEqual(result["parity_archive"]["evaluated_frames"], 0)
        self.assertTrue(result["parity_archive_complete"])
        checked = R.verify_offline_evidence(target_directory=self.target_root, target_descriptors=[],
            archive_directory=self.archive_root, expected_archive_manifest_sha256=self.manifest_digest(),
            evaluator_result=result, expected_ticket_identities=[self.identity, P.PredictionArchive.identity(second)])
        self.assertEqual(checked["unscored_predictions"], 2)
        self.assertEqual(checked["missing_complete_targets_for_archived_predictions"], 2)
        self.assertFalse(checked["live_qualified"])


if __name__ == "__main__":
    unittest.main()
