"""Independent CPU-only adversarial checks of the additive GT optimizations."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as retained
from rl_agent.splitfusion_live_parity_repair_v1 import test_source_geometry as fixtures
from rl_agent.splitfusion_live_target_parity_v1 import adapter as adapter
from . import source_geometry as candidate
from . import target_builder as target_candidate
from . import runtime as optimized_runtime


class IndependentSourceReviewTests(unittest.TestCase):
    def test_nearest_preserves_nonbinary_values_and_strided_views(self):
        original = retained._original()[3]._nearest_model_count
        random = np.random.default_rng(1743)
        labels = random.choice(np.array([0, 1, 2, 127, 255], np.uint8), (720, 1280))
        for mask in (labels, labels[::-1], labels[:, ::-1], np.asfortranarray(labels),
                     labels != 0, (labels != 0)[::-1, ::-1]):
            with self.subTest(dtype=str(mask.dtype), strides=mask.strides):
                before = mask.copy()
                self.assertEqual(candidate._nearest_model_count(mask, 768, 432),
                                 original(mask, 768, 432))
                np.testing.assert_array_equal(mask, before)

    def test_nonproduction_geometry_and_dtype_use_original_semantics(self):
        original = retained._original()[3]._nearest_model_count
        for shape, dimensions, dtype in (((9, 13), (7, 5), np.uint8),
                                        ((720, 1280), (768, 432), np.int16),
                                        ((720, 1280), (31, 19), np.bool_)):
            mask = np.arange(np.prod(shape)).reshape(shape) % 3
            mask = mask.astype(dtype)
            self.assertEqual(candidate._nearest_model_count(mask, *dimensions),
                             original(mask, *dimensions))

    def test_original_source_method_code_is_preserved(self):
        for name in ("__init__", "build"):
            self.assertIs(getattr(candidate.OptimizedSourceProducer, name).__code__,
                          getattr(candidate.BASE_PRODUCER, name).__code__)
        self.assertIs(candidate.OptimizedSourceProducer.register_population,
                      candidate.BASE_PRODUCER.register_population)

    def test_private_visibility_and_static_global_chains_do_not_alias(self):
        old = retained._original()
        first, second = candidate._original_private(), candidate._original_private()
        for index in (2, 3):
            self.assertIsNot(first[index], second[index])
            self.assertIsNot(first[index], old[index])
        self.assertIs(first[2]["visibility_from_corners"], first[3].visibility_from_corners)
        self.assertIs(second[2]["visibility_from_corners"], second[3].visibility_from_corners)
        self.assertIs(first[2]["audit_visibility"].__globals__, first[2])
        self.assertIs(second[2]["audit_visibility"].__globals__, second[2])
        fg = first[3].visibility_from_corners.__globals__
        sg = second[3].visibility_from_corners.__globals__
        self.assertIsNot(fg, sg)
        self.assertIsNot(fg, vars(old[3]))
        fg["_independent_review_marker"] = object()
        first[2]["_independent_review_marker"] = object()
        self.assertNotIn("_independent_review_marker", sg)
        self.assertNotIn("_independent_review_marker", second[2])
        self.assertNotIn("_independent_review_marker", old[2])
        self.assertNotIn("_independent_review_marker", vars(old[3]))

    def test_producers_have_private_history_and_equal_first_frame_bytes(self):
        fixture = fixtures.SourceGeometryTests()
        # Retain every evidence field, controlling only the observation clock
        # so two sequential executions can have byte-identical provenance.
        with fixture.fixture() as (directory, kwargs), \
                mock.patch("time.clock_gettime_ns", return_value=123456789):
            args = dict(static_catalogue_dir=directory / "catalogue", experiment_id="independent_cpu",
                        ego_id=999, dynamic_population_ids=[1, 2],
                        population_evidence_sha256="d" * 64, max_frames=1000)
            original = candidate.BASE_PRODUCER(**args)
            first = candidate.OptimizedSourceProducer(**args)
            second = candidate.OptimizedSourceProducer(**args)
            self.assertIsNot(first.tracker, second.tracker)
            self.assertIsNot(first.conflict_state, second.conflict_state)
            self.assertIsNot(first.conflict_state.pairs, second.conflict_state.pairs)
            expected = original.build(**kwargs)
            actual = first.build(**kwargs)
            self.assertEqual(second.processed_frames, 0)
            self.assertEqual(second.conflict_state.frame_count, 0)
            self.assertIsNone(second.last_evidence)
            for name in ("depth_bgra", "raw_semantic", "original_v3_mask"):
                np.testing.assert_array_equal(getattr(expected, name), getattr(actual, name))
            self.assertEqual(adapter.input_content_sha256(expected), adapter.input_content_sha256(actual))
            self.assertEqual(adapter.canonical(adapter._json_value(original.last_evidence)),
                             adapter.canonical(adapter._json_value(first.last_evidence)))
            first.register_population([1, 2, 3], "e" * 64, 101)
            self.assertIsNone(second.pending_registration)
            rebuilt = second.build(**kwargs)
            self.assertEqual(adapter.input_content_sha256(rebuilt), adapter.input_content_sha256(expected))

    def test_code_prewarm_does_not_construct_or_advance_producer_state(self):
        with mock.patch.object(candidate.BASE_PRODUCER, "__init__",
                               side_effect=AssertionError("producer created in prewarm")), \
                mock.patch.object(candidate.BASE_PRODUCER, "build",
                                  side_effect=AssertionError("frame consumed in prewarm")):
            result = candidate.prewarm()
        self.assertIs(result["state_or_history_created"], False)
        self.assertIs(result["fixed_grid_proved"], True)

    def test_cached_prewarm_rechecks_actual_pinned_source_bytes(self):
        candidate.prewarm()
        with tempfile.TemporaryDirectory(prefix="independent-source-pin-") as scratch:
            root = Path(scratch)
            for relative, _digest in {**adapter.SOURCES, **retained.SOURCES}.values():
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(adapter.ROOT / relative, destination)
            builder = root / adapter.SOURCES["builder"][0]
            with builder.open("ab") as stream:
                stream.write(b"\n# independent corruption fixture\n")
            with mock.patch.object(retained, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    candidate.prewarm()

    def test_scope_rejects_nested_or_foreign_install_and_restores_on_error(self):
        original = retained.LiveCompleteTargetProducer
        with self.assertRaisesRegex(RuntimeError, "caller failure"):
            with candidate.install():
                self.assertIs(retained.LiveCompleteTargetProducer, candidate.OptimizedSourceProducer)
                with self.assertRaisesRegex(RuntimeError, "unexpected source producer seam"):
                    with candidate.install():
                        self.fail("nested installation unexpectedly admitted")
                raise RuntimeError("caller failure")
        self.assertIs(retained.LiveCompleteTargetProducer, original)


class IndependentTargetReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.SourceGeometryTests().fixture()
        directory, kwargs = cls.fixture.__enter__()
        cls.inputs = []
        for experiment in ("independent_first", "independent_second"):
            producer = candidate.BASE_PRODUCER(static_catalogue_dir=directory / "catalogue",
                experiment_id=experiment, ego_id=999, dynamic_population_ids=[1, 2],
                population_evidence_sha256="d" * 64, max_frames=1000)
            cls.inputs.append(producer.build(**kwargs))
        cls.expected = [adapter.build_complete_v010(value) for value in cls.inputs]

    @classmethod
    def tearDownClass(cls):
        cls.fixture.__exit__(None, None, None)

    def assert_exact_target(self, actual, expected):
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(set(actual), set(expected))

    def test_returned_mutations_cannot_poison_later_frame_or_same_frame(self):
        first = target_candidate.build_complete_v010(self.inputs[0])
        self.assert_exact_target(first, self.expected[0])
        first["segmentation"][:] = 37
        first["object_ignore"][:] = 39
        first["tables"]["target_manifest.csv"][0]["sample_id"] = "foreign"
        first["tables"]["object_boxes.csv"].append({"sample_id": "foreign"})
        first["summary"]["frames"] = -100
        first["sources"].clear()
        for inputs, expected in zip(self.inputs[::-1], self.expected[::-1]):
            self.assert_exact_target(target_candidate.build_complete_v010(inputs), expected)

    def test_compiled_cache_never_shares_module_or_reconstruction_globals(self):
        root, _sources, payloads = target_candidate._verified_sources(adapter.ROOT)
        first = target_candidate._modules(root, payloads)
        second = target_candidate._modules(root, payloads)
        self.assertIsNot(first[0].__dict__, second[0].__dict__)
        self.assertIsNot(first[1].__dict__, second[1].__dict__)
        self.assertIsNot(first[2].roi_mask.__globals__, second[2].roi_mask.__globals__)
        self.assertIs(first[0].build_contract.__code__, second[0].build_contract.__code__)
        self.assertIs(first[2].roi_mask.__globals__["reconstruct_consistent_mask"],
                      first[1].reconstruct_consistent_mask)
        first[0].cv2 = object()
        first[1].MIN_MODEL_VISIBLE_PX = -100
        self.assertIsNot(first[0].cv2, second[0].cv2)
        self.assertNotEqual(second[1].MIN_MODEL_VISIBLE_PX, -100)

    def test_original_global_cv2_and_builder_binding_remain_unchanged(self):
        original_builder = adapter.build_complete_v010
        original_cv2 = adapter.cv2
        first = target_candidate.build_complete_v010(self.inputs[0])
        self.assert_exact_target(first, self.expected[0])
        self.assertIs(adapter.build_complete_v010, original_builder)
        self.assertIs(adapter.cv2, original_cv2)
        with self.assertRaisesRegex(RuntimeError, "inside target scope"):
            with target_candidate.install_target_builder():
                self.assert_exact_target(adapter.build_complete_v010(self.inputs[1]), self.expected[1])
                raise RuntimeError("inside target scope")
        self.assertIs(adapter.build_complete_v010, original_builder)

    def test_png_failure_does_not_return_partial_target_or_poison_retry(self):
        with mock.patch.object(target_candidate.cv2, "imencode", return_value=(False, None)):
            with self.assertRaises(RuntimeError):
                target_candidate.build_complete_v010(self.inputs[0])
        self.assert_exact_target(target_candidate.build_complete_v010(self.inputs[1]), self.expected[1])

    def test_memory_output_refuses_escape_foreign_reads_and_duplicate_writes(self):
        memory = target_candidate._MemoryFiles("bounded_sample", "val")
        for relative in ("../outside", "/outside", "contracts/../../outside"):
            with self.subTest(path=relative), self.assertRaises(adapter.TargetInputError):
                memory.root / relative
        with self.assertRaises(adapter.TargetInputError):
            (memory.root / "foreign.csv").open("x")
        directory = memory.root / "contracts/v010/val"
        directory.mkdir(parents=True)
        path = directory / "object_boxes.csv"
        with self.assertRaises(FileNotFoundError):
            path.open("rb")
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write("a,b\r\n1,2\r\n")
        with path.open("rb") as stream:
            self.assertEqual(stream.read(), b"a,b\r\n1,2\r\n")
        with self.assertRaises(FileExistsError):
            path.open("x")
        with self.assertRaises(adapter.TargetInputError):
            path.open("w")

    def test_source_change_after_warmup_and_during_build_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="independent-target-pin-") as scratch:
            root = Path(scratch)
            for relative, _digest in adapter.SOURCES.values():
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(adapter.ROOT / relative, destination)
            target_candidate.prewarm(root=root)
            path = root / adapter.SOURCES["scorer"][0]
            original = path.read_bytes()
            with path.open("ab") as stream:
                stream.write(b"\n# changed after prewarm\n")
            with self.assertRaisesRegex(adapter.TargetInputError, "source changed"):
                target_candidate.build_complete_v010(self.inputs[0], root=root)
            path.write_bytes(original)
            real_encode = target_candidate.cv2.imencode
            def mutate_after_verification(*args, **kwargs):
                with path.open("ab") as stream:
                    stream.write(b"\n# changed during build\n")
                return real_encode(*args, **kwargs)
            with mock.patch.object(target_candidate.cv2, "imencode", side_effect=mutate_after_verification):
                with self.assertRaisesRegex(adapter.TargetInputError, "source changed"):
                    target_candidate.build_complete_v010(self.inputs[0], root=root)


class IndependentScopeReviewTests(unittest.TestCase):
    def test_prospective_binding_is_owned_not_aliased_to_caller_or_report(self):
        expected = optimized_runtime.source_inventory()
        original = dict(expected)
        with optimized_runtime.install_cpu_gt_optimizations(expected_sources=expected) as state:
            expected.clear()
            state["sources"].clear()
        self.assertIsNone(state["source_integrity_error"])
        self.assertTrue(state["retained_settings_unchanged"])
        self.assertEqual(optimized_runtime.source_inventory(), original)

    def test_failure_path_restores_seams_and_records_integrity_without_masking_exception(self):
        expected = optimized_runtime.source_inventory()
        changed = False
        def inventory():
            return dict(expected, **({"runtime.py": "0" * 64} if changed else {}))
        before_source = retained.LiveCompleteTargetProducer
        before_target = adapter.build_complete_v010
        failure = RuntimeError("original caller failure")
        with mock.patch.object(optimized_runtime, "source_inventory", side_effect=inventory):
            with self.assertRaises(RuntimeError) as captured:
                with optimized_runtime.install_cpu_gt_optimizations(expected_sources=expected) as state:
                    changed = True
                    raise failure
        self.assertIs(captured.exception, failure)
        self.assertIn("source inventory differs", state["source_integrity_error"])
        self.assertIs(retained.LiveCompleteTargetProducer, before_source)
        self.assertIs(adapter.build_complete_v010, before_target)
        self.assertFalse(optimized_runtime._LOCK.locked())


if __name__ == "__main__":
    unittest.main()
