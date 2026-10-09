"""Independent CPU-only equivalence and adversarial checks of the v2 scope."""
from __future__ import annotations

from collections import OrderedDict
import copy
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock

import cv2
import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v1 import runtime as V1
from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S
from rl_agent.splitfusion_live_gt_optimization_v1 import target_builder as V1T
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_live_parity_repair_v1 import test_source_geometry as fixtures
from rl_agent.splitfusion_live_parity_repair_v2 import gt_timing
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from . import candidate as C


def bindings():
    return (S.LiveCompleteTargetProducer, A.build_complete_v010, E.prepare_target,
            E.restore_prepared_target, T.write_bundle, G.write_source_archive)


class IndependentUniqueTests(unittest.TestCase):
    def test_full_uint8_domain_empty_scalar_strided_and_readonly(self):
        values = np.tile(np.arange(256, dtype=np.uint8), (23, 1))
        readonly = values.copy()
        readonly.flags.writeable = False
        samples = (values, values[::-1, ::-1], values[:, ::3],
                   np.asfortranarray(values), readonly,
                   np.empty((0, 8), dtype=np.uint8), np.array(255, dtype=np.uint8))
        for value in samples:
            with self.subTest(shape=value.shape, strides=value.strides):
                before = value.copy()
                actual, expected = C.ExactArrayOperations().unique(value), np.unique(value)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, expected.dtype)
                np.testing.assert_array_equal(value, before)

    def test_every_pixel_counts_and_same_array_mutation_is_not_cached(self):
        operations = C.ExactArrayOperations()
        value = np.zeros((720, 1280), np.uint8)
        for position, label in (((0, 0), 3), ((719, 1279), 254), ((361, 643), 255)):
            value.fill(0)
            value[position] = label
            np.testing.assert_array_equal(operations.unique(value), np.array([0, label], np.uint8))
        value.fill(2)
        np.testing.assert_array_equal(operations.unique(value), np.array([2], np.uint8))

    def test_subclass_and_nondefault_signatures_delegate_to_numpy(self):
        class DispatchArray(np.ndarray):
            def __array_function__(self, function, types, args, kwargs):
                if function is np.unique:
                    return "subclass unique contract"
                return super().__array_function__(function, types, args, kwargs)

        operations = C.ExactArrayOperations()
        self.assertEqual(operations.unique(np.arange(3, dtype=np.uint8).view(DispatchArray)),
                         "subclass unique contract")
        value = np.array([[3, 2, 3], [1, 2, 1]], np.uint8)
        cases = ((value, (), {"return_counts": True}),
                 (value, (True, True, True), {}),
                 (value, (), {"axis": 0}),
                 (value, (), {"equal_nan": False}),
                 (value.astype(np.int16), (), {}),
                 ([3, 2, 3], (), {}))
        for array, args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs, kind=type(array)):
                expected = np.unique(array, *args, **kwargs)
                actual = operations.unique(array, *args, **kwargs)
                if isinstance(expected, tuple):
                    self.assertEqual(len(actual), len(expected))
                    for left, right in zip(actual, expected):
                        np.testing.assert_array_equal(left, right)
                else:
                    np.testing.assert_array_equal(actual, expected)


class IndependentNamespaceTests(unittest.TestCase):
    def test_transitive_namespaces_private_and_original_classes_retained(self):
        first, second = C.private_functions(), C.private_functions()
        for key in ("prepare", "restore", "write", "archive"):
            self.assertIsNot(first[key].__globals__, second[key].__globals__)
        adapter = first["write"].__globals__["targets"]
        self.assertIs(adapter.CompleteTargetInputs, A.CompleteTargetInputs)
        self.assertIs(adapter.TargetInputError, A.TargetInputError)
        self.assertIs(first["restore"].__globals__["PreparedEvaluationTarget"], E.PreparedEvaluationTarget)
        self.assertIs(first["write"].__globals__["_validate_inputs"].__globals__,
                      first["write"].__globals__)
        self.assertIs(first["archive"].__globals__["sha"].__globals__,
                      first["archive"].__globals__)
        self.assertIs(adapter._validate.__globals__["_json_value"], V1S.json_value)
        self.assertIs(first["prepare"].__globals__["restore_prepared_target"], first["restore"])
        first["write"].__globals__["_independent_marker"] = True
        self.assertNotIn("_independent_marker", second["write"].__globals__)
        self.assertNotIn("_independent_marker", vars(T))
        self.assertIs(A._validate.__globals__, vars(A))
        self.assertIs(A._validate.__globals__["np"], np)

    def test_metadata_canonical_bytes_and_error_behavior_match_original(self):
        adapter = C.private_functions()["write"].__globals__["targets"]
        values = (None, True, -0.0, 10 ** 80, "é\n\\",
                  MappingProxyType({1: ("x", {3, 2, 1})}),
                  OrderedDict((("b", [1, 2]), ("a", frozenset(("x", "y"))))),
                  {"numpy": np.float64(3.25)}, {"unsupported": np.int64(1)},
                  {"nan": float("nan")}, {"infinity": float("inf")})
        for value in values:
            with self.subTest(value=repr(value)):
                try:
                    expected = A.canonical(A._json_value(value))
                except (TypeError, ValueError) as error:
                    with self.assertRaises(type(error)):
                        adapter.canonical(adapter._json_value(value))
                else:
                    self.assertEqual(expected, adapter.canonical(adapter._json_value(value)))

    def test_scope_requires_v1_rejects_nesting_and_restores_on_exception(self):
        original = bindings()
        with self.assertRaisesRegex(RuntimeError, "require.*v1"):
            with C.install():
                self.fail("v2 scope admitted without retained v1")
        self.assertEqual(original, bindings())
        settings = V1.settings()
        with V1.install_cpu_gt_optimizations(expected_sources=V1.source_inventory()):
            retained = bindings()
            error = RuntimeError("independent caller failure")
            with self.assertRaises(RuntimeError) as caught:
                with C.install():
                    active = bindings()
                    with self.assertRaisesRegex(RuntimeError, "nest"):
                        with C.install():
                            self.fail("nested v2 scope admitted")
                    self.assertEqual(active, bindings())
                    self.assertIs(A._json_value.__globals__, vars(A))
                    raise error
            self.assertIs(caught.exception, error)
            self.assertEqual(retained, bindings())
        self.assertEqual(original, bindings())
        self.assertEqual(settings, V1.settings())

    def test_retained_stage_timing_wraps_and_restores_actual_candidates(self):
        with tempfile.TemporaryDirectory(prefix="v2-independent-timing-") as temporary:
            with V1.install_cpu_gt_optimizations(expected_sources=V1.source_inventory()), C.install():
                active = bindings()
                with gt_timing.install_gt_timing(attempt_dir=Path(temporary), max_frames=1,
                                                drain_check=lambda: True):
                    self.assertIs(E.prepare_target.__wrapped__, active[2])
                    self.assertIs(T.write_bundle.__wrapped__, active[4])
                    self.assertIs(G.write_source_archive.__wrapped__, active[5])
                self.assertEqual(active, bindings())

    def test_foreign_entrypoints_rejected_without_displacing_caller(self):
        slots = ((E, "prepare_target"), (E, "restore_prepared_target"),
                 (T, "write_bundle"), (G, "write_source_archive"))
        with V1.install_cpu_gt_optimizations(expected_sources=V1.source_inventory()):
            retained = bindings()
            for owner, name in slots:
                with self.subTest(entrypoint=name):
                    marker = lambda *args, **kwargs: None
                    with mock.patch.object(owner, name, marker):
                        foreign = bindings()
                        with self.assertRaises(RuntimeError):
                            with C.install():
                                self.fail("foreign entrypoint was silently displaced")
                        self.assertEqual(foreign, bindings())
                    self.assertEqual(retained, bindings())
            # Failed entry attempts must also release the candidate lock.
            with C.install():
                self.assertIsNot(E.prepare_target, retained[2])
            self.assertEqual(retained, bindings())


class IndependentFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.SourceGeometryTests().fixture()
        directory, kwargs = cls.fixture.__enter__()
        cls.directory, cls.kwargs = directory, kwargs
        cls.producer_args = dict(static_catalogue_dir=directory / "catalogue",
            experiment_id="independent_v2", ego_id=999, dynamic_population_ids=[1, 2],
            population_evidence_sha256="d" * 64, max_frames=1000)
        producer = V1S.OptimizedSourceProducer(**cls.producer_args)
        with mock.patch("time.clock_gettime_ns", return_value=123456):
            cls.inputs = producer.build(**kwargs)
        cls.evidence = producer.last_evidence
        target = V1T.build_complete_v010(cls.inputs)
        cls.prepared = E.restore_prepared_target(frame_id=100,
            sample_id=cls.inputs.frame["sample_id"], target=target,
            content_sha256=E._target_digest(100, cls.inputs.frame["sample_id"], target))

    @classmethod
    def tearDownClass(cls):
        cls.fixture.__exit__(None, None, None)

    def test_all_ablations_preserve_source_evidence_and_complete_target(self):
        for labels, metadata in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(labels=labels, metadata=metadata):
                private = C.private_functions(fast_labels=labels, fast_metadata=metadata)
                first, second = (private["producer"](**self.producer_args) for _ in range(2))
                self.assertIsNot(first.conflict_state, second.conflict_state)
                self.assertIsNot(first.tracker, second.tracker)
                with mock.patch("time.clock_gettime_ns", return_value=123456):
                    actual = first.build(**self.kwargs)
                self.assertEqual(actual.provenance, self.inputs.provenance)
                self.assertEqual(G.canonical(first.last_evidence), G.canonical(self.evidence))
                self.assertEqual(second.processed_frames, 0)
                prepared = private["prepare"](actual)
                self.assertEqual(prepared.content_sha256, self.prepared.content_sha256)
                self.assertEqual(set(prepared.target), set(self.prepared.target))
                for key, expected in self.prepared.target.items():
                    if isinstance(expected, np.ndarray):
                        np.testing.assert_array_equal(prepared.target[key], expected)
                    else:
                        self.assertEqual(prepared.target[key], expected, key)

    def test_already_validated_array_metadata_and_quarantine_mutations_rejected(self):
        private = C.private_functions()
        mutations = {
            "depth": lambda value: value.depth_bgra.__setitem__((0, 0, 0), value.depth_bgra[0, 0, 0] ^ 1),
            "semantic": lambda value: value.raw_semantic.__setitem__((719, 1279), 99),
            "mask": lambda value: value.original_v3_mask.__setitem__((719, 1279), 3),
            "actor": lambda value: value.actor_boxes[0].__setitem__("gt_distance_m", 999),
            "static": lambda value: value.static_candidates[0].__setitem__("range_m", 999),
            "quarantine": lambda value: value.quarantine["manual_static_ids"].add("foreign"),
        }
        for label, mutate in mutations.items():
            with self.subTest(mutation=label), tempfile.TemporaryDirectory(prefix="v2-independent-mutate-") as temporary:
                bad = copy.deepcopy(self.inputs)
                A._validate(bad)
                mutate(bad)
                with self.assertRaises(A.TargetInputError):
                    private["prepare"](bad)
                with self.assertRaises(A.TargetInputError):
                    private["write"](temporary, identity=self.kwargs["identity"], inputs=bad, prepared=self.prepared)
                self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_prepared_invalid_label_and_png_array_mismatch_not_hidden_by_digest(self):
        restore = C.private_functions()["restore"]
        for label in (3, 254):
            bad = copy.deepcopy(self.prepared.target)
            bad["segmentation"][431, 767] = label
            encoded, png = cv2.imencode(".png", bad["segmentation"])
            self.assertTrue(encoded)
            bad["segmentation_png"] = png.tobytes()
            digest = E._target_digest(100, self.prepared.sample_id, bad)
            with self.assertRaisesRegex(E.EvaluationInputError, "labels/PNG"):
                restore(frame_id=100, sample_id=self.prepared.sample_id, target=bad, content_sha256=digest)
        bad = copy.deepcopy(self.prepared.target)
        bad["segmentation"][0, 0] = (int(bad["segmentation"][0, 0]) + 1) % 3
        with self.assertRaisesRegex(E.EvaluationInputError, "PNG/array mismatch"):
            restore(frame_id=100, sample_id=self.prepared.sample_id, target=bad,
                    content_sha256=E._target_digest(100, self.prepared.sample_id, bad))

    def test_output_mutation_cannot_poison_next_prepare(self):
        prepare = C.private_functions()["prepare"]
        altered = prepare(self.inputs)
        altered.target["tables"]["object_boxes.csv"].append({"sample_id": "foreign"})
        altered.target["sources"].clear()
        rebuilt = prepare(self.inputs)
        self.assertEqual(rebuilt.content_sha256, self.prepared.content_sha256)
        self.assertEqual(rebuilt.target["tables"], self.prepared.target["tables"])
        self.assertEqual(rebuilt.target["sources"], self.prepared.target["sources"])

    def test_payload_archive_bytes_and_write_order_exact(self):
        private = C.private_functions()
        with tempfile.TemporaryDirectory(prefix="v2-independent-bytes-") as temporary:
            root = Path(temporary)
            publications, archives, sequences = [], [], []
            for index, (write, archive) in enumerate(((T.write_bundle, G.write_source_archive),
                                                    (private["write"], private["archive"]))):
                calls = []
                original_fsync, original_link = T.os.fsync, T.os.link
                def fsync(fd):
                    calls.append("fsync")
                    return original_fsync(fd)
                def link(source, destination):
                    calls.append("link_READY")
                    return original_link(source, destination)
                with mock.patch.object(T.time, "clock_gettime_ns", return_value=123456), \
                        mock.patch.object(T.time, "time_ns", return_value=987654), \
                        mock.patch.object(T.os, "fsync", side_effect=fsync), \
                        mock.patch.object(T.os, "link", side_effect=link):
                    result = write(root / str(index), identity=self.kwargs["identity"],
                                   inputs=self.inputs, prepared=self.prepared)
                directory = Path(result["directory"])
                publications.append({path.name: path.read_bytes() for path in directory.iterdir()})
                sequences.append(calls)
                archive_dir = root / f"source_{index}"
                archive(archive_dir, inputs=self.inputs, evidence=self.evidence,
                        rgb={**self.kwargs["rgb"], "_rgb_image": self.kwargs["semantic_image"]},
                        radar=self.kwargs["radar_world_xyz"])
                archives.append({path.name: path.read_bytes() for path in archive_dir.iterdir()})
            self.assertEqual(publications[0], publications[1])
            self.assertEqual(set(publications[0]), set(T.PAYLOAD_NAMES) | {T.READY_NAME})
            self.assertEqual(archives[0], archives[1])
            self.assertEqual(set(archives[0]), {"SOURCE_EVIDENCE.json", "rgb_bgra.npy", "radar_world_xyz.npy", "MANIFEST.json"})
            self.assertEqual(sequences[0], sequences[1])
            self.assertEqual(sequences[1], ["fsync"] * 8 + ["link_READY", "fsync", "fsync"])

    def test_fsync_failure_never_publishes_ready(self):
        write = C.private_functions()["write"]
        with tempfile.TemporaryDirectory(prefix="v2-independent-fsync-") as temporary:
            with mock.patch.object(T.os, "fsync", side_effect=OSError("independent disk failure")):
                with self.assertRaisesRegex(OSError, "independent disk failure"):
                    write(temporary, identity=self.kwargs["identity"], inputs=self.inputs, prepared=self.prepared)
            self.assertFalse((T.bundle_path(temporary, self.kwargs["identity"]) / T.READY_NAME).exists())

    def test_installed_writer_uses_installed_prepared_validator(self):
        with tempfile.TemporaryDirectory(prefix="v2-independent-installed-") as temporary:
            with V1.install_cpu_gt_optimizations(expected_sources=V1.source_inventory()), C.install():
                validator = E.restore_prepared_target
                self.assertIsInstance(validator.__globals__["np"], C.ExactArrayOperations)
                with mock.patch.object(E, "restore_prepared_target", wraps=validator) as called:
                    descriptor = T.write_bundle(temporary, identity=self.kwargs["identity"],
                                                inputs=self.inputs, prepared=self.prepared)
                called.assert_called_once()
                self.assertEqual(descriptor["prepared_content_sha256"], self.prepared.content_sha256)

    def test_cached_private_functions_recheck_source_pins(self):
        private = C.private_functions()
        wrong = dict(A.SOURCES)
        role = next(iter(wrong))
        wrong[role] = (wrong[role][0], "0" * 64)
        # New dictionary on the private adapter preserves actual source files.
        adapter = private["prepare"].__globals__["targets"]
        original = adapter.SOURCES
        adapter.SOURCES = wrong
        try:
            with self.assertRaisesRegex(A.TargetInputError, "source changed"):
                private["prepare"](self.inputs)
        finally:
            adapter.SOURCES = original
        self.assertEqual(private["prepare"](self.inputs).content_sha256, self.prepared.content_sha256)


if __name__ == "__main__":
    unittest.main()
