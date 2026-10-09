"""CPU parity, source-pin, frame-isolation and concurrency regression tests."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from . import target_builder as B
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E


def inputs(frame=1):
    sample = f"adversarial_frame{frame}"
    metadata = dict(sample_id=sample, experiment_id="adversarial", frame_id=frame,
        camera_width=8, camera_height=6, mask_path="mask.png", instance_raw_path="raw.png")
    depth = np.zeros((6, 8, 4), np.uint8); depth[:, :, 0] = 10; depth[:, :, 3] = 255
    semantic = np.zeros((6, 8), np.uint8)
    semantic[frame % 2:frame % 2+4, frame % 3:frame % 3+4] = 14
    provenance = dict(schema=A.SCHEMA, scope="SYNTHETIC_CPU_TEST_NOT_LIVE", sample_id=sample,
        complete=dict.fromkeys(A.COMPLETENESS_KEYS, True), evidence_sha256={"synthetic": "a"*64})
    value = A.CompleteTargetInputs(metadata, "val", depth, semantic, np.zeros((6, 8), np.uint8),
        [], [], [], {key: [] for key in A.QUARANTINE_KEYS}, provenance)
    return replace(value, provenance={**provenance, "input_content_sha256": A.input_content_sha256(value)})


def serialized(target):
    return A.canonical({k: v for k, v in target.items() if k not in
        ("segmentation", "object_ignore", "segmentation_png", "object_ignore_png")})


class TargetBuilderTests(unittest.TestCase):
    def assertExact(self, actual, expected):
        self.assertEqual(serialized(actual), serialized(expected))
        for name in ("segmentation", "object_ignore"):
            self.assertEqual(actual[name+"_png"], expected[name+"_png"])
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_original_numerics_csv_png_summary_and_prepared_contract_exact(self):
        for frame in (1, 2, 3):
            value = inputs(frame); before = A.input_content_sha256(value)
            expected = A.build_complete_v010(value)
            self.assertExact(B.build_complete_v010(value), expected)
            reference = E.prepare_target(value)
            with B.install_target_builder(): prepared = E.prepare_target(value)
            self.assertEqual(prepared.content_sha256, reference.content_sha256)
            self.assertExact(prepared.target, reference.target)
            self.assertEqual(A.input_content_sha256(value), before)

    def test_prewarm_is_code_only_and_each_namespace_is_fresh(self):
        with mock.patch.object(A, "_validate", side_effect=AssertionError("frame validation during prewarm")):
            result = B.prewarm()
        self.assertEqual(result["frame_inputs_consumed"], 0)
        self.assertFalse(result["frame_history_advanced"])
        root, _, raw = B._verified_sources(A.ROOT)
        one = B._modules(root, raw); two = B._modules(root, raw)
        self.assertIsNot(one[0], two[0]); self.assertIsNot(one[1], two[1])
        self.assertIsNot(one[0].build_contract.__globals__, two[0].build_contract.__globals__)
        self.assertIs(one[0].build_contract.__code__, two[0].build_contract.__code__)
        one[0].cv2 = "corrupted private module"
        self.assertIs(two[0].cv2, B.cv2)

    def test_mutated_result_cannot_leak_into_other_frame_or_repeated_call(self):
        one, two = inputs(1), inputs(2)
        expected = E.prepare_target(two)
        with B.install_target_builder():
            mutated = E.prepare_target(one)
            mutated.target["tables"]["object_ignore_regions.csv"][0]["sample_id"] = "corrupt"
            mutated.target["provenance"]["complete"].clear()
            mutated.target["summary"]["ignore_records"] = -99
            mutated.target["segmentation"].setflags(write=True)
            mutated.target["segmentation"][:] = 99
            actual = E.prepare_target(two)
            repeated = E.prepare_target(one)
        self.assertExact(actual.target, expected.target)
        self.assertExact(repeated.target, E.prepare_target(one).target)

    def test_concurrent_distinct_frames_force_overlap_without_cv2_cross_contamination(self):
        values = [inputs(i) for i in range(1, 5)]
        expected = [A.build_complete_v010(v) for v in values]
        barrier = threading.Barrier(4)
        original = B.cv2.connectedComponentsWithStats
        def synchronize(*args, **kwargs):
            barrier.wait(timeout=10)
            return original(*args, **kwargs)
        with mock.patch.object(B.cv2, "connectedComponentsWithStats", side_effect=synchronize):
            with ThreadPoolExecutor(max_workers=4) as pool:
                actual = list(pool.map(B.build_complete_v010, values*3))
        for i, target in enumerate(actual): self.assertExact(target, expected[i % 4])

    def test_source_root_isolation_and_post_prewarm_source_change_refused(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            roots = [Path(first), Path(second)]
            for root in roots:
                for relative, _ in A.SOURCES.values():
                    path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(A.ROOT/relative, path)
                B.prewarm(root=root)
            expected = B.build_complete_v010(inputs(1), root=roots[0])
            self.assertExact(B.build_complete_v010(inputs(1), root=roots[1]), expected)
            with (roots[0]/A.SOURCES["builder"][0]).open("ab") as stream: stream.write(b"\n# changed after prewarm\n")
            with self.assertRaisesRegex(A.TargetInputError, "source changed"):
                B.build_complete_v010(inputs(1), root=roots[0])
            self.assertExact(B.build_complete_v010(inputs(1), root=roots[1]), expected)

    def test_all_input_validation_and_plausible_depth_checks_remain(self):
        value = inputs()
        for broken in (replace(value, quarantine={}), replace(value, depth_bgra=value.depth_bgra[:, :, :3]),
                       replace(value, provenance={**value.provenance, "complete": {}})):
            with self.assertRaises(A.TargetInputError): B.build_complete_v010(broken)
        invalid = replace(value, depth_bgra=np.zeros_like(value.depth_bgra))
        invalid = replace(invalid, provenance={**invalid.provenance, "input_content_sha256": A.input_content_sha256(invalid)})
        with self.assertRaisesRegex(RuntimeError, "implausible retained depth"):
            B.build_complete_v010(invalid)

    def test_no_temporary_filesystem_output_and_context_restores_after_exception(self):
        original = A.build_complete_v010
        with mock.patch.object(A.tempfile, "TemporaryDirectory", side_effect=AssertionError("temporary filesystem output")):
            with B.install_target_builder(): E.prepare_target(inputs())
        self.assertIs(A.build_complete_v010, original)
        with self.assertRaisesRegex(RuntimeError, "scope fixture"):
            with B.install_target_builder(): raise RuntimeError("scope fixture")
        self.assertIs(A.build_complete_v010, original)

    def test_memory_path_rejects_unregistered_escaping_and_duplicate_operations(self):
        output = B._MemoryFiles("sample", "val")
        with self.assertRaises(A.TargetInputError): output.root/"../escape"
        with self.assertRaises(A.TargetInputError): output.root/"/outside"
        with self.assertRaises(A.TargetInputError): (output.root/"unknown.csv").open("x")
        file = output.root/"contracts/v010/val/object_boxes.csv"
        with self.assertRaises(FileNotFoundError): file.open("r")
        with self.assertRaises(A.TargetInputError): file.open("w")
        with self.assertRaises(FileNotFoundError): file.open("x")
        file.parent.mkdir(parents=True)
        with file.open("x", encoding="utf-8", newline="") as stream: stream.write("a\r\n1\r\n")
        with self.assertRaises(FileExistsError): file.open("x")
        with file.open("rb") as stream: self.assertEqual(stream.read(), b"a\r\n1\r\n")

    def test_png_encoder_failure_propagates_no_fabricated_target(self):
        with mock.patch.object(B.cv2, "imencode", return_value=(False, None)):
            with self.assertRaisesRegex(RuntimeError, "failed writing"):
                B.build_complete_v010(inputs())


if __name__ == "__main__":
    unittest.main()
