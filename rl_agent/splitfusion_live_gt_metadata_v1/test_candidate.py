"""Synthetic exact bytes, independent rejection and private wiring; CPU only."""
from collections import OrderedDict, UserDict
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest

import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T, gt_bridge as G, evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1.test_target_transport import fixture
from rl_agent.splitfusion_live_gt_handoff_v1 import engine as H, scoring_packet as P, deferred_files as D
from rl_agent.splitfusion_live_gt_handoff_v1.test_scoring_packet import packet_fixture, BINDING
from . import candidate as C


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity, cls.prepared, cls.avo, cls.packet_bytes = packet_fixture()
        cls.metadata = P._unpack(cls.packet_bytes, binding=BINDING)[0]["input_metadata"]
        cls.inputs = replace(fixture()[1], **cls.metadata)

    def encode_args(self):
        return dict(identity=self.identity, prepared=self.prepared, avo_document=copy.deepcopy(self.avo),
                    input_metadata=copy.deepcopy(self.metadata), ready_raw_ns=123, ready_wall_ns=123457, binding=BINDING)

    def test_recursive_normalizer_canonical_bytes_exact(self):
        private = C.private_bindings()
        values = [None, True, False, 3, -0., "é\n\u0000", np.float64(2.),
                  MappingProxyType({"a": (1, 2)}), UserDict({"a": [1, False]}),
                  OrderedDict([("z", 3), ("a", 2)]),
                  {1: "key", "1": "same-normalized-key", "sets": [{(1, 2), (3, 4)}, frozenset(("a", "b"))]}]
        for i in range(30):
            values.append({"n": i, "rows": [{"i": j, "v": [i/7, None, False]} for j in range(8)]})
        for value in values:
            with self.subTest(value=repr(value)[:50]):
                before = repr(value)
                self.assertEqual(private["transport"]._canonical(value), T._canonical(value))
                self.assertEqual(private["bridge"].canonical(value), G.canonical(value))
                self.assertEqual(private["bridge"].sha(value), G.sha(value))
                self.assertEqual(repr(value), before)
        self.assertGreater(private["counters"]["json_normalize"], 0)

    def test_bad_json_inputs_retain_original_rejection(self):
        private = C.private_bindings()
        cycle = []; cycle.append(cycle)
        for value in (float("nan"), float("inf"), {"nested": [float("-inf")]}, np.int64(2), object(), cycle):
            try:
                T._canonical(value)
            except (ValueError, TypeError, RecursionError) as error:
                with self.assertRaises(type(error)):
                    private["transport"]._canonical(value)
            else:
                self.fail("fixture should be rejected")

    def test_original_modules_and_functions_unchanged(self):
        modules = (A, T, G, E, H, P, D)
        before = {module: {key: id(value) for key, value in vars(module).items()} for module in modules}
        private = C.private_bindings()
        private["packet"].encode_packet(**self.encode_args())
        private["bridge"].sha({"a": [1, 2]})
        self.assertEqual(before, {module: {key: id(value) for key, value in vars(module).items()} for module in modules})
        self.assertIs(private["packet"].encode_packet.__wrapped__.__globals__["E"], E)
        self.assertIs(private["transport"].targets.canonical, A.canonical)

    def test_private_packet_bytes_and_baseline_decoder_exact(self):
        private = C.private_bindings()
        actual = private["packet"].encode_packet(**self.encode_args())
        self.assertEqual(actual, self.packet_bytes)
        result = P.decode_packet(actual, expected_identity=self.identity, binding=BINDING)
        self.assertEqual(result["prepared"].content_sha256, self.prepared.content_sha256)
        self.assertEqual(result["avo_document"], self.avo)
        self.assertEqual(private["counters"]["packet_encode"], 1)
        self.assertGreater(private["counters"]["transport_canonical"], 0)
        self.assertGreater(private["counters"]["json_normalize"], 0)

    def test_encode_still_rejects_adversarial_values(self):
        private = C.private_bindings()
        changes = (
            lambda a: a.update(identity=dict(a["identity"], frame_id=999)),
            lambda a: a.update(ready_raw_ns=0),
            lambda a: a.update(binding="wrong"),
            lambda a: a["avo_document"]["avo"].update(threshold=.5),
            lambda a: a["avo_document"]["avo"].update(qualified=a["avo_document"]["avo"]["qualified"]*2),
            lambda a: a["avo_document"].update(nonfinite=float("nan")),
            lambda a: a["input_metadata"]["provenance"].update(scope="foreign"),
            lambda a: a["input_metadata"]["frame"].update(camera_inverse_matrix_json="[[1,2]]"),
            lambda a: a.update(prepared=replace(a["prepared"], content_sha256="b"*64)),
            lambda a: a.update(prepared=replace(a["prepared"], target=dict(a["prepared"].target,
                               segmentation=np.zeros((3, 4), np.uint8)))),
        )
        for change in changes:
            args = self.encode_args(); change(args)
            for function in (P.encode_packet, private["packet"].encode_packet):
                with self.assertRaises(ValueError):
                    function(**args)

    def test_deferred_target_files_byte_exact(self):
        private = C.private_bindings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            baseline, candidate = D.FileBatch((path,)), D.FileBatch((path,))
            args = dict(identity=self.identity, inputs=self.inputs, prepared=self.prepared)
            a = D.target_files(baseline, path, **args)
            b = private["deferred"].target_files(candidate, path, **args)
            self.assertEqual(a, b)
            self.assertEqual(baseline.files, candidate.files)
            self.assertEqual(baseline.bytes, candidate.bytes)
            self.assertEqual(private["counters"]["target_files"], 1)
            self.assertEqual(private["counters"]["transport_input_metadata"], 1)

    def test_instance_methods_precede_timer_binding_and_keep_validator(self):
        validator = object()
        class StubBase(H.Engine):
            def __init__(self):
                self.validate_inputs = validator
                self.publish_timed = self._publish
                self.serialize_timed = self._serialize
        candidate_type = C.make_engine_class(StubBase)
        first, second = candidate_type(), candidate_type()
        self.assertIs(first.validate_inputs, validator)
        self.assertIs(first.publish_timed.__func__.__wrapped__.__code__, H.Engine._publish.__code__)
        self.assertIs(first.serialize_timed.__func__.__wrapped__.__code__, H.Engine._serialize.__code__)
        namespace = first.publish_timed.__func__.__wrapped__.__globals__
        self.assertIsNot(namespace["T"], T)
        self.assertIs(namespace["encode_packet"].__wrapped__.__globals__["T"], namespace["T"])
        self.assertIs(namespace["target_files"].__wrapped__.__globals__["T"], namespace["T"])
        self.assertIsNot(first.metadata_counters, second.metadata_counters)
        namespace["T"]._canonical({"x": 1})
        self.assertGreater(first.metadata_counters["json_normalize"], 0)
        self.assertEqual(second.metadata_counters, dict.fromkeys(C.COUNTERS, 0))

    def test_native_publisher_validation_is_not_replaced_or_skipped(self):
        private = C.private_bindings()
        called = []
        def reject(*args):
            called.append(args)
            raise RuntimeError("independent native validator")
        owner = SimpleNamespace(validate_inputs=reject)
        with self.assertRaisesRegex(RuntimeError, "independent native validator"):
            private["methods"]["_publish"](owner, Path("/unused"), identity=self.identity,
                inputs=self.inputs, prepared=self.prepared, avo_document=self.avo)
        self.assertEqual(len(called), 1)
        self.assertEqual(private["counters"]["publish"], 1)
        self.assertEqual(private["counters"]["packet_encode"], 0)

    def test_source_evidence_digest_check_stays_independent(self):
        private = C.private_bindings()
        inputs = replace(self.inputs, provenance=dict(self.inputs.provenance,
                         evidence_sha256=dict(self.inputs.provenance["evidence_sha256"], owned_source="0"*64)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for function in (H.Engine._serialize, private["methods"]["_serialize"]):
                with self.assertRaisesRegex(ValueError, "source evidence digest differs"):
                    function(SimpleNamespace(), D.FileBatch((path,)), path, identity=self.identity,
                             inputs=inputs, prepared=self.prepared, evidence={"tampered": True}, rgb={}, radar=None)
        self.assertGreater(private["counters"]["archive_sha"], 0)
        self.assertGreater(private["counters"]["archive_canonical"], 0)

    def test_complete_serializer_payloads_exact(self):
        private = C.private_bindings()
        evidence = {"identity": self.identity, "synthetic": [1, {"nested": [True, None]}]}
        inputs = replace(self.inputs, provenance=dict(self.inputs.provenance,
                         evidence_sha256=dict(self.inputs.provenance["evidence_sha256"], owned_source=G.sha(evidence))))
        target = dict(self.prepared.target, provenance=inputs.provenance)
        prepared = E.restore_prepared_target(frame_id=self.prepared.frame_id, sample_id=self.prepared.sample_id,
            target=target, content_sha256=E._target_digest(self.prepared.frame_id, self.prepared.sample_id, target))
        image = SimpleNamespace(frame=self.identity["frame_id"], width=1280, height=720, raw_data=bytes(1280*720*4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            baseline, candidate = D.FileBatch((path,)), D.FileBatch((path,))
            args = dict(identity=self.identity, inputs=inputs, prepared=prepared, evidence=evidence,
                        rgb={"_rgb_image": image}, radar=np.zeros((1, 3), np.float64))
            a = H.Engine._serialize(SimpleNamespace(), baseline, path, **args)
            b = private["methods"]["_serialize"](SimpleNamespace(), candidate, path, **args)
            self.assertEqual(a, b)
            self.assertEqual(baseline.files, candidate.files)
        self.assertEqual(private["counters"]["serialize"], 1)
        self.assertEqual(private["counters"]["archive_sha"], 1)
        self.assertEqual(private["counters"]["archive_canonical"], 3)

    def test_failed_base_init_keeps_diagnostic_counters(self):
        observed = []
        class FailingBase(H.Engine):
            def __init__(self):
                observed.append(dict(self.metadata_counters))
                raise RuntimeError("startup fault")
        with self.assertRaisesRegex(RuntimeError, "startup fault"):
            C.make_engine_class(FailingBase)()
        self.assertEqual(observed, [dict.fromkeys(C.COUNTERS, 0)])

    def test_foreign_engine_override_rejected(self):
        class Changed(H.Engine):
            def _publish(self, *args, **kwargs):
                raise AssertionError("foreign")
        for base in (object, Changed, None):
            with self.assertRaises(ValueError):
                C.make_engine_class(base)


if __name__ == "__main__":
    unittest.main()
