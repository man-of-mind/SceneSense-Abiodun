"""Boundary, immutable ownership and legacy fixture tests; CPU only."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from . import contract as C


def fixture():
    capture = dict(run_id="run", cell_id="cell", stream_id="stream", frame_id=10,
                   action_id=None, profile_id=None, capture_timestamp_ns=123)
    policy = dict(frame_id=10, session_uuid="session", decision_seq=0, tensor_seq=1, ticket_seq=0)
    provenance = dict(schema=C.INPUT_SCHEMA, scope=C.LIVE_SCOPE, sample_id="live_frame10",
        capture_identity=capture, input_content_sha256="a" * 64,
        complete=dict.fromkeys(C.COMPLETENESS, True), evidence_sha256={"synthetic_not_live": "b" * 64})
    row = dict(sample_id="live_frame10", frame_id="10", label="vehicle", source_identity="dynamic:7",
        contract_state="valid", object_world_x="1.2345678901234567", object_world_y="-0.0",
        gt_bbox_x="1.5", gt_bbox_y="2.0", gt_bbox_w="3", gt_bbox_h="4", gt_bbox_area_px="12")
    metadata = dict(schema=C.SCHEMA, target_contract=C.TARGET_CONTRACT, binding="c" * 64,
        capture_identity=capture, policy_identity=policy, plan_sha256="d" * 64, sample_id="live_frame10",
        input_content_sha256="a" * 64, history_before_sha256="e" * 64, history_after_sha256="f" * 64,
        provenance=provenance, camera_inverse=np.eye(4).tolist(), object_boxes=[row],
        object_ignore_regions=[], avo=dict(frame_id=10, threshold=.65,
            qualified=[dict(frame_id=10, gt_actor_id="9", world_x=1., world_y=2., qualified=True, avo=.65)],
            structural_ignored=[dict(frame_id=10, gt_actor_id="11", world_x=0., world_y=0.,
                                     qualified=False, reasons=["ambiguous"]) ]),
        artifact_source_sha256="1" * 64, avo_document_sha256="2" * 64)
    expected = dict(expected_binding="c" * 64, expected_capture_identity=copy.deepcopy(capture),
                    expected_policy_identity=copy.deepcopy(policy), expected_plan_sha256="d" * 64)
    return metadata, bytes(C.MASK_BYTES), bytes([255]) * C.MASK_BYTES, expected


def admitted():
    metadata, segmentation, object_ignore, expected = fixture()
    return C.admit_target(metadata=metadata, segmentation=segmentation, object_ignore=object_ignore, **expected)


def rewrite(payload, mutation):
    magic, size, _ = C.HEADER.unpack_from(payload)
    body = payload[C.HEADER.size:]
    meta = json.loads(body[:size]); mutation(meta)
    encoded = C.canonical(meta)
    newbody = encoded + body[size:]
    return C.HEADER.pack(magic, len(encoded), hashlib.sha256(newbody).digest()) + newbody


class ContractTests(unittest.TestCase):
    def test_roundtrip_exact_identity_rows_masks_and_clocks(self):
        target = admitted(); expected = fixture()[3]
        packet = C.encode_target(target, ready_raw_ns=1000, ready_wall_ns=2000)
        result = C.decode_target(packet, **expected)
        self.assertEqual(result.target.metadata_bytes, target.metadata_bytes)
        self.assertEqual(result.target.semantic_sha256, target.semantic_sha256)
        self.assertEqual(result.target.segmentation_bytes, target.segmentation_bytes)
        self.assertEqual(result.target.object_ignore_bytes, target.object_ignore_bytes)
        self.assertEqual((result.ready_raw_ns, result.ready_wall_ns), (1000, 2000))
        self.assertEqual(result.target.metadata["object_boxes"][0]["object_world_y"], "-0.0")

    def test_deep_immutable_not_reversible_numpy_flag(self):
        metadata, seg, obj, expected = fixture()
        target = C.admit_target(metadata=metadata, segmentation=seg, object_ignore=obj, **expected)
        original = target.metadata_bytes
        metadata["object_boxes"][0]["object_world_x"] = "999"
        metadata["provenance"]["complete"].clear()
        self.assertEqual(target.metadata_bytes, original)
        with self.assertRaises(TypeError):
            target.metadata["object_boxes"][0]["object_world_x"] = "123"
        with self.assertRaises(ValueError):
            target.segmentation.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            target.segmentation_bytes = b"changed"
        mutable = C.thaw(target.metadata)
        mutable["avo"]["qualified"].clear()
        self.assertEqual(len(target.metadata["avo"]["qualified"]), 1)

    def test_no_public_unchecked_constructor_or_flag(self):
        with self.assertRaises(C.ContractError):
            C.NumericalTarget(already_validated=True)
        with self.assertRaises(C.ContractError):
            C.require_admitted({"already_validated": True})

    def test_semantic_digest_binds_values_row_order_and_provenance(self):
        baseline = admitted().semantic_sha256
        mutations = (lambda m: m.update(plan_sha256="2" * 64),
            lambda m: m["object_boxes"][0].update(object_world_x="1.3"),
            lambda m: m["avo"]["qualified"][0].update(avo=.7),
            lambda m: m.update(history_after_sha256="2" * 64))
        for change in mutations:
            m, s, o, expected = fixture(); change(m)
            expected["expected_plan_sha256"] = m["plan_sha256"]
            target = C.admit_target(metadata=m, segmentation=s, object_ignore=o, **expected)
            self.assertNotEqual(target.semantic_sha256, baseline)

    def test_foreign_binding_capture_policy_plan_rejected(self):
        for key, replacement in (("expected_binding", "a" * 64),
                                 ("expected_plan_sha256", "e" * 64)):
            m, s, o, expected = fixture(); expected[key] = replacement
            with self.assertRaises(C.ContractError):
                C.admit_target(metadata=m, segmentation=s, object_ignore=o, **expected)
        for field, changes in (("capture_identity", {"frame_id": 11}),
                               ("policy_identity", {"ticket_seq": 9}),
                               ("provenance", {"scope": "HISTORICAL"})):
            m, s, o, expected = fixture(); m[field].update(changes)
            with self.assertRaises(C.ContractError):
                C.admit_target(metadata=m, segmentation=s, object_ignore=o, **expected)

    def test_malformed_metadata_labels_and_mutable_buffers_refused(self):
        changes = (lambda m: m.update(extra=True), lambda m: m.update(camera_inverse=[]),
            lambda m: m["provenance"].update(complete={}),
            lambda m: m["object_boxes"][0].update(frame_id="11"),
            lambda m: m["object_boxes"][0].update(object_world_x="nan"),
            lambda m: m["avo"].update(threshold=.5),
            lambda m: m["avo"]["qualified"].extend(m["avo"]["qualified"]),
            lambda m: m["avo"]["qualified"][0].update(world_x=float("inf")))
        for change in changes:
            m, s, o, expected = fixture(); change(m)
            with self.assertRaises(C.ContractError):
                C.admit_target(metadata=m, segmentation=s, object_ignore=o, **expected)
        for seg in (bytearray(C.MASK_BYTES), bytes(C.MASK_BYTES - 1), b"\x03" + bytes(C.MASK_BYTES - 1)):
            m, _, o, expected = fixture()
            with self.assertRaises(C.ContractError):
                C.admit_target(metadata=m, segmentation=seg, object_ignore=o, **expected)

    def test_provenance_identity_cannot_use_bool_as_integer(self):
        m, s, o, expected = fixture()
        m["capture_identity"]["capture_timestamp_ns"] = 1
        expected["expected_capture_identity"]["capture_timestamp_ns"] = 1
        m["provenance"]["capture_identity"] = dict(m["capture_identity"], capture_timestamp_ns=True)
        with self.assertRaises(C.ContractError):
            C.admit_target(metadata=m, segmentation=s, object_ignore=o, **expected)

    def test_wire_corrupt_truncated_oversized_or_mutable_refused(self):
        packet = C.encode_target(admitted(), ready_raw_ns=1, ready_wall_ns=2)
        expected = fixture()[3]
        variants = (packet[:-1], packet + b"x", b"x" + packet[1:], bytearray(packet),
            packet[:-1] + bytes([packet[-1] ^ 1]),
            C.HEADER.pack(C.MAGIC, C.MAX_METADATA_BYTES + 1, b"0" * 32) + b"x")
        for payload in variants:
            with self.assertRaises(C.ContractError):
                C.decode_target(payload, **expected)

    def test_outer_checksum_cannot_hide_bad_semantics_or_clock(self):
        packet = C.encode_target(admitted(), ready_raw_ns=1, ready_wall_ns=2)
        for change in (lambda m: m.update(semantic_sha256="f" * 64),
                       lambda m: m.update(ready_raw_ns=True),
                       lambda m: m["target"]["avo"].update(threshold=.1)):
            with self.assertRaises(C.ContractError):
                C.decode_target(rewrite(packet, change), **fixture()[3])

    def test_json_duplicates_nonfinite_deep_or_unexpected_type_refused(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":1e999}', b'{"x":NaN}',
                    b'[' * 100 + b'0' + b']' * 100):
            with self.assertRaises(C.ContractError):
                C._parse(raw)
        for value in ({1: "bad key"}, {"set": {1}}, {"np": np.float32(1)}, {"large": 1 << 200}):
            with self.assertRaises(C.ContractError):
                C.canonical(value)

    def test_legacy_hash_binding_separate_not_needed_for_readiness(self):
        target = admitted()
        artifacts = dict.fromkeys(("segmentation.png", "object_ignore.png", "object_boxes.csv",
                                   "object_ignore_regions.csv", "target_manifest.csv"), "a" * 64)
        receipt = C.bind_legacy_artifacts(target, prepared_content_sha256="b" * 64, artifact_hashes=artifacts)
        self.assertEqual(receipt["semantic_sha256"], target.semantic_sha256)
        self.assertEqual(receipt["original_prepared_content_sha256"], "b" * 64)
        self.assertNotEqual(receipt["semantic_sha256"], receipt["original_prepared_content_sha256"])
        with self.assertRaises(C.ContractError):
            C.bind_legacy_artifacts(target, prepared_content_sha256="b" * 64, artifact_hashes={})

    def test_scoring_view_has_no_io_png_or_revalidation(self):
        from .oracle_bridge import scorer_inputs
        target = admitted()
        with patch("builtins.open", side_effect=AssertionError("no reads")), \
             patch.object(C, "_metadata", side_effect=AssertionError("no repeat admission")):
            inputs = scorer_inputs(target)
        self.assertEqual(inputs["object_boxes"], target.metadata["object_boxes"])
        self.assertTrue(np.shares_memory(inputs["segmentation"], target.segmentation))

    def test_legacy_fixture_bridge_exact_and_never_a_live_producer(self):
        from rl_agent.splitfusion_live_gt_handoff_v1.test_scoring_packet import packet_fixture
        from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as P
        from .oracle_bridge import from_legacy
        from .evidence import admit_source_evidence, own_legacy_reference, reconcile_source
        capture, prepared, avo, packet = packet_fixture()
        metadata, _, _ = P._unpack(packet, binding="a" * 64)
        source = admit_source_evidence(input_metadata=metadata["input_metadata"], avo_document=avo,
                                      expected_capture_identity=capture)
        target = from_legacy(prepared=prepared, source_evidence=source,
            policy_identity=dict(frame_id=capture["frame_id"], session_uuid="s", decision_seq=0,
                                 tensor_seq=0, ticket_seq=0),
            binding="a" * 64, plan_sha256="b" * 64, history_before_sha256="c" * 64,
            history_after_sha256="d" * 64)
        self.assertTrue(np.array_equal(target.segmentation, prepared.target["segmentation"]))
        self.assertEqual(C.thaw(target.metadata["avo"]), avo["avo"])
        self.assertEqual(C.thaw(target.metadata["object_boxes"]), prepared.target["tables"]["object_boxes.csv"])
        reference = own_legacy_reference(target, source, target_tables=prepared.target["tables"],
                                         old_prepared_sha256=prepared.content_sha256)
        self.assertEqual(C.thaw(reference.target_tables), prepared.target["tables"])
        self.assertEqual(reference.old_prepared_sha256, prepared.content_sha256)
        foreign = copy.deepcopy(avo); foreign["identity"]["run_id"] = "foreign_run"
        with self.assertRaises(ValueError):
            admit_source_evidence(input_metadata=metadata["input_metadata"], avo_document=foreign,
                                  expected_capture_identity=capture)
        altered = copy.deepcopy(avo); altered["avo"]["qualified"][0]["avo"] = .9
        other = admit_source_evidence(input_metadata=metadata["input_metadata"], avo_document=altered,
                                      expected_capture_identity=capture)
        with self.assertRaises(ValueError):
            reconcile_source(target, other)
        with self.assertRaises(TypeError):
            reference.target_tables["target_manifest.csv"][0]["sample_id"] = "mutated"


if __name__ == "__main__":
    unittest.main()
