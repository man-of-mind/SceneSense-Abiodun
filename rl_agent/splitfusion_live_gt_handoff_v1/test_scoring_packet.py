"""Exact packet contracts with real frozen target validators; CPU only."""
from dataclasses import replace
import hashlib
import json
import unittest

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1.test_target_transport import fixture
from . import scoring_packet as P

BINDING = "a" * 64


def packet_fixture(frame=10, **identity_changes):
    identity, inputs, prepared = fixture()
    identity = dict(identity, frame_id=frame, **identity_changes)
    sample = f"live_test_frame{frame}"
    target = dict(prepared.target)
    target["provenance"] = dict(target["provenance"], sample_id=sample, capture_identity=identity)
    inputs = replace(inputs, frame=dict(inputs.frame, frame_id=frame, sample_id=sample,
        camera_inverse_matrix_json=json.dumps(np.eye(4).tolist())), provenance=target["provenance"])
    target["provenance"] = dict(target["provenance"], input_content_sha256=P.E.targets.input_content_sha256(inputs))
    inputs = replace(inputs, provenance=target["provenance"])
    target["tables"] = {name: [dict(row, sample_id=sample, frame_id=str(frame)) for row in rows]
                        for name, rows in target["tables"].items()}
    prepared = P.E.restore_prepared_target(frame_id=frame, sample_id=sample, target=target,
        content_sha256=P.E._target_digest(frame, sample, target))
    avo = dict(schema=P.LG.SCHEMA, mask_contract=P.LG.MASK_CONTRACT, record_kind="AVO_SCORED_FRAME",
        identity=identity, source_frame_id=frame, native_shape=[720, 1280], ready_wall_ns=123456,
        opaque_preserved_evidence=dict(source="SYNTHETIC_NOT_LIVE", values=[1, 2, 3]),
        avo=dict(frame_id=frame, timestamp=1., threshold=.65,
            qualified=[dict(frame_id=frame, gt_actor_id="42", world_x=1., world_y=2.,
                            qualified=True, avo=.75, no_support=False)], structural_ignored=[]))
    payload = P.encode_packet(identity=identity, prepared=prepared, avo_document=avo,
        input_metadata=P.T._input_metadata(inputs),
        ready_raw_ns=123, ready_wall_ns=123457, binding=BINDING)
    return identity, prepared, avo, payload


def rewritten(payload, change):
    metadata, blobs, _avo = P._unpack(payload, binding=BINDING)
    change(metadata, blobs)
    metadata["blobs"] = {name: dict(bytes=len(value), sha256=P.digest(value)) for name, value in blobs.items()}
    encoded = P.T._canonical(metadata)
    body = encoded + b"".join(blobs[name] for name in P.BLOBS)
    return P.HEADER.pack(P.MAGIC, len(encoded), hashlib.sha256(body).digest()) + body


class PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity, cls.prepared, cls.avo, cls.payload = packet_fixture()

    def decode(self, value=None, **kwargs):
        return P.decode_packet(self.payload if value is None else value,
            expected_identity=kwargs.get("identity", self.identity), binding=kwargs.get("binding", BINDING))

    def test_exact_roundtrip_and_immutable_wire_arrays(self):
        result = self.decode()
        self.assertEqual(type(self.payload), bytes)
        self.assertEqual(result["avo_document"], self.avo)
        self.assertEqual(result["avo"], self.avo["avo"])
        self.assertEqual(result["avo_sidecar_sha256"], P.digest(P.T._canonical(self.avo)))
        self.assertEqual(result["manifest_sha256"], P.digest(self.payload))
        self.assertEqual(result["prepared"].content_sha256, self.prepared.content_sha256)
        self.assertEqual(result["prepared"].target["tables"], self.prepared.target["tables"])
        self.assertEqual(result["gt_ready_wall_ns"], 123457)
        for name in ("segmentation", "object_ignore"):
            expected = self.prepared.target[name]
            actual = result["prepared"].target[name]
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(result["prepared"].target[name + "_png"], self.prepared.target[name + "_png"])
            with self.assertRaises(ValueError):
                actual.setflags(write=True)
        self.assertNotIn(b"depth_bgra.npy", self.payload)

    def test_corruption_truncation_extra_bytes_and_mutable_payload_rejected(self):
        for invalid in (self.payload[:-1], self.payload + b"x", b"x" + self.payload[1:],
                        self.payload[:-1] + bytes([self.payload[-1] ^ 1]), bytearray(self.payload), b""):
            with self.subTest(length=len(invalid)):
                with self.assertRaises(ValueError):
                    self.decode(invalid)

    def test_foreign_identity_binding_and_bad_clocks(self):
        for changes in (dict(identity=dict(self.identity, frame_id=11)), dict(binding="b" * 64),
                        dict(binding="invalid")):
            with self.assertRaises(ValueError):
                self.decode(**changes)
        for name, value in (("ready_raw_ns", 0), ("ready_wall_ns", True), ("binding", "b" * 64),
                            ("frame_id", 11), ("content_sha256", "f" * 64)):
            with self.subTest(name=name):
                bad = rewritten(self.payload, lambda meta, blobs: meta.update({name: value}))
                with self.assertRaises(ValueError):
                    self.decode(bad)

    def test_valid_outer_checksum_does_not_hide_invalid_png(self):
        for replacement in (b"not a PNG", self.prepared.target["segmentation_png"][:16]
                            + b"\x00\x00\xff\xff" + self.prepared.target["segmentation_png"][20:]):
            bad = rewritten(self.payload, lambda meta, blobs: blobs.update({"segmentation.png": replacement}))
            with self.assertRaises(ValueError):
                self.decode(bad)

    def test_avo_threshold_identity_and_actor_integrity(self):
        mutations = [lambda doc: doc["avo"].update(threshold=.5),
            lambda doc: doc.update(source_frame_id=11),
            lambda doc: doc["identity"].update(frame_id=11),
            lambda doc: doc["avo"].update(qualified=doc["avo"]["qualified"] * 2),
            lambda doc: doc["avo"]["qualified"][0].update(avo=2.),
            lambda doc: doc["avo"].pop("structural_ignored")]
        for mutate in mutations:
            def change(meta, blobs):
                doc = json.loads(blobs["avo.json"])
                mutate(doc)
                blobs["avo.json"] = P.T._canonical(doc)
            with self.assertRaises(ValueError):
                self.decode(rewritten(self.payload, change))

    def test_provenance_and_complete_flags_remain_originally_validated(self):
        for name, value in (("scope", "HISTORICAL"), ("complete", {}),
                            ("input_content_sha256", "invalid"), ("evidence_sha256", {})):
            bad = rewritten(self.payload, lambda meta, blobs: meta["target"]["provenance"].update({name: value}))
            with self.assertRaises(ValueError):
                self.decode(bad)

    def test_encode_revalidates_arrays_and_identity(self):
        invalid = dict(self.prepared.target, segmentation=np.zeros((3, 4), np.uint8))
        for prepared, identity in ((replace(self.prepared, target=invalid), self.identity),
                                   (self.prepared, dict(self.identity, frame_id=11))):
            with self.assertRaises(ValueError):
                P.encode_packet(identity=identity, prepared=prepared, avo_document=self.avo,
                    input_metadata=P._unpack(self.payload, binding=BINDING)[0]["input_metadata"],
                    ready_raw_ns=1, ready_wall_ns=2, binding=BINDING)

    def test_duplicate_keys_nonfinite_numbers_and_unknown_envelope_fields(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":1e999}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                P._json(raw)
        bad = rewritten(self.payload, lambda meta, blobs: meta.update(extra="foreign"))
        with self.assertRaises(ValueError):
            self.decode(bad)

    def test_bounded_header_before_json_decode(self):
        for size in (0, P.MAX_METADATA_BYTES + 1):
            invalid = P.HEADER.pack(P.MAGIC, size, b"0" * 32) + b"x"
            with self.assertRaises(ValueError):
                self.decode(invalid)

    def test_complete_native_metadata_pose_and_independent_binding(self):
        result = self.decode()
        self.assertEqual(json.loads(result["input_metadata"]["frame"]["camera_inverse_matrix_json"]), np.eye(4).tolist())
        self.assertEqual(result["input_content_sha256"], self.prepared.target["provenance"]["input_content_sha256"])
        mutations = [lambda m: m["input_metadata"]["frame"].update(frame_id=11),
            lambda m: m["input_metadata"]["frame"].update(camera_inverse_matrix_json="[[1,2]]"),
            lambda m: m["input_metadata"]["provenance"].update(capture_identity=dict(self.identity, frame_id=11)),
            lambda m: m.update(input_content_sha256="b" * 64)]
        for mutate in mutations:
            with self.assertRaises(ValueError):
                self.decode(rewritten(self.payload, lambda m, blobs: mutate(m)))


if __name__ == "__main__":
    unittest.main()
