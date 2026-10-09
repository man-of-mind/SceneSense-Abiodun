"""Admission/wire/pose adversarial tests; CPU-only and no timing claims."""
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_producer_v1 import payload as P
from rl_agent.splitfusion_live_dispatch_v1 import frame_context as FC
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_parity_repair_v2 import prediction_archive as PA
from . import ready as R


def fixture():
    """Exact producer payload type; synthetic values, real native calibration."""
    capture = dict(run_id="run", cell_id="cell", stream_id="stream", frame_id=10,
                   action_id=None, profile_id=None, capture_timestamp_ns=123)
    action = dict(frame_id=10, session_uuid="session", decision_seq=0, tensor_seq=1, ticket_seq=0,
        controller_lineage_sha256="1" * 64, mode_id=11, q_e4=2999,
        execution_bundle_sha256="2" * 64, anchor_action_id=None,
        reward_requested=True, frame_kind="POLICY_DECISION")
    context = FC.build_frame_context_v1(stream_id="stream", frame_id=10, sequence_id=1,
        capture_timestamp_ns=123, ego_world_x=2., ego_world_y=-4., ego_world_z=.5,
        ego_world_pitch=0., ego_world_yaw=20., ego_world_roll=0.)
    camera = PO.context_camera(context, R.camera_registry())
    inverse = np.linalg.inv(camera)
    cal = {name: float(value) for name, value in FC.CAMERA_MODEL_NUMERIC_FIELDS}
    intrinsic = [[cal["camera_fx"], 0., cal["camera_cx"]],
                 [0., cal["camera_fy"], cal["camera_cy"]], [0., 0., 1.]]
    sample = "live_frame10"
    frame = dict(experiment_id="synthetic", sample_id=sample, frame_id=10, timestamp=10.0,
        camera_width=1280, camera_height=720, camera_fx=intrinsic[0][0], camera_fy=intrinsic[1][1],
        camera_cx=640., camera_cy=360., camera_x=camera[0, 3], camera_y=camera[1, 3],
        camera_z=camera[2, 3], camera_inverse_matrix_json=json.dumps(inverse.tolist()),
        mask_path="masks/live_frame10.png", instance_raw_path="semantic/live_frame10.png",
        radar_points_path="radar/live_frame10.npz")
    provenance = dict(schema=C.INPUT_SCHEMA, scope=C.LIVE_SCOPE, sample_id=sample,
        capture_identity=capture, input_content_sha256="a" * 64,
        complete=dict.fromkeys(C.COMPLETENESS, True),
        evidence_sha256=dict.fromkeys(("owned_source", "population", "catalogue", "conflicts"), "b" * 64))
    row = dict(sample_id=sample, frame_id="10", label="vehicle", source_identity="dynamic:7",
        contract_state="valid", object_world_x="1.2345678901234567", object_world_y="-0.0",
        gt_bbox_x="1.5", gt_bbox_y="2.0", gt_bbox_w="3", gt_bbox_h="4", gt_bbox_area_px="12")
    payload = P.NumericGtPayload(schema=P.SCHEMA, capture_identity=P.freeze(capture), sample_id=sample,
        frame=P.freeze(frame), camera_matrix=P.freeze(camera.tolist()), camera_inverse=P.freeze(inverse.tolist()),
        intrinsics=P.freeze(intrinsic), segmentation_bytes=bytes(C.MASK_BYTES),
        object_ignore_bytes=bytes([255]) * C.MASK_BYTES, object_boxes=P.freeze([row]), object_ignore_regions=(),
        avo=P.freeze(dict(frame_id=10, timestamp=10., threshold=.65,
            qualified=[dict(frame_id=10, gt_actor_id="9", world_x=1., world_y=2., qualified=True, avo=.65)],
            structural_ignored=[dict(frame_id=10, gt_actor_id="11", world_x=0., world_y=0.,
                                     qualified=False, reasons=["ambiguous"])])),
        provenance=P.freeze(provenance), input_content_sha256="a" * 64,
        history_before_sha256="e" * 64, history_after_sha256="f" * 64,
        binding_sha256="c" * 64, semantic_sha256="")
    payload = redigest(payload)
    expected = dict(expected_capture_identity=copy.deepcopy(capture), expected_action_identity=copy.deepcopy(action),
        frame_context=context, plan_sha256="d" * 64, run_binding="4" * 64, producer_binding="c" * 64,
        prediction_contract=PA.native_prediction_contract().document())
    return payload, expected


def redigest(payload):
    return replace(payload, semantic_sha256=P.semantic_digest(P.A.canonical(payload.metadata()),
        payload.segmentation_bytes, payload.object_ignore_bytes))


def rewrite(packet, mutation, *, semantic=False, producer=False):
    magic, size, _ = R.HEADER.unpack_from(packet)
    body = packet[R.HEADER.size:]
    outer = json.loads(body[:size]); mutation(outer)
    if producer:
        outer["target"]["producer_semantic_sha256"] = P.semantic_digest(
            C.canonical(R._producer_metadata(outer["target"])),
            body[size:size + R.MASK_BYTES], body[size + R.MASK_BYTES:])
    if semantic:
        outer["semantic_sha256"] = R._semantic(C.canonical(outer["target"]),
            body[size:size + R.MASK_BYTES], body[size + R.MASK_BYTES:])
    encoded = C.canonical(outer)
    newbody = encoded + body[size:]
    return R.HEADER.pack(magic, len(encoded), hashlib.sha256(newbody).digest()) + newbody


class ReadyTests(unittest.TestCase):
    def test_roundtrip_complete_identity_pose_and_masks(self):
        payload, expected = fixture()
        target = R.admit_payload(payload, **expected)
        decoded = R.decode_target(R.encode_target(target), **expected)
        self.assertEqual(target.metadata_bytes, decoded.metadata_bytes)
        self.assertEqual(target.identity_bytes, decoded.identity_bytes)
        self.assertEqual(target.semantic_sha256, decoded.semantic_sha256)
        self.assertEqual(set(target.metadata["action_identity"]), set(expected["expected_action_identity"]))
        self.assertEqual(target.segmentation_bytes, decoded.segmentation_bytes)
        self.assertEqual(target.object_ignore_bytes, decoded.object_ignore_bytes)
        self.assertEqual(target.metadata["object_boxes"][0]["object_world_y"], "-0.0")
        self.assertNotEqual(target.semantic_sha256, payload.semantic_sha256)
        self.assertNotIn("avo_document_sha256", target.metadata)
        self.assertNotIn("artifact_source_sha256", target.metadata)

    def test_no_archive_io_at_readiness_or_wire_admission(self):
        payload, expected = fixture()
        with patch("builtins.open", side_effect=AssertionError("file read")), \
             patch("io.open", side_effect=AssertionError("file read")), \
             patch("cv2.imencode", side_effect=AssertionError("PNG")), \
             patch("cv2.imdecode", side_effect=AssertionError("PNG")), \
             patch("numpy.save", side_effect=AssertionError("archive array")), \
             patch.object(FC.StaticCameraRegistry, "audited", side_effect=AssertionError("repeat camera audit")):
            target = R.admit_payload(payload, **expected)
            R.decode_target(R.encode_target(target), **expected)

    def test_private_issuer_and_no_v1_forgery(self):
        with self.assertRaises(R.ReadyError):
            R.ReadyTarget(already_validated=True)
        with self.assertRaises(R.ReadyError):
            R.require_ready(dict(validated=True))
        with self.assertRaises(R.ReadyError):
            R.require_ready(object.__new__(R.ReadyTarget))
        from rl_agent.splitfusion_numeric_gt_integration_v1.test_contract import admitted
        with self.assertRaises(R.ReadyError):
            R.require_ready(admitted())

    def test_deep_immutable_masks_and_source(self):
        payload, expected = fixture(); target = R.admit_payload(payload, **expected)
        expected["expected_action_identity"]["q_e4"] = 30
        self.assertEqual(target.metadata["action_identity"]["q_e4"], 2999)
        with self.assertRaises(TypeError): target.metadata["frame"]["camera_x"] = 99
        with self.assertRaises(ValueError): target.segmentation.setflags(write=True)
        with self.assertRaises(ValueError): target.object_ignore[0, 0] = 0
        with self.assertRaises(FrozenInstanceError): target.semantic_sha256 = "0" * 64

    def test_wrong_payload_type_binding_or_digest_rejected(self):
        payload, expected = fixture()
        for broken in (object(), replace(payload, semantic_sha256="f" * 64),
                       replace(payload, binding_sha256="f" * 64),
                       replace(payload, segmentation_bytes=bytearray(R.MASK_BYTES))):
            with self.assertRaises(ValueError): R.admit_payload(broken, **expected)

    def test_full_action_identity_not_just_five_fields(self):
        payload, expected = fixture(); packet = R.encode_target(R.admit_payload(payload, **expected))
        for key, value in (("mode_id", 10), ("q_e4", 3000), ("execution_bundle_sha256", "8" * 64),
                           ("controller_lineage_sha256", "8" * 64), ("session_uuid", "foreign"),
                           ("ticket_seq", 9), ("decision_seq", 8), ("tensor_seq", 8),
                           ("anchor_action_id", 71), ("reward_requested", False),
                           ("frame_kind", "POLICY_HOLD"), ("frame_id", 11)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                R.decode_target(rewrite(packet, lambda m: m["target"]["action_identity"].update({key: value}),
                    semantic=True), **expected)

    def test_foreign_expected_plan_run_producer_and_capture_refused(self):
        payload, expected = fixture(); packet = R.encode_target(R.admit_payload(payload, **expected))
        for key in ("plan_sha256", "run_binding", "producer_binding"):
            kw = dict(expected, **{key: "8" * 64})
            with self.assertRaises(ValueError): R.decode_target(packet, **kw)
        for key, value in (("run_id", "foreign"), ("cell_id", "foreign"), ("stream_id", "foreign"),
                           ("frame_id", 11), ("capture_timestamp_ns", 124)):
            kw = dict(expected, expected_capture_identity=dict(expected["expected_capture_identity"], **{key: value}))
            with self.assertRaises(ValueError): R.decode_target(packet, **kw)

    def test_actual_context_required_and_sequence_pose_calibration_checked(self):
        payload, expected = fixture(); context = expected["frame_context"]
        changes = (dict(frame_context=PO.context_document(context)),
            dict(frame_context=replace(context, sequence_id=2)),
            dict(frame_context=replace(context, stream_id="foreign")),
            dict(frame_context=replace(context, capture_timestamp_ns=124)),
            dict(frame_context=replace(context, ego_world=replace(context.ego_world, x=5.))),
            dict(frame_context=replace(context, camera_model_sha256="0" * 64)),
            dict(prediction_contract=dict(expected["prediction_contract"], shape=[432, 768])) )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                R.admit_payload(payload, **{**expected, **change})

    def test_rigid_pose_inverse_intrinsic_and_manifest_consistency(self):
        payload, expected = fixture()
        transforms = []
        matrix = C.thaw(payload.camera_matrix); matrix[0][0] *= 2
        transforms.append(replace(payload, camera_matrix=P.freeze(matrix)))
        inverse = C.thaw(payload.camera_inverse); inverse[0][3] += .1
        transforms.append(replace(payload, camera_inverse=P.freeze(inverse)))
        intrinsic = C.thaw(payload.intrinsics); intrinsic[0][0] += .1
        transforms.append(replace(payload, intrinsics=P.freeze(intrinsic)))
        for name, row, column, value in (("camera_inverse", 3, 3, True),
                                         ("camera_matrix", 3, 3, "1.0"),
                                         ("intrinsics", 2, 2, True)):
            matrix = C.thaw(getattr(payload, name)); matrix[row][column] = value
            transforms.append(replace(payload, **{name: P.freeze(matrix)}))
        for name, value in (("camera_x", 100.), ("camera_fx", 100.), ("frame_id", 11),
                             ("camera_width", True), ("timestamp", 11.)):
            transforms.append(replace(payload, frame=P.freeze({**payload.frame, name: value})))
        for item in transforms:
            with self.assertRaises(ValueError): R.admit_payload(redigest(item), **expected)

    def test_original_pose_tolerance_preserved(self):
        payload, expected = fixture(); context = expected["frame_context"]
        nearby = replace(context, ego_world=replace(context.ego_world, x=context.ego_world.x + .5e-5))
        R.admit_payload(payload, **{**expected, "frame_context": nearby})
        distant = replace(context, ego_world=replace(context.ego_world, x=context.ego_world.x + 2e-5))
        with self.assertRaises(ValueError): R.admit_payload(payload, **{**expected, "frame_context": distant})

    def test_masks_rows_avo_and_provenance_validation_preserved(self):
        payload, expected = fixture()
        changes = [replace(payload, segmentation_bytes=b"\x03" + bytes(R.MASK_BYTES - 1)),
            replace(payload, object_ignore_bytes=b"\x01" + bytes(R.MASK_BYTES - 1)),
            replace(payload, object_boxes=P.freeze([{**payload.object_boxes[0], "object_world_x": "nan"}])),
            replace(payload, avo=P.freeze({**payload.avo, "threshold": .5})),
            replace(payload, avo=P.freeze({**payload.avo, "qualified": list(payload.avo["qualified"]) * 2})),
            replace(payload, provenance=P.freeze({**payload.provenance, "complete": {}})),
            replace(payload, provenance=P.freeze({**payload.provenance, "scope": "FOREIGN"})),
            replace(payload, provenance=P.freeze({**payload.provenance, "evidence_sha256": {"catalogue": "b" * 64}}))]
        for item in changes:
            with self.assertRaises(ValueError): R.admit_payload(redigest(item), **expected)

    def test_producer_numpy_scalars_normalized_but_nonfinite_refused(self):
        payload, expected = fixture()
        raw = replace(payload, frame=P.freeze({**payload.frame, "camera_x": np.float64(payload.frame["camera_x"])}))
        self.assertEqual(R.admit_payload(redigest(raw), **expected).metadata_bytes,
                         R.admit_payload(payload, **expected).metadata_bytes)
        bad = replace(payload, frame=P.freeze({**payload.frame, "camera_x": float("nan")}))
        with self.assertRaises(ValueError): R.admit_payload(bad, **expected)

    def test_wire_corruption_bounds_and_mutability_refused(self):
        payload, expected = fixture(); packet = R.encode_target(R.admit_payload(payload, **expected))
        for broken in (packet[:-1], packet + b"x", b"x" + packet[1:], bytearray(packet),
                       packet[:-1] + bytes([packet[-1] ^ 1]),
                       R.HEADER.pack(R.MAGIC, R.MAX_METADATA_BYTES + 1, bytes(32)) + b"x"):
            with self.assertRaises(ValueError): R.decode_target(broken, **expected)

    def test_outer_checksum_cannot_hide_semantic_or_producer_mismatch(self):
        payload, expected = fixture(); packet = R.encode_target(R.admit_payload(payload, **expected))
        for change in (lambda m: m.update(semantic_sha256="f" * 64),
                       lambda m: m["target"].update(producer_semantic_sha256="f" * 64),
                       lambda m: m["target"]["object_boxes"][0].update(object_world_x="12.5")):
            with self.assertRaises(ValueError): R.decode_target(rewrite(packet, change), **expected)

    def test_rechecks_invalid_numerics_even_with_consistent_hashes(self):
        payload, expected = fixture(); packet = R.encode_target(R.admit_payload(payload, **expected))
        for change in (lambda m: m["target"]["avo"].update(threshold=.4),
                       lambda m: m["target"]["camera_matrix"][0].__setitem__(0, 2.),
                       lambda m: m["target"]["provenance"].update(complete={}),
                       lambda m: m["target"].update(avo_document_sha256="f" * 64)):
            with self.assertRaises(ValueError):
                R.decode_target(rewrite(packet, change, producer=True, semantic=True), **expected)


if __name__ == "__main__":
    unittest.main()
