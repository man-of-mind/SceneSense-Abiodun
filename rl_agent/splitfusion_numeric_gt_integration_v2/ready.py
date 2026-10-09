"""Owned numerical target admission, with real wire/ticket/pose reconciliation.

This is a new semantic and wire schema, not a forged v1 capability. Original
artifact and full AVO-document hashes belong to deferred archival; all numerical
AVO values, complete source provenance and full frame/action identity are bound
now. The checksum detects corruption, NOT an authenticated publisher. A live
transport still must authenticate the producer, reconcile its session, and use
the coordinator's accepted-history/send-confirmation ownership protocol.

The private issuer protects ordinary internal API use, not hostile Python code
in the same interpreter. The startup camera audit allocates CPU tensors only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import struct
from collections.abc import Mapping

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_live_dispatch_v1 import frame_context as FC
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_parity_repair_v2 import prediction_archive as PA
from .edge_digest import PAYLOAD_SCHEMA, semantic_digest

SCHEMA = "splitfusion.numeric_gt.ready_target.v2"
WIRE_SCHEMA = "splitfusion.numeric_gt.ready_envelope.v2"
MAGIC = b"SFGTNUM2"
HEADER = struct.Struct("!8sQ32s")
MASK_BYTES = C.MASK_BYTES
SHAPE = C.SHAPE
MAX_METADATA_BYTES = C.MAX_METADATA_BYTES
MAX_PACKET_BYTES = HEADER.size + MAX_METADATA_BYTES + 2 * MASK_BYTES
ReadyError = C.ContractError
_ISSUER = object()
# One CPU-only audited registry per process, before any admitted frame.
_CAMERA_REGISTRY = FC.StaticCameraRegistry.audited()

_PRODUCER_FIELDS = frozenset(("schema", "capture_identity", "sample_id", "frame", "camera_matrix",
    "camera_inverse", "intrinsics", "object_boxes", "object_ignore_regions", "avo", "provenance",
    "input_content_sha256", "history_before_sha256", "history_after_sha256", "binding_sha256"))
_FIELDS = frozenset(("schema", "target_contract", "run_binding", "producer_binding", "producer_semantic_sha256",
    "capture_identity", "action_identity", "plan_sha256", "sample_id", "input_content_sha256",
    "history_before_sha256", "history_after_sha256", "provenance", "frame", "camera_matrix", "camera_inverse",
    "intrinsics", "object_boxes", "object_ignore_regions", "avo", "frame_context", "prediction_contract"))


def camera_registry():
    """The immutable registry audited once at module startup, never per frame."""
    return _CAMERA_REGISTRY


def _context_document(value):
    C.require(type(value) is FC.FrameContextV1 and type(value.ego_world) is FC.Pose6D,
              "actual immutable FrameContextV1 required")
    try:
        FC.validate_frame_context(value)
    except FC.DispatchContractError as exc:
        raise ReadyError(str(exc)) from exc
    result = C._json_owned(PO.context_document(value))
    C.require(all(type(x) in (int, float) for x in result["ego_world"]), "finite numerical ego pose required")
    return result


def _action(value, capture):
    try:
        MP.validate_run4_identity(value)
    except MP.DP.DirectMapProtocolError as exc:
        raise ReadyError(str(exc)) from exc
    C.require(type(value["session_uuid"]) is str and 0 < len(value["session_uuid"]) <= 512,
              "invalid action session identity")
    C.require(value["frame_id"] == capture["frame_id"] and value["anchor_action_id"] == capture["action_id"],
              "action and capture identity disagree")
    C.require(value["reward_requested"] is True and value["frame_kind"] == "POLICY_DECISION",
              "numerical GT admits accepted reward decisions only")
    for name in ("decision_seq", "ticket_seq", "tensor_seq", "frame_id"):
        C.require(C._nonnegative(value[name]), "invalid bounded policy counter")


def _prediction_contract(value, context):
    try:
        spec = _CAMERA_REGISTRY.resolve(context.camera_model_sha256, context.camera_mount_sha256)
    except FC.DispatchContractError as exc:
        raise ReadyError(str(exc)) from exc
    expected = dict(schema=PA.CONTRACT_SCHEMA, kind="NATIVE_CALIBRATED",
        shape=[spec.source_height, spec.source_width], dtype="uint8", labels=[0, 1, 2],
        camera_model_sha256=spec.camera_model_sha256, camera_mount_sha256=spec.camera_mount_sha256,
        transformation="NONE__PRESERVE_ACTUAL_NATIVE_LABELS")
    # Canonical equality also refuses bools substituting for integer dimensions/labels.
    C.require(C.canonical(value) == C.canonical(expected), "registered native prediction contract differs")
    return spec


def _numeric_matrix(value, side, name):
    C.require(type(value) is list and len(value) == side
        and all(type(row) is list and len(row) == side
            and all(type(x) in (int, float) and C._finite_numeric(x) for x in row) for row in value),
        "exact finite numerical matrix required: " + name)
    return value


def _frame_geometry(value, context):
    frame, capture = value["frame"], value["capture_identity"]
    C.require(type(frame) is dict and frame.get("sample_id") == value["sample_id"]
              and type(frame.get("frame_id")) is int and frame["frame_id"] == capture["frame_id"],
              "full producer frame identity differs")
    for name in ("timestamp", "camera_width", "camera_height", "camera_fx", "camera_fy", "camera_cx",
                 "camera_cy", "camera_x", "camera_y", "camera_z"):
        C.require(type(frame.get(name)) in (int, float) and C._finite_numeric(frame[name]),
                  "missing/nonfinite producer frame calibration: " + name)
    C.require(type(frame["camera_width"]) is int and type(frame["camera_height"]) is int
              and (frame["camera_height"], frame["camera_width"]) == (720, 1280),
              "source frame is not native calibrated geometry")
    C.require(type(frame.get("experiment_id")) is str and bool(frame["experiment_id"]),
              "missing producer experiment identity")
    C.require(type(frame.get("camera_inverse_matrix_json")) is str, "missing original inverse matrix JSON")
    camera = PO._matrix(_numeric_matrix(value["camera_matrix"], 4, "camera"), "producer camera")
    inverse = PO._matrix(_numeric_matrix(value["camera_inverse"], 4, "inverse"), "producer inverse")
    embedded = C._parse(frame["camera_inverse_matrix_json"].encode("utf-8"))
    PO._agree(inverse, _numeric_matrix(embedded, 4, "frame inverse"), "source frame inverse")
    C.require(np.allclose(camera @ inverse, np.eye(4), rtol=0, atol=1e-8),
              "producer camera inverse differs")
    PO._agree(camera, PO.context_camera(context, _CAMERA_REGISTRY), "live context/source camera")
    C.require(np.array_equal(camera[:3, 3], [frame["camera_x"], frame["camera_y"], frame["camera_z"]]),
              "source manifest camera location differs")
    intrinsic = np.asarray(_numeric_matrix(value["intrinsics"], 3, "intrinsic"), dtype=np.float64)
    values = {key: float(x) for key, x in FC.CAMERA_MODEL_NUMERIC_FIELDS}
    expected = np.asarray(((values["camera_fx"], 0., values["camera_cx"]),
        (0., values["camera_fy"], values["camera_cy"]), (0., 0., 1.)))
    C.require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
              and np.array_equal(intrinsic, expected), "producer native intrinsics differ from registry")
    for key, location in (("camera_fx", (0, 0)), ("camera_fy", (1, 1)),
                          ("camera_cx", (0, 2)), ("camera_cy", (1, 2))):
        C.require(frame[key] == intrinsic[location], "source manifest intrinsic differs")
    C.require(type(value["avo"].get("timestamp")) in (int, float)
              and value["avo"]["timestamp"] == frame["timestamp"], "AVO/source simulation timestamp differs")


def _metadata(value, *, expected_capture_identity, expected_action_identity, frame_context,
              plan_sha256, run_binding, producer_binding, prediction_contract):
    C.require(type(value) is dict and set(value) == _FIELDS and value["schema"] == SCHEMA
              and value["target_contract"] == C.TARGET_CONTRACT, "numerical ready target fields/schema differ")
    capture, action = C._json_owned(expected_capture_identity), C._json_owned(expected_action_identity)
    C._capture(capture); C._capture(value["capture_identity"])
    _action(action, capture); _action(value["action_identity"], value["capture_identity"])
    for name in ("run_binding", "producer_binding", "producer_semantic_sha256", "plan_sha256",
                 "input_content_sha256", "history_before_sha256", "history_after_sha256"):
        C.require(C.is_digest(value[name]), "required digest missing: " + name)
    C.require(all(C.is_digest(x) for x in (plan_sha256, run_binding, producer_binding)), "expected binding invalid")
    expected_context = _context_document(frame_context)
    contract = C._json_owned(prediction_contract)
    _prediction_contract(contract, frame_context)
    C.require(value["run_binding"] == run_binding and value["producer_binding"] == producer_binding
        and value["plan_sha256"] == plan_sha256 and C.canonical(value["capture_identity"]) == C.canonical(capture)
        and C.canonical(value["action_identity"]) == C.canonical(action)
        and C.canonical(value["frame_context"]) == C.canonical(expected_context)
        and C.canonical(value["prediction_contract"]) == C.canonical(contract), "foreign ready frame/ticket/binding")
    C.require((frame_context.stream_id, frame_context.frame_id, frame_context.capture_timestamp_ns)
        == (capture["stream_id"], capture["frame_id"], capture["capture_timestamp_ns"])
        and frame_context.sequence_id == action["tensor_seq"], "capture/action/live context disagreement")
    sample = value["sample_id"]
    C.require(type(sample) is str and 0 < len(sample) <= 512 and re.fullmatch(r"[A-Za-z0-9_-]+", sample)
              and not sample.startswith("canonical_v3_"), "new live sample identity required")
    provenance = value["provenance"]
    C.require(type(provenance) is dict, "complete source provenance required")
    C._capture(provenance.get("capture_identity"))
    C.require(provenance.get("schema") == C.INPUT_SCHEMA and provenance.get("scope") == C.LIVE_SCOPE
        and provenance.get("sample_id") == sample and provenance["capture_identity"] == capture
        and provenance.get("input_content_sha256") == value["input_content_sha256"], "source provenance differs")
    C.require(type(provenance.get("complete")) is dict
        and all(provenance["complete"].get(k) is True for k in C.COMPLETENESS), "incomplete source declarations")
    pins = provenance.get("evidence_sha256")
    C.require(type(pins) is dict and {"owned_source", "population", "catalogue", "conflicts"} <= set(pins)
              and all(C.is_digest(x) for x in pins.values()), "complete source evidence pins missing")
    C._rows(value["object_boxes"], frame=capture["frame_id"], sample=sample, boxes=True)
    C._rows(value["object_ignore_regions"], frame=capture["frame_id"], sample=sample, boxes=False)
    C._avo(value["avo"], capture["frame_id"])
    _frame_geometry(value, frame_context)


def _producer_metadata(value):
    result = {k: value[k] for k in _PRODUCER_FIELDS - {"schema", "binding_sha256"}}
    result.update(schema=PAYLOAD_SCHEMA, binding_sha256=value["producer_binding"])
    return result


def _semantic(metadata_bytes, segmentation, object_ignore):
    h = hashlib.sha256(SCHEMA.encode() + b"\0" + struct.pack("!Q", len(metadata_bytes)))
    h.update(metadata_bytes); h.update(segmentation); h.update(object_ignore)
    return h.hexdigest()


def _canonical_owned(value):
    """Serialize already owned finite JSON; do not recursively copy it again."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True, init=False)
class ReadyTarget:
    metadata: Mapping
    metadata_bytes: bytes
    segmentation_bytes: bytes
    object_ignore_bytes: bytes
    semantic_sha256: str
    identity_bytes: bytes
    _issuer: object

    def __init__(self, *args, **kwargs):
        raise ReadyError("use admit_payload or decode_target")

    @property
    def segmentation(self):
        return np.frombuffer(self.segmentation_bytes, np.uint8).reshape(SHAPE)

    @property
    def object_ignore(self):
        return np.frombuffer(self.object_ignore_bytes, np.uint8).reshape(SHAPE)


def require_ready(value):
    C.require(type(value) is ReadyTarget and getattr(value, "_issuer", None) is _ISSUER,
              "private admitted v2 numerical target required")
    return value


def _admit(metadata, segmentation, object_ignore, *, producer_bytes=None, **expected):
    _metadata(metadata, **expected)
    C._mask(segmentation, {0, 1, 2, 255}); C._mask(object_ignore, {0, 255})
    encoded = _canonical_owned(metadata)
    C.require(len(encoded) <= MAX_METADATA_BYTES, "ready metadata exceeds bound")
    producer_encoded = producer_bytes if producer_bytes is not None else _canonical_owned(_producer_metadata(metadata))
    C.require(semantic_digest(producer_encoded, segmentation, object_ignore) == metadata["producer_semantic_sha256"],
              "producer numerical semantic digest differs")
    target = object.__new__(ReadyTarget)
    values = dict(metadata=C._freeze(metadata), metadata_bytes=encoded, segmentation_bytes=segmentation,
        object_ignore_bytes=object_ignore, semantic_sha256=_semantic(encoded, segmentation, object_ignore),
        identity_bytes=C.canonical({k: metadata[k] for k in ("capture_identity", "action_identity", "plan_sha256",
            "run_binding", "producer_binding", "frame_context", "prediction_contract")}), _issuer=_ISSUER)
    for key, value in values.items():
        object.__setattr__(target, key, value)
    return target


def admit_payload(payload, *, expected_capture_identity, expected_action_identity, frame_context,
                  plan_sha256, run_binding, producer_binding, prediction_contract):
    """One actual producer boundary; no PNG, source-file read or archive document.

    Original canonical JSON performs its established NumPy-scalar normalization.
    Owned finite JSON is then strictly validated; arbitrary duck-typed payloads
    and fabricated old artifact digests are never accepted.
    """
    # This producer-only boundary is never invoked by the edge packet decoder.
    # Keep its original type/canonicalization checks, but do not import source
    # geometry (SciPy) merely to import the edge-safe ready module.
    from rl_agent.splitfusion_numeric_gt_producer_v1 import payload as P
    C.require(type(payload) is P.NumericGtPayload, "exact numeric producer payload required")
    raw = P.A.canonical(payload.metadata())
    owned = C._parse(raw)
    C.require(type(owned) is dict and set(owned) == _PRODUCER_FIELDS and owned["schema"] == P.SCHEMA,
              "producer numerical payload schema/fields differ")
    C.require(owned["binding_sha256"] == producer_binding, "foreign producer binding")
    owned.pop("schema"); owned.pop("binding_sha256")
    owned.update(schema=SCHEMA, target_contract=C.TARGET_CONTRACT, producer_binding=producer_binding,
        producer_semantic_sha256=payload.semantic_sha256, run_binding=run_binding, plan_sha256=plan_sha256,
        action_identity=C._json_owned(expected_action_identity), frame_context=_context_document(frame_context),
        prediction_contract=C._json_owned(prediction_contract))
    expected = dict(expected_capture_identity=expected_capture_identity, expected_action_identity=expected_action_identity,
        frame_context=frame_context, plan_sha256=plan_sha256, run_binding=run_binding, producer_binding=producer_binding,
        prediction_contract=prediction_contract)
    return _admit(owned, payload.segmentation_bytes, payload.object_ignore_bytes, producer_bytes=raw, **expected)


def encode_target(target):
    target = require_ready(target)
    # Reuse the admitted canonical metadata bytes; no restore or second JSON
    # validation pass merely to wrap a fixed envelope around immutable bytes.
    prefix = _canonical_owned(dict(schema=WIRE_SCHEMA, semantic_sha256=target.semantic_sha256))
    outer = prefix[:-1] + b',"target":' + target.metadata_bytes + b"}"
    C.require(len(outer) <= MAX_METADATA_BYTES, "ready envelope exceeds bound")
    body = outer + target.segmentation_bytes + target.object_ignore_bytes
    return HEADER.pack(MAGIC, len(outer), hashlib.sha256(body).digest()) + body


def decode_target(packet, *, expected_capture_identity, expected_action_identity, frame_context,
                  plan_sha256, run_binding, producer_binding, prediction_contract):
    """Wire admission needs expected ticket/context, never the producer object."""
    C.require(type(packet) is bytes and HEADER.size < len(packet) <= MAX_PACKET_BYTES,
              "bounded immutable ready packet required")
    magic, size, digest = HEADER.unpack_from(packet)
    C.require(magic == MAGIC and 0 < size <= MAX_METADATA_BYTES
              and len(packet) == HEADER.size + size + 2 * MASK_BYTES, "malformed ready packet bounds")
    body = memoryview(packet)[HEADER.size:]
    C.require(hashlib.sha256(body).digest() == digest, "ready packet checksum differs")
    outer = C._parse(bytes(body[:size]))
    C.require(type(outer) is dict and set(outer) == {"schema", "target", "semantic_sha256"}
              and outer["schema"] == WIRE_SCHEMA, "ready envelope fields/schema differ")
    target = _admit(outer["target"], bytes(body[size:size + MASK_BYTES]), bytes(body[size + MASK_BYTES:]),
        expected_capture_identity=expected_capture_identity, expected_action_identity=expected_action_identity,
        frame_context=frame_context, plan_sha256=plan_sha256, run_binding=run_binding,
        producer_binding=producer_binding, prediction_contract=prediction_contract)
    C.require(target.semantic_sha256 == outer["semantic_sha256"], "ready semantic checksum differs")
    return target
