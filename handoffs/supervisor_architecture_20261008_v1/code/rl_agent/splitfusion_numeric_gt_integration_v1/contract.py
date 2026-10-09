"""Immutable, admitted numerical targets; independent of producer/scorer code.

Only the admission factories construct NumericalTarget. This is an internal
typed capability, not a sandbox against arbitrary Python running in-process.
The wire checksum provides integrity, not publisher authentication: eventual
live transport must separately authenticate its publisher and reconcile pose.

Old prepared-target hashes include PNGs. The semantic hash here intentionally
uses another schema. Legacy artifacts are bound afterward, never fabricated.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from types import MappingProxyType

import numpy as np

SCHEMA = "splitfusion.numeric_gt.semantic_target.v1"
WIRE_SCHEMA = "splitfusion.numeric_gt.envelope.v1"
ARTIFACT_SCHEMA = "splitfusion.numeric_gt.legacy_artifact_binding.v1"
LIVE_SCOPE = "PROSPECTIVE_LIVE_V010_CONSERVATIVE_CONFLICTS_V1"
INPUT_SCHEMA = "splitfusion.complete_v010_inputs.v1"
TARGET_CONTRACT = "ORIGINAL_ROUTE_B_V3_1_V010_COMPLETE"
COMPLETENESS = (
    "native_depth_and_semantic", "original_v3_person_mask", "all_dynamic_actor_boxes",
    "all_dynamic_actor_visibility", "all_static_candidate_rows", "static_catalogue_provenance",
    "identity_and_frame_conflicts", "registered_quarantine_scope",
)
CAPTURE_FIELDS = frozenset(("run_id", "cell_id", "stream_id", "frame_id", "action_id",
                            "profile_id", "capture_timestamp_ns"))
POLICY_FIELDS = frozenset(("frame_id", "session_uuid", "decision_seq", "tensor_seq", "ticket_seq"))
SHAPE = (432, 768)
MASK_BYTES = SHAPE[0] * SHAPE[1]
MAX_ROWS = 10000
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_PACKET_BYTES = MAX_METADATA_BYTES + 2 * MASK_BYTES + 64
MAGIC = b"SFGTNUM1"
HEADER = struct.Struct("!8sQ32s")
_ISSUER = object()


class ContractError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ContractError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def is_digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json_owned(value, depth=0):
    """Exact finite JSON values; no coercion of key, float, row or list order."""
    require(depth <= 40, "metadata nesting limit exceeded")
    if value is None or type(value) in (str, bool, int):
        if type(value) is str:
            require(len(value) <= MAX_METADATA_BYTES, "metadata string too large")
        if type(value) is int:
            require(value.bit_length() <= 128, "metadata integer too large")
        return value
    if type(value) is float:
        require(math.isfinite(value), "nonfinite metadata")
        return value
    if isinstance(value, Mapping):
        require(len(value) <= 10000 and all(type(k) is str for k in value), "invalid metadata keys")
        return {k: _json_owned(v, depth + 1) for k, v in value.items()}
    if type(value) in (list, tuple):
        require(len(value) <= MAX_ROWS, "metadata sequence limit exceeded")
        return [_json_owned(v, depth + 1) for v in value]
    raise ContractError("metadata must contain finite JSON primitives")


def canonical(value):
    try:
        return json.dumps(_json_owned(value), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ContractError("invalid canonical metadata") from exc


def _parse(raw):
    require(type(raw) is bytes and len(raw) <= MAX_METADATA_BYTES, "bounded metadata bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate metadata key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite JSON")))
        return _json_owned(value)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ContractError("malformed finite metadata JSON") from exc


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def thaw(value):
    """Explicit owned mutable export, for artifacts/adapters, not repeated scoring."""
    return _json_owned(value)


def _nonnegative(value):
    return type(value) is int and 0 <= value < 2**63


def _capture(value):
    require(type(value) is dict and set(value) == CAPTURE_FIELDS, "complete capture identity required")
    for key in ("run_id", "cell_id", "stream_id"):
        require(type(value[key]) is str and 0 < len(value[key]) <= 512, "invalid capture name")
    for key in ("frame_id", "capture_timestamp_ns"):
        require(_nonnegative(value[key]), "invalid capture frame/clock")
    action, profile = value["action_id"], value["profile_id"]
    require((action is None) == (profile is None), "partial anchor identity")
    if action is not None:
        require(_nonnegative(action) and type(profile) is str and 0 < len(profile) <= 512,
                "invalid anchor identity")


def _policy(value, frame):
    require(type(value) is dict and set(value) == POLICY_FIELDS, "exact policy identity required")
    require(value["frame_id"] == frame, "capture/policy frame differs")
    for key in POLICY_FIELDS - {"session_uuid"}:
        require(_nonnegative(value[key]), "invalid policy counter")
    require(type(value["session_uuid"]) is str and 0 < len(value["session_uuid"]) <= 512,
            "invalid policy session")


def _finite_numeric(value):
    if type(value) not in (int, float, str):
        return False
    try:
        return math.isfinite(float(value))
    except (ValueError, OverflowError):
        return False


def _rows(rows, *, frame, sample, boxes):
    require(type(rows) is list and len(rows) <= MAX_ROWS, "bounded ordered target rows required")
    for row in rows:
        require(type(row) is dict and row.get("sample_id") == sample
                and str(row.get("frame_id")) == str(frame), "foreign target row")
        if boxes:
            require(row.get("label") in ("vehicle", "person"), "invalid target class")
            require(type(row.get("source_identity")) is str and row["source_identity"], "missing target source")
            require(type(row.get("contract_state")) is str and row["contract_state"], "missing contract state")
            for key in ("object_world_x", "object_world_y", "gt_bbox_x", "gt_bbox_y",
                        "gt_bbox_w", "gt_bbox_h", "gt_bbox_area_px"):
                require(_finite_numeric(row.get(key)), "missing/nonfinite localization target")


def _avo(value, frame):
    require(type(value) is dict and value.get("frame_id") == frame
            and type(value.get("frame_id")) is int, "foreign AVO frame")
    require(type(value.get("threshold")) in (int, float) and value["threshold"] == .65,
            "AVO threshold changed")
    actors = set()
    for name in ("qualified", "structural_ignored"):
        rows = value.get(name)
        require(type(rows) is list and len(rows) <= MAX_ROWS, "complete bounded AVO lists required")
        for row in rows:
            require(type(row) is dict and type(row.get("frame_id")) is int
                    and row["frame_id"] == frame, "foreign AVO row")
            actor = row.get("gt_actor_id")
            require(type(actor) in (int, str) and str(actor) and str(actor) not in actors,
                    "duplicate/missing AVO actor")
            actors.add(str(actor))
            for key in ("world_x", "world_y"):
                require(type(row.get(key)) in (int, float) and _finite_numeric(row[key]),
                        "invalid AVO coordinate")
            if name == "qualified":
                require(row.get("qualified") is True and type(row.get("avo")) in (int, float)
                        and 0 <= row["avo"] <= 1, "invalid qualified AVO row")
            else:
                require(row.get("qualified") is False and type(row.get("reasons")) is list,
                        "invalid structural AVO row")


_FIELDS = frozenset(("schema", "target_contract", "binding", "capture_identity", "policy_identity",
    "plan_sha256", "sample_id", "input_content_sha256", "history_before_sha256", "history_after_sha256",
    "provenance", "camera_inverse", "object_boxes", "object_ignore_regions", "avo", "artifact_source_sha256",
    "avo_document_sha256"))


def _metadata(value):
    require(type(value) is dict and set(value) == _FIELDS, "numerical target fields differ")
    require(value["schema"] == SCHEMA and value["target_contract"] == TARGET_CONTRACT, "target contract differs")
    _capture(value["capture_identity"])
    frame = value["capture_identity"]["frame_id"]
    _policy(value["policy_identity"], frame)
    sample = value["sample_id"]
    require(type(sample) is str and 0 < len(sample) <= 512 and re.fullmatch(r"[A-Za-z0-9_-]+", sample)
            and not sample.startswith("canonical_v3_"),
            "new live sample identity required")
    for name in ("binding", "plan_sha256", "input_content_sha256", "history_before_sha256",
                 "history_after_sha256", "artifact_source_sha256", "avo_document_sha256"):
        require(is_digest(value[name]), "required source/plan/history digest missing")
    p = value["provenance"]
    require(type(p) is dict, "complete live provenance required")
    _capture(p.get("capture_identity"))
    require(type(p) is dict and p.get("scope") == LIVE_SCOPE and p.get("schema") == INPUT_SCHEMA
            and p.get("sample_id") == sample and p.get("capture_identity") == value["capture_identity"]
            and p.get("input_content_sha256") == value["input_content_sha256"], "complete live provenance differs")
    require(type(p.get("complete")) is dict and all(p["complete"].get(k) is True for k in COMPLETENESS),
            "incomplete source declarations")
    pins = p.get("evidence_sha256")
    require(type(pins) is dict and pins and all(is_digest(v) for v in pins.values()), "source evidence pins missing")
    camera = value["camera_inverse"]
    require(type(camera) is list and len(camera) == 4 and all(type(row) is list and len(row) == 4
        and all(type(x) in (int, float) and _finite_numeric(x) for x in row) for row in camera),
        "exact finite camera inverse required")
    _rows(value["object_boxes"], frame=frame, sample=sample, boxes=True)
    _rows(value["object_ignore_regions"], frame=frame, sample=sample, boxes=False)
    _avo(value["avo"], frame)


_LABEL_TABLES = {}


def _label_table(labels):
    # Mask-lookup v1: allowed[v] is exactly ``v in labels`` for every uint8 value, so the reduction below accepts
    # and rejects exactly what ``set(np.unique(mask)).issubset(labels)`` did, without sorting the mask.
    key = frozenset(labels)
    table = _LABEL_TABLES.get(key)
    if table is None:
        table = np.array([value in key for value in range(256)], dtype=bool)
        table.setflags(write=False)
        _LABEL_TABLES[key] = table
    return table


def _mask(raw, labels):
    require(type(raw) is bytes and len(raw) == MASK_BYTES, "exact immutable uint8 mask bytes required")
    require(bool(_label_table(labels)[np.frombuffer(raw, dtype=np.uint8)].all()), "illegal target labels")


def _semantic(metadata_bytes, segmentation, object_ignore):
    # Distinct domain and length-delimited metadata; two fixed-size raw masks.
    h = hashlib.sha256(SCHEMA.encode() + b"\0" + struct.pack("!Q", len(metadata_bytes)))
    h.update(metadata_bytes); h.update(segmentation); h.update(object_ignore)
    return h.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class NumericalTarget:
    metadata: Mapping
    metadata_bytes: bytes
    segmentation_bytes: bytes
    object_ignore_bytes: bytes
    semantic_sha256: str
    _issuer: object

    def __init__(self, *args, **kwargs):
        raise ContractError("use admit_target or decode_target")

    @property
    def segmentation(self):
        return np.frombuffer(self.segmentation_bytes, dtype=np.uint8).reshape(SHAPE)

    @property
    def object_ignore(self):
        return np.frombuffer(self.object_ignore_bytes, dtype=np.uint8).reshape(SHAPE)

    @property
    def identity_bytes(self):
        return canonical({k: self.metadata[k] for k in
            ("capture_identity", "policy_identity", "plan_sha256", "binding")})


def require_admitted(target):
    require(type(target) is NumericalTarget and target._issuer is _ISSUER, "admitted numerical target required")
    return target


def admit_target(*, metadata, segmentation: bytes, object_ignore: bytes, expected_binding: str,
                 expected_capture_identity, expected_policy_identity, expected_plan_sha256: str):
    """Validate at a real boundary once; caller separately proves native inputs.

    Numerical target/provenance is not an independent proof of scene completeness.
    Those declarations must originate from the qualified exact source producer.
    """
    require(is_digest(expected_binding) and is_digest(expected_plan_sha256), "invalid expected binding")
    owned = _json_owned(metadata)
    encoded = canonical(owned)
    require(len(encoded) <= MAX_METADATA_BYTES, "target metadata too large")
    _metadata(owned)
    capture, policy = _json_owned(expected_capture_identity), _json_owned(expected_policy_identity)
    _capture(capture); _policy(policy, capture["frame_id"])
    require(owned["binding"] == expected_binding and owned["plan_sha256"] == expected_plan_sha256
        and owned["capture_identity"] == capture and owned["policy_identity"] == policy, "foreign target identity")
    _mask(segmentation, {0, 1, 2, 255}); _mask(object_ignore, {0, 255})
    target = object.__new__(NumericalTarget)
    for key, value in dict(metadata=_freeze(owned), metadata_bytes=encoded,
        segmentation_bytes=segmentation, object_ignore_bytes=object_ignore,
        semantic_sha256=_semantic(encoded, segmentation, object_ignore), _issuer=_ISSUER).items():
        object.__setattr__(target, key, value)
    return target


def encode_target(target, *, ready_raw_ns, ready_wall_ns):
    target = require_admitted(target)
    require(_nonnegative(ready_raw_ns) and ready_raw_ns > 0 and _nonnegative(ready_wall_ns)
            and ready_wall_ns > 0, "positive ready clocks required")
    meta = canonical(dict(schema=WIRE_SCHEMA, target=_parse(target.metadata_bytes),
        semantic_sha256=target.semantic_sha256, ready_raw_ns=ready_raw_ns, ready_wall_ns=ready_wall_ns))
    require(len(meta) <= MAX_METADATA_BYTES, "envelope metadata too large")
    body = meta + target.segmentation_bytes + target.object_ignore_bytes
    return HEADER.pack(MAGIC, len(meta), hashlib.sha256(body).digest()) + body


@dataclass(frozen=True, slots=True)
class DecodedTarget:
    target: NumericalTarget
    ready_raw_ns: int
    ready_wall_ns: int
    wire_sha256: str


def decode_target(payload, **expected):
    require(type(payload) is bytes and HEADER.size < len(payload) <= MAX_PACKET_BYTES,
            "bounded immutable numerical packet required")
    magic, size, digest = HEADER.unpack_from(payload)
    require(magic == MAGIC and 0 < size <= MAX_METADATA_BYTES
            and len(payload) == HEADER.size + size + 2 * MASK_BYTES, "malformed packet bounds")
    body = memoryview(payload)[HEADER.size:]
    require(hashlib.sha256(body).digest() == digest, "packet checksum differs")
    outer = _parse(bytes(body[:size]))
    require(type(outer) is dict and set(outer) == {"schema", "target", "semantic_sha256",
        "ready_raw_ns", "ready_wall_ns"} and outer["schema"] == WIRE_SCHEMA, "numerical envelope fields differ")
    for name in ("ready_raw_ns", "ready_wall_ns"):
        require(_nonnegative(outer[name]) and outer[name] > 0, "invalid ready clock")
    target = admit_target(metadata=outer["target"], segmentation=bytes(body[size:size + MASK_BYTES]),
        object_ignore=bytes(body[size + MASK_BYTES:]), **expected)
    require(target.semantic_sha256 == outer["semantic_sha256"], "semantic digest differs")
    return DecodedTarget(target, outer["ready_raw_ns"], outer["ready_wall_ns"], sha256(payload))


def bind_legacy_artifacts(target, *, prepared_content_sha256, artifact_hashes):
    """Artifact-worker operation; caller proves bytes and old numerical parity.

    Records a cross-binding, not independent certification of equivalence.
    No old hash is renamed, inferred, or required before numerical readiness.
    """
    target = require_admitted(target)
    require(is_digest(prepared_content_sha256), "original prepared-target digest required")
    hashes = _json_owned(artifact_hashes)
    names = {"segmentation.png", "object_ignore.png", "object_boxes.csv",
             "object_ignore_regions.csv", "target_manifest.csv"}
    require(type(hashes) is dict and set(hashes) == names and all(is_digest(v) for v in hashes.values()),
            "complete original artifact digests required")
    return _freeze(dict(schema=ARTIFACT_SCHEMA, semantic_sha256=target.semantic_sha256,
        original_prepared_content_sha256=prepared_content_sha256, artifacts=hashes,
        identity=_parse(target.identity_bytes)))
