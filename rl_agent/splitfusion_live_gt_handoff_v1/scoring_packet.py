"""Bounded exact prepared-target/AVO wire values; no persistence or timers.

The immutable packet is an evaluator-only handoff, not durable evidence and
not a native-input completeness proof. Original numerical validators remain
authoritative. The caller owns archival and the unchanged action-open clock.
"""
from dataclasses import replace
import hashlib
import json
import math
import struct

import cv2
import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG

SCHEMA = "splitfusion.immutable_gt_scoring_packet.v1"
MAGIC = b"SFGTHP1\0"
HEADER = struct.Struct("!8sQ32s")
MAX_PACKET_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
BLOBS = ("segmentation.png", "object_ignore.png", "avo.json")


class PacketError(ValueError):
    pass


def require(value, message):
    if not value:
        raise PacketError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def validate_binding(binding):
    require(type(binding) is str and T._digest(binding), "64-hex run binding required")
    return binding


def _json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    try:
        return json.loads(payload, object_pairs_hook=pairs,
            parse_float=number,
            parse_constant=lambda _: (_ for _ in ()).throw(PacketError("nonfinite JSON")))
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise PacketError("invalid packet JSON") from exc


def _clock(value):
    require(type(value) is int and 0 < value < 2**63, "positive bounded ready clock required")


def identity_key(identity):
    return digest(T._canonical(T.validate_identity(identity)))


def _avo(document, identity):
    require(isinstance(document, dict) and document.get("schema") == LG.SCHEMA
        and document.get("mask_contract") == LG.MASK_CONTRACT
        and document.get("record_kind") == "AVO_SCORED_FRAME", "complete scored AVO document required")
    require(T.validate_identity(document.get("identity")) == identity
        and type(document.get("source_frame_id")) is int
        and document["source_frame_id"] == identity["frame_id"], "foreign AVO source identity")
    _clock(document.get("ready_wall_ns"))
    value = document.get("avo")
    require(isinstance(value, dict) and type(value.get("frame_id")) is int
        and value["frame_id"] == identity["frame_id"]
        and type(value.get("threshold")) in (int, float) and value["threshold"] == .65,
        "AVO frame/threshold differs")
    seen = set()
    for name in ("qualified", "structural_ignored"):
        rows = value.get(name)
        require(type(rows) is list and len(rows) <= 10000, "bounded complete AVO actor lists required")
        for row in rows:
            require(isinstance(row, dict) and row.get("frame_id") == identity["frame_id"]
                and type(row.get("gt_actor_id")) in (str, int), "foreign/malformed AVO actor")
            actor = str(row["gt_actor_id"])
            require(actor and actor not in seen, "duplicate/empty AVO actor identity")
            seen.add(actor)
            for field in ("world_x", "world_y"):
                require(type(row.get(field)) in (int, float), "AVO actor coordinate missing")
            if name == "qualified":
                require(row.get("qualified") is True and type(row.get("avo")) in (int, float)
                    and 0 <= row["avo"] <= 1, "invalid qualified AVO value")
            else:
                require(row.get("qualified") is False and type(row.get("reasons")) is list,
                    "invalid structural AVO ignore")


def _provenance(target, identity, sample_id):
    require(isinstance(target, dict) and target.get("live_qualified") is False,
        "unqualified complete target required")
    provenance = target.get("provenance", {})
    require(isinstance(provenance, dict) and provenance.get("scope") == T.LIVE_SCOPE
        and provenance.get("capture_identity") == identity
        and provenance.get("sample_id") == sample_id
        and not sample_id.startswith("canonical_v3_"), "foreign/historical target provenance")
    require(T._digest(provenance.get("input_content_sha256")), "native input digest missing")
    evidence = provenance.get("evidence_sha256")
    require(isinstance(evidence, dict) and evidence and all(T._digest(v) for v in evidence.values()),
        "source evidence bindings missing")


def _input_metadata(value, identity, target, sample_id):
    require(type(value) is dict, "complete native input metadata required")
    T._validate_metadata(value, identity)
    require(value["frame"]["sample_id"] == sample_id
        and T._canonical(value["provenance"]) == T._canonical(target["provenance"]),
        "prepared/native metadata provenance differs")
    inverse = value["frame"].get("camera_inverse_matrix_json")
    if inverse is not None:
        require(type(inverse) is str, "native camera inverse JSON must be a string")
        matrix = _json(inverse)
        require(type(matrix) is list and len(matrix) == 4
            and all(type(row) is list and len(row) == 4
                and all(type(x) in (int, float) and math.isfinite(x) for x in row) for row in matrix),
            "invalid native camera inverse metadata")


def encode_packet(*, identity, prepared, avo_document, input_metadata,
                  ready_raw_ns, ready_wall_ns, binding):
    """Serialize exact existing PNGs, tables and full AVO once; no raw sources."""
    identity = T.validate_identity(identity)
    validate_binding(binding)
    _clock(ready_raw_ns)
    _clock(ready_wall_ns)
    owned = E.restore_prepared_target(frame_id=prepared.frame_id, sample_id=prepared.sample_id,
        target=prepared.target, content_sha256=prepared.content_sha256)
    require(owned.frame_id == identity["frame_id"], "foreign prepared frame")
    _provenance(owned.target, identity, owned.sample_id)
    _input_metadata(input_metadata, identity, owned.target, owned.sample_id)
    # Roundtrip canonical JSON before validation rejects nonfinite values and
    # caller-owned non-JSON data; the entire original document is retained.
    avo_bytes = T._canonical(avo_document)
    _avo(_json(avo_bytes), identity)
    target = {k: v for k, v in owned.target.items() if k not in
        ("segmentation", "object_ignore", "segmentation_png", "object_ignore_png")}
    blobs = (owned.target["segmentation_png"], owned.target["object_ignore_png"], avo_bytes)
    metadata = dict(schema=SCHEMA, binding=binding, identity=identity,
        frame_id=owned.frame_id, sample_id=owned.sample_id, content_sha256=owned.content_sha256,
        input_content_sha256=input_metadata["provenance"]["input_content_sha256"],
        input_metadata=input_metadata,
        ready_raw_ns=ready_raw_ns, ready_wall_ns=ready_wall_ns, target=target,
        blobs={name: dict(bytes=len(value), sha256=digest(value)) for name, value in zip(BLOBS, blobs)})
    encoded = T._canonical(metadata)
    require(len(encoded) <= MAX_METADATA_BYTES, "packet metadata limit exceeded")
    require(HEADER.size + len(encoded) + sum(map(len, blobs)) <= MAX_PACKET_BYTES,
        "scoring packet size limit exceeded")
    body = b"".join((encoded, *blobs))
    return HEADER.pack(MAGIC, len(encoded), hashlib.sha256(body).digest()) + body


def _unpack(payload, *, binding, expected_identity=None):
    validate_binding(binding)
    require(type(payload) is bytes and HEADER.size < len(payload) <= MAX_PACKET_BYTES,
        "bounded immutable packet bytes required")
    magic, size, checksum = HEADER.unpack_from(payload)
    require(magic == MAGIC and 0 < size <= MAX_METADATA_BYTES and HEADER.size + size <= len(payload),
        "invalid packet header")
    body = memoryview(payload)[HEADER.size:]
    require(hashlib.sha256(body).digest() == checksum, "packet checksum differs")
    metadata = _json(bytes(body[:size]))
    fields = {"schema", "binding", "identity", "frame_id", "sample_id", "content_sha256",
              "input_metadata", "input_content_sha256",
              "ready_raw_ns", "ready_wall_ns", "target", "blobs"}
    require(type(metadata) is dict and set(metadata) == fields and metadata["schema"] == SCHEMA,
        "packet schema differs")
    require(metadata["binding"] == binding, "foreign packet run binding")
    identity = T.validate_identity(metadata["identity"])
    if expected_identity is not None:
        require(identity == T.validate_identity(expected_identity), "foreign packet identity")
    require(type(metadata["frame_id"]) is int and metadata["frame_id"] == identity["frame_id"]
        and type(metadata["sample_id"]) is str and metadata["sample_id"]
        and T._digest(metadata["content_sha256"]), "invalid prepared identity/digest")
    _clock(metadata["ready_raw_ns"])
    _clock(metadata["ready_wall_ns"])
    _provenance(metadata["target"], identity, metadata["sample_id"])
    _input_metadata(metadata["input_metadata"], identity, metadata["target"], metadata["sample_id"])
    require(T._digest(metadata["input_content_sha256"])
        and metadata["input_content_sha256"] == metadata["input_metadata"]["provenance"]["input_content_sha256"],
        "native input content binding differs")
    require(type(metadata["blobs"]) is dict and set(metadata["blobs"]) == set(BLOBS),
        "packet blob inventory differs")
    offset, blobs = size, {}
    for name in BLOBS:
        item = metadata["blobs"][name]
        require(type(item) is dict and set(item) == {"bytes", "sha256"}
            and type(item["bytes"]) is int and 0 < item["bytes"] <= MAX_PACKET_BYTES
            and T._digest(item["sha256"]), "invalid packet blob binding")
        end = offset + item["bytes"]
        require(end <= len(body), "truncated packet blob")
        raw = bytes(body[offset:end])
        require(digest(raw) == item["sha256"], "packet blob checksum differs")
        blobs[name], offset = raw, end
    require(offset == len(body), "unexpected trailing packet bytes")
    avo = _json(blobs["avo.json"])
    _avo(avo, identity)
    return metadata, blobs, avo


def inspect_packet(payload, *, binding, expected_identity=None):
    """Validate bounded wire integrity/identity without numerical PNG decoding."""
    metadata, _blobs, _avo_document = _unpack(payload, binding=binding, expected_identity=expected_identity)
    return dict(identity=metadata["identity"], binding=binding, packet_sha256=digest(payload),
        bytes=len(payload), ready_raw_ns=metadata["ready_raw_ns"], ready_wall_ns=metadata["ready_wall_ns"])


def decode_packet(payload, *, expected_identity, binding):
    """Restore original evaluator inputs; original validators run before use."""
    metadata, blobs, avo = _unpack(payload, binding=binding, expected_identity=expected_identity)
    target = metadata["target"]
    for name in ("segmentation", "object_ignore"):
        raw = blobs[name + ".png"]
        # Reject enormous/deceptive PNG dimensions before native allocation.
        require(len(raw) >= 33 and raw[:8] == b"\x89PNG\r\n\x1a\n"
            and raw[12:16] == b"IHDR" and struct.unpack("!II", raw[16:24]) == (768, 432),
            "prepared PNG geometry differs")
        target[name + "_png"] = raw
        target[name] = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    prepared = E.restore_prepared_target(frame_id=metadata["frame_id"], sample_id=metadata["sample_id"],
        target=target, content_sha256=metadata["content_sha256"])
    # Strengthen ownership for the delivered arrays, beyond a reversible flag.
    own = dict(prepared.target)
    for name in ("segmentation", "object_ignore"):
        value = own[name]
        own[name] = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    prepared = replace(prepared, target=own)
    return dict(prepared=prepared, manifest_sha256=digest(payload), avo=avo["avo"],
        avo_document=avo, avo_sidecar_sha256=digest(blobs["avo.json"]),
        avo_ready_wall_ns=avo["ready_wall_ns"], avo_measurement_semantics=LG.MEASUREMENT_SEMANTICS,
        gt_ready_raw_ns=metadata["ready_raw_ns"],
        gt_ready_wall_ns=max(metadata["ready_wall_ns"], avo["ready_wall_ns"]),
        input_metadata=metadata["input_metadata"], input_content_sha256=metadata["input_content_sha256"],
        scoring_packet_schema=SCHEMA, binding=binding)
