"""GT-free canonical prediction bundle for the operational edge path.

This module deliberately contains only ownership/canonicalization and a
digest-protected wire envelope.  It does not import the historical quality
scorer, ground truth, Q_perc, reward code, torch, or a live runtime.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .operational_ack_v1 import FrameActionIdentityV1


SCHEMA = "scenesense.splitfusion.run4b5b.tail_evidence.v2"
KIND = "PREDICTION"
MASK_ENCODING = "uint8-row-major-class-labels-0-1-2"
_MAGIC = b"R45P"
_FIXED = struct.Struct("!4sIQ")
_DIGEST_BYTES = 32


class TailEvidenceCodecError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TailEvidenceCodecError(message)


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TailEvidenceCodecError("metadata is not canonicalizable") from exc


def immutable_rows(
        rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Own the historical scorer's exact JSON-canonical row copy."""
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    decoded = json.loads(encoded)
    _require(isinstance(decoded, list), "object evidence is not a list")
    return tuple(dict(value) for value in decoded)


def immutable_mask(value: np.ndarray) -> np.ndarray:
    """Own the historical scorer's exact contiguous uint8 mask copy."""
    result = np.ascontiguousarray(value, dtype=np.uint8).copy()
    _require(result.ndim == 2,
             f"semantic mask must be HxW, got {result.shape}")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class TailEvidenceV2:
    identity: FrameActionIdentityV1
    objects: tuple[dict[str, Any], ...]
    semantic_mask: np.ndarray

    def __post_init__(self) -> None:
        _require(type(self.identity) is FrameActionIdentityV1,
                 "identity has a foreign type")
        rows = immutable_rows(self.objects)
        mask = immutable_mask(self.semantic_mask)
        _require(bool(np.all(np.isin(
            np.unique(mask), np.asarray([0, 1, 2], dtype=np.uint8)))),
            "semantic mask contains a class outside {0,1,2}")
        object.__setattr__(self, "objects", rows)
        object.__setattr__(self, "semantic_mask", mask)


def encode(*, identity: FrameActionIdentityV1,
           objects: Sequence[Mapping[str, Any]],
           semantic_mask: np.ndarray) -> bytes:
    owned = TailEvidenceV2(identity=identity, objects=tuple(objects),
                           semantic_mask=semantic_mask)
    mask_bytes = owned.semantic_mask.tobytes(order="C")
    header = {
        "schema": SCHEMA,
        "kind": KIND,
        "identity": owned.identity.as_dict(),
        "identity_sha256": owned.identity.exact_sha256(),
        "objects": list(owned.objects),
        "mask_encoding": MASK_ENCODING,
        "mask_shape": list(owned.semantic_mask.shape),
        "mask_nbytes": len(mask_bytes),
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
    }
    header_bytes = _canonical(header)
    body = (_FIXED.pack(_MAGIC, len(header_bytes), len(mask_bytes))
            + header_bytes + mask_bytes)
    return body + hashlib.sha256(body).digest()


def decode(payload: bytes, *,
           expected_identity: FrameActionIdentityV1 | None = None,
           ) -> TailEvidenceV2:
    _require(type(payload) is bytes
             and len(payload) >= _FIXED.size + 2 + _DIGEST_BYTES,
             "tail evidence is truncated or foreign")
    body, digest = payload[:-_DIGEST_BYTES], payload[-_DIGEST_BYTES:]
    _require(hashlib.sha256(body).digest() == digest,
             "tail evidence digest mismatch")
    magic, header_length, mask_length = _FIXED.unpack(body[:_FIXED.size])
    _require(magic == _MAGIC, "tail evidence magic drift")
    _require(len(body) == _FIXED.size + header_length + mask_length,
             "tail evidence length mismatch")
    header_bytes = body[_FIXED.size:_FIXED.size + header_length]
    mask_bytes = body[_FIXED.size + header_length:]
    try:
        raw = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TailEvidenceCodecError("tail evidence header is not JSON") from exc
    fields = {"schema", "kind", "identity", "identity_sha256", "objects",
              "mask_encoding", "mask_shape", "mask_nbytes", "mask_sha256"}
    _require(type(raw) is dict and set(raw) == fields,
             "tail evidence fields are incomplete or foreign")
    _require(_canonical(raw) == header_bytes, "tail evidence is noncanonical")
    _require(raw["schema"] == SCHEMA and raw["kind"] == KIND,
             "tail evidence schema/kind drift")
    identity = FrameActionIdentityV1.from_mapping(raw["identity"])
    _require(raw["identity_sha256"] == identity.exact_sha256(),
             "tail evidence identity digest mismatch")
    if expected_identity is not None:
        _require(identity == expected_identity,
                 "tail evidence exact identity differs")
    shape = raw["mask_shape"]
    _require(type(shape) is list and len(shape) == 2
             and all(type(value) is int and value > 0 for value in shape),
             "tail evidence mask shape is invalid")
    _require(raw["mask_encoding"] == MASK_ENCODING
             and raw["mask_nbytes"] == len(mask_bytes)
             and len(mask_bytes) == shape[0] * shape[1]
             and raw["mask_sha256"] == hashlib.sha256(mask_bytes).hexdigest(),
             "tail evidence mask metadata differs")
    _require(type(raw["objects"]) is list
             and all(type(row) is dict for row in raw["objects"]),
             "tail evidence objects are invalid")
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(tuple(shape)).copy()
    return TailEvidenceV2(identity=identity, objects=tuple(raw["objects"]),
                          semantic_mask=mask)
