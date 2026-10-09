"""Immutable numerical payload with legacy-exact string tables and a semantic digest.

The digest schema is new and distinct from the PNG/CSV prepared-target digest.
Masks are bytes-backed read-only arrays; rows are tuples of read-only mappings.
The coordinator's DTO may copy these fields; this type is not the wire format.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import struct
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v1.source_geometry import json_value
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A

from . import originals as O

SCHEMA = "splitfusion.numeric_gt_producer.payload.v1"
SHAPE = (432, 768)
MASK_BYTES = SHAPE[0] * SHAPE[1]


class PayloadError(ValueError):
    pass


def legacy_rows(rows, fields) -> list[dict[str, str]]:
    """Exact legacy CSV semantics: DictWriter(extrasaction=ignore) then DictReader."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return list(csv.DictReader(io.StringIO(stream.getvalue(), newline="")))


def legacy_csv_bytes(rows, fields) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze(v) for v in value), key=A.canonical))
    return value


def _mask_bytes(array: np.ndarray, labels) -> bytes:
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.shape != SHAPE:
        raise PayloadError("masks must be 432x768 uint8")
    raw = array.tobytes(order="C")
    counts = np.bincount(np.frombuffer(raw, np.uint8), minlength=256)
    if set(np.flatnonzero(counts).tolist()) - set(labels):
        raise PayloadError("illegal mask labels")
    return raw


@dataclass(frozen=True)
class NumericGtPayload:
    schema: str
    capture_identity: Mapping[str, Any]
    sample_id: str
    frame: Mapping[str, Any]
    camera_matrix: tuple
    camera_inverse: tuple
    intrinsics: tuple
    segmentation_bytes: bytes
    object_ignore_bytes: bytes
    object_boxes: tuple            # legacy-exact string rows, original order
    object_ignore_regions: tuple   # legacy-exact string rows, original order
    avo: Mapping[str, Any]
    provenance: Mapping[str, Any]
    input_content_sha256: str
    history_before_sha256: str
    history_after_sha256: str
    binding_sha256: str
    semantic_sha256: str

    @property
    def segmentation(self) -> np.ndarray:
        return np.frombuffer(self.segmentation_bytes, dtype=np.uint8).reshape(SHAPE)

    @property
    def object_ignore(self) -> np.ndarray:
        return np.frombuffer(self.object_ignore_bytes, dtype=np.uint8).reshape(SHAPE)

    def metadata(self) -> dict:
        """Owned JSON-compatible copy of everything except the two masks."""
        return json_value({"schema": self.schema, "capture_identity": self.capture_identity, "sample_id": self.sample_id,
            "frame": self.frame, "camera_matrix": self.camera_matrix, "camera_inverse": self.camera_inverse,
            "intrinsics": self.intrinsics, "object_boxes": self.object_boxes,
            "object_ignore_regions": self.object_ignore_regions, "avo": self.avo, "provenance": self.provenance,
            "input_content_sha256": self.input_content_sha256, "history_before_sha256": self.history_before_sha256,
            "history_after_sha256": self.history_after_sha256, "binding_sha256": self.binding_sha256})


def semantic_digest(metadata_bytes: bytes, segmentation: bytes, object_ignore: bytes) -> str:
    h = hashlib.sha256(SCHEMA.encode() + b"\0" + struct.pack("!Q", len(metadata_bytes)))
    h.update(metadata_bytes); h.update(segmentation); h.update(object_ignore)
    return h.hexdigest()


def build_payload(*, staged, target, avo) -> NumericGtPayload:
    """Assemble the immutable payload from a staged source frame, target and AVO."""
    bound = O.bound()
    builder = bound.builder
    if staged.inputs is None:
        raise PayloadError("staged frame lacks validated complete inputs")
    provenance = staged.inputs.provenance
    if avo.frame_id != staged.frame_id or avo.avo["frame_id"] != staged.frame_id:
        raise PayloadError("AVO frame differs from the staged source frame")
    segmentation = _mask_bytes(target.segmentation, (0, 1, 2, 255))
    object_ignore = _mask_bytes(target.object_ignore, (0, 255))
    boxes = legacy_rows(target.positive_rows, builder.OBJECT_FIELDS)
    regions = legacy_rows(target.ignore_rows, builder.IGNORE_FIELDS)
    for row in [*boxes, *regions]:
        if row["sample_id"] != staged.sid or row["frame_id"] != str(staged.frame_id):
            raise PayloadError("target row identity differs from the frame")
    fields = dict(schema=SCHEMA, capture_identity=freeze(staged.identity), sample_id=staged.sid,
        frame=freeze(staged.frame), camera_matrix=freeze(staged.camera_matrix.tolist()),
        camera_inverse=freeze(staged.camera_inverse.tolist()), intrinsics=freeze(avo.intrinsics.tolist()),
        segmentation_bytes=segmentation, object_ignore_bytes=object_ignore,
        object_boxes=freeze(boxes), object_ignore_regions=freeze(regions), avo=freeze(avo.avo),
        provenance=freeze(provenance), input_content_sha256=provenance["input_content_sha256"],
        history_before_sha256=staged.history_before_sha256, history_after_sha256=staged.history_after_sha256,
        binding_sha256=bound.binding_sha256, semantic_sha256="")
    draft = NumericGtPayload(**fields)
    metadata_bytes = A.canonical(draft.metadata())
    return NumericGtPayload(**{**fields, "semantic_sha256": semantic_digest(metadata_bytes, segmentation, object_ignore)})


def verify_payload(payload: NumericGtPayload) -> str:
    """Recompute and check the semantic digest and mask immutability."""
    if type(payload) is not NumericGtPayload:
        raise PayloadError("numeric payload type required")
    for name in ("segmentation_bytes", "object_ignore_bytes"):
        raw = getattr(payload, name)
        if type(raw) is not bytes or len(raw) != MASK_BYTES:
            raise PayloadError("exact immutable mask bytes required")
    if payload.segmentation.flags.writeable or payload.object_ignore.flags.writeable:
        raise PayloadError("mask views must be read-only")
    digest = semantic_digest(A.canonical(payload.metadata()), payload.segmentation_bytes, payload.object_ignore_bytes)
    if digest != payload.semantic_sha256:
        raise PayloadError("semantic digest mismatch")
    return digest
