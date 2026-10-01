"""Create-only CARLA GT evidence and exact offline-evaluation bundles.

The live policy path never reads these artifacts.  The edge retains one
prediction bundle and the CARLA host retains one ground-truth bundle under the
same :class:`FrameActionIdentityV1`.  A later offline evaluator can therefore
join the two without sending simulator ground truth over the radio or delaying
the operational acknowledgement.

The binary envelope is deliberately small and deterministic: a canonical JSON
header owns the object rows, the semantic label mask follows as raw uint8
bytes, and a SHA-256 trailer authenticates both.  It is not a pickle, NPZ, or
Torch object and importing this module performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from rl_agent.splitfusion_quality_feedback_probe_v1.scoring import (
    immutable_mask,
    immutable_rows,
)

from .operational_ack_v1 import FrameActionIdentityV1


BUNDLE_SCHEMA = "scenesense.splitfusion.run4b5b.offline_evaluation_bundle.v1"
GT_RECORD_SCHEMA = "scenesense.splitfusion.run4b5b.carla_gt_evidence_record.v1"
PREDICTION_KIND = "PREDICTION"
GROUND_TRUTH_KIND = "CARLA_GROUND_TRUTH"
MASK_ENCODING = "uint8-row-major-class-labels-0-1-2"

_MAGIC = b"R45E"
_FIXED = struct.Struct("!4sIQ")
_DIGEST_BYTES = 32
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PostRunArtifactError(RuntimeError):
    """An offline prediction/GT artifact is invalid or inconsistent."""


class GroundTruthCreateOnlyError(PostRunArtifactError):
    """A create-only CARLA evidence path already exists."""


class GroundTruthIntegrityError(PostRunArtifactError):
    """CARLA evidence is missing, foreign, or hash-inconsistent."""


def _require(condition: bool, message: str,
             error: type[PostRunArtifactError] = PostRunArtifactError) -> None:
    if not condition:
        raise error(message)


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PostRunArtifactError("artifact metadata is not canonicalizable") from exc


def _sha256(value: Any, field: str) -> str:
    _require(type(value) is str and bool(_SHA256_RE.fullmatch(value)),
             f"{field} is not a lowercase SHA-256", GroundTruthIntegrityError)
    return value


@dataclass(frozen=True, slots=True)
class EvaluationBundleV1:
    """Owned final objects and semantic class labels for one exact frame."""

    kind: str
    identity: FrameActionIdentityV1
    objects: tuple[dict[str, Any], ...]
    semantic_mask: np.ndarray

    def __post_init__(self) -> None:
        _require(self.kind in {PREDICTION_KIND, GROUND_TRUTH_KIND},
                 "bundle kind is foreign")
        _require(type(self.identity) is FrameActionIdentityV1,
                 "bundle identity must be exactly FrameActionIdentityV1")
        owned_rows = immutable_rows(self.objects)
        owned_mask = immutable_mask(self.semantic_mask)
        labels = np.unique(owned_mask)
        _require(bool(np.all(np.isin(labels, np.asarray([0, 1, 2], dtype=np.uint8)))),
                 "semantic mask contains a class outside {0,1,2}")
        object.__setattr__(self, "objects", owned_rows)
        object.__setattr__(self, "semantic_mask", owned_mask)


def encode_bundle(
    *, kind: str, identity: FrameActionIdentityV1,
    objects: Sequence[Mapping[str, Any]], semantic_mask: np.ndarray,
) -> bytes:
    """Encode one deterministic, identity-bound evaluation bundle."""

    bundle = EvaluationBundleV1(
        kind=kind,
        identity=identity,
        objects=tuple(dict(row) for row in objects),
        semantic_mask=semantic_mask,
    )
    mask_bytes = bundle.semantic_mask.tobytes(order="C")
    header = {
        "schema": BUNDLE_SCHEMA,
        "kind": bundle.kind,
        "identity": bundle.identity.as_dict(),
        "identity_sha256": bundle.identity.exact_sha256(),
        "objects": list(bundle.objects),
        "mask_encoding": MASK_ENCODING,
        "mask_shape": list(bundle.semantic_mask.shape),
        "mask_nbytes": len(mask_bytes),
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
    }
    header_bytes = _canonical(header)
    body = _FIXED.pack(_MAGIC, len(header_bytes), len(mask_bytes))
    body += header_bytes + mask_bytes
    return body + hashlib.sha256(body).digest()


def decode_bundle(payload: bytes, *, expected_kind: str | None = None,
                  expected_identity: FrameActionIdentityV1 | None = None,
                  ) -> EvaluationBundleV1:
    """Decode and authenticate an evaluation bundle without object code."""

    _require(type(payload) is bytes, "bundle payload must be exact bytes")
    _require(len(payload) >= _FIXED.size + 2 + _DIGEST_BYTES,
             "bundle payload is truncated")
    body, digest = payload[:-_DIGEST_BYTES], payload[-_DIGEST_BYTES:]
    _require(hashlib.sha256(body).digest() == digest,
             "bundle payload digest mismatch")
    magic, header_length, mask_length = _FIXED.unpack(body[:_FIXED.size])
    _require(magic == _MAGIC, "bundle magic drift")
    expected_length = _FIXED.size + header_length + mask_length
    _require(len(body) == expected_length, "bundle length mismatch")
    header_bytes = body[_FIXED.size:_FIXED.size + header_length]
    mask_bytes = body[_FIXED.size + header_length:]
    try:
        raw = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostRunArtifactError("bundle header is not JSON") from exc
    fields = {
        "schema", "kind", "identity", "identity_sha256", "objects",
        "mask_encoding", "mask_shape", "mask_nbytes", "mask_sha256",
    }
    _require(type(raw) is dict and set(raw) == fields,
             "bundle header fields are incomplete or foreign")
    _require(_canonical(raw) == header_bytes,
             "bundle header is not canonical")
    _require(raw["schema"] == BUNDLE_SCHEMA, "bundle schema drift")
    _require(raw["kind"] in {PREDICTION_KIND, GROUND_TRUTH_KIND},
             "bundle kind is foreign")
    if expected_kind is not None:
        _require(raw["kind"] == expected_kind, "bundle kind differs from expected")
    identity = FrameActionIdentityV1.from_mapping(raw["identity"])
    _require(raw["identity_sha256"] == identity.exact_sha256(),
             "bundle identity digest mismatch")
    if expected_identity is not None:
        _require(identity == expected_identity,
                 "bundle exact identity differs from expected")
    _require(raw["mask_encoding"] == MASK_ENCODING, "mask encoding drift")
    shape = raw["mask_shape"]
    _require(type(shape) is list and len(shape) == 2
             and all(type(value) is int and value > 0 for value in shape),
             "semantic mask shape is invalid")
    _require(type(raw["mask_nbytes"]) is int
             and raw["mask_nbytes"] == len(mask_bytes)
             and len(mask_bytes) == shape[0] * shape[1],
             "semantic mask byte count differs from shape")
    _require(_sha256(raw["mask_sha256"], "mask_sha256")
             == hashlib.sha256(mask_bytes).hexdigest(),
             "semantic mask digest mismatch")
    _require(type(raw["objects"]) is list
             and all(type(row) is dict for row in raw["objects"]),
             "object evidence is not a list of mappings")
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(tuple(shape)).copy()
    return EvaluationBundleV1(
        kind=raw["kind"], identity=identity,
        objects=tuple(raw["objects"]), semantic_mask=mask,
    )


@dataclass(frozen=True, slots=True)
class GroundTruthEvidenceRecordV1:
    """Metadata for one separately retained CARLA ground-truth bundle."""

    identity: FrameActionIdentityV1
    identity_sha256: str
    postrun_join_key: Mapping[str, Any]
    bundle_sha256: str
    artifact_relative_path: str
    recorded_monotonic_raw_ns: int

    def __post_init__(self) -> None:
        _require(type(self.identity) is FrameActionIdentityV1,
                 "GT identity must be exactly FrameActionIdentityV1",
                 GroundTruthIntegrityError)
        _sha256(self.identity_sha256, "identity_sha256")
        _require(self.identity_sha256 == self.identity.exact_sha256(),
                 "GT identity SHA-256 mismatch", GroundTruthIntegrityError)
        _require(dict(self.postrun_join_key) == self.identity.postrun_join_key(),
                 "GT join key differs from exact identity",
                 GroundTruthIntegrityError)
        _sha256(self.bundle_sha256, "bundle_sha256")
        _require(type(self.recorded_monotonic_raw_ns) is int
                 and self.recorded_monotonic_raw_ns >= 0,
                 "GT recorded time must be nonnegative",
                 GroundTruthIntegrityError)
        path = PurePosixPath(self.artifact_relative_path)
        _require(not path.is_absolute() and ".." not in path.parts
                 and path.parts == ("artifacts", f"{self.identity_sha256}.bin"),
                 "GT artifact path is not identity-bound",
                 GroundTruthIntegrityError)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GroundTruthEvidenceRecordV1":
        fields = {
            "schema", "identity", "identity_sha256", "postrun_join_key",
            "bundle_sha256", "artifact_relative_path",
            "recorded_monotonic_raw_ns",
        }
        _require(isinstance(raw, Mapping) and set(raw) == fields,
                 "GT record fields are incomplete or foreign",
                 GroundTruthIntegrityError)
        _require(raw["schema"] == GT_RECORD_SCHEMA, "GT record schema drift",
                 GroundTruthIntegrityError)
        return cls(
            identity=FrameActionIdentityV1.from_mapping(raw["identity"]),
            identity_sha256=raw["identity_sha256"],
            postrun_join_key=raw["postrun_join_key"],
            bundle_sha256=raw["bundle_sha256"],
            artifact_relative_path=raw["artifact_relative_path"],
            recorded_monotonic_raw_ns=raw["recorded_monotonic_raw_ns"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GT_RECORD_SCHEMA,
            "identity": self.identity.as_dict(),
            "identity_sha256": self.identity_sha256,
            "postrun_join_key": dict(self.postrun_join_key),
            "bundle_sha256": self.bundle_sha256,
            "artifact_relative_path": self.artifact_relative_path,
            "recorded_monotonic_raw_ns": self.recorded_monotonic_raw_ns,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical(self.as_dict()) + b"\n"


class GroundTruthEvidenceStoreV1:
    """Create-only, locally retained CARLA GT; never a live policy input."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.records = self.root / "records"
        self.artifacts = self.root / "artifacts"

    @classmethod
    def create(cls, root: Path) -> "GroundTruthEvidenceStoreV1":
        store = cls(root)
        try:
            store.root.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise GroundTruthCreateOnlyError(
                f"ground-truth root already exists: {store.root}"
            ) from exc
        store.records.mkdir()
        store.artifacts.mkdir()
        return store

    @classmethod
    def open_existing(cls, root: Path) -> "GroundTruthEvidenceStoreV1":
        store = cls(root)
        _require(store.root.is_dir() and store.records.is_dir()
                 and store.artifacts.is_dir(),
                 "GT store is incomplete", GroundTruthIntegrityError)
        return store

    @staticmethod
    def _write_exclusive(path: Path, payload: bytes) -> None:
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise GroundTruthCreateOnlyError(
                f"create-only ground-truth evidence exists: {path}"
            ) from exc

    def write(self, *, identity: FrameActionIdentityV1,
              eligible_objects: Sequence[Mapping[str, Any]],
              semantic_mask: np.ndarray,
              recorded_monotonic_raw_ns: int,
              ) -> GroundTruthEvidenceRecordV1:
        payload = encode_bundle(
            kind=GROUND_TRUTH_KIND, identity=identity,
            objects=eligible_objects, semantic_mask=semantic_mask,
        )
        identity_sha = identity.exact_sha256()
        artifact_rel = f"artifacts/{identity_sha}.bin"
        record = GroundTruthEvidenceRecordV1(
            identity=identity,
            identity_sha256=identity_sha,
            postrun_join_key=identity.postrun_join_key(),
            bundle_sha256=hashlib.sha256(payload).hexdigest(),
            artifact_relative_path=artifact_rel,
            recorded_monotonic_raw_ns=recorded_monotonic_raw_ns,
        )
        artifact_path = self.root / artifact_rel
        record_path = self.records / f"{identity_sha}.json"
        if artifact_path.exists() or record_path.exists():
            raise GroundTruthCreateOnlyError("ground-truth identity already exists")
        self._write_exclusive(artifact_path, payload)
        self._write_exclusive(record_path, record.canonical_bytes())
        return record

    def verify_all(self) -> tuple[GroundTruthEvidenceRecordV1, ...]:
        _require(self.root.is_dir() and self.records.is_dir()
                 and self.artifacts.is_dir(), "GT store is incomplete",
                 GroundTruthIntegrityError)
        _require(not self.root.is_symlink() and not self.records.is_symlink()
                 and not self.artifacts.is_symlink(),
                 "GT store directories may not be symlinks",
                 GroundTruthIntegrityError)
        _require(not [path for path in self.root.iterdir()
                      if path.name not in {"records", "artifacts"}],
                 "foreign path in GT store", GroundTruthIntegrityError)
        records: list[GroundTruthEvidenceRecordV1] = []
        expected_artifacts: set[str] = set()
        for path in sorted(self.records.iterdir()):
            _require(path.is_file() and not path.is_symlink()
                     and path.suffix == ".json", "foreign GT record entry",
                     GroundTruthIntegrityError)
            try:
                raw_bytes = path.read_bytes()
                raw = json.loads(raw_bytes.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GroundTruthIntegrityError("GT record is invalid JSON") from exc
            record = GroundTruthEvidenceRecordV1.from_mapping(raw)
            _require(path.name == f"{record.identity_sha256}.json",
                     "GT record filename differs from identity",
                     GroundTruthIntegrityError)
            _require(raw_bytes == record.canonical_bytes(),
                     "GT record is not canonical", GroundTruthIntegrityError)
            artifact = self.root / record.artifact_relative_path
            _require(artifact.is_file() and not artifact.is_symlink(),
                     "GT artifact is missing or a symlink",
                     GroundTruthIntegrityError)
            payload = artifact.read_bytes()
            _require(hashlib.sha256(payload).hexdigest() == record.bundle_sha256,
                     "GT artifact digest mismatch", GroundTruthIntegrityError)
            decode_bundle(payload, expected_kind=GROUND_TRUTH_KIND,
                          expected_identity=record.identity)
            expected_artifacts.add(artifact.name)
            records.append(record)
        actual_artifacts = set()
        for path in self.artifacts.iterdir():
            _require(path.is_file() and not path.is_symlink()
                     and path.suffix == ".bin", "foreign GT artifact entry",
                     GroundTruthIntegrityError)
            actual_artifacts.add(path.name)
        _require(actual_artifacts == expected_artifacts,
                 "orphan or missing GT artifact", GroundTruthIntegrityError)
        return tuple(records)


__all__ = [
    "BUNDLE_SCHEMA", "GT_RECORD_SCHEMA", "PREDICTION_KIND",
    "GROUND_TRUTH_KIND", "MASK_ENCODING", "PostRunArtifactError",
    "GroundTruthCreateOnlyError", "GroundTruthIntegrityError",
    "EvaluationBundleV1", "encode_bundle", "decode_bundle",
    "GroundTruthEvidenceRecordV1", "GroundTruthEvidenceStoreV1",
]
