"""Independent tail-output branches and create-only prediction evidence.

The operational ACK is constructed at tail-output readiness.  The map branch
and the optional research-evaluation branch receive independent immutable work
items; either branch can finish or fail without changing the ACK or the other
branch.  Prediction bytes are retained create-only with the exact identity
needed for a later, offline join to separately retained CARLA ground truth.

Importing this module creates no directory and starts no worker.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

from .operational_ack_v1 import (
    FrameActionIdentityV1,
    TailOutputAckV1,
)


PREDICTION_RECORD_SCHEMA = (
    "scenesense.splitfusion.run4b5b.prediction_evidence_record.v1"
)
PREDICTION_ENCODING = "application/vnd.scenesense.tail-output.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_CODE_RE = re.compile(r"[A-Z0-9][A-Z0-9_.:-]{0,127}")


class BranchEvidenceError(RuntimeError):
    """The branch or create-only evidence contract was violated."""


class PublicationConflict(BranchEvidenceError):
    """One logical decision was published under two different identities."""


class BranchConflict(BranchEvidenceError):
    """One branch was completed with two different outcomes."""


class CreateOnlyError(BranchEvidenceError):
    """A create-only directory or record already exists."""


class EvidenceIntegrityError(BranchEvidenceError):
    """Prediction evidence is missing, foreign, or hash-inconsistent."""


def _require(condition: bool, message: str,
             error: type[BranchEvidenceError] = BranchEvidenceError) -> None:
    if not condition:
        raise error(message)


def _sha256(value: Any, field: str) -> str:
    _require(type(value) is str and bool(_SHA256_RE.fullmatch(value)),
             f"{field} is not a lowercase SHA-256", EvidenceIntegrityError)
    return value


def _canonical(raw: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(dict(raw), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise EvidenceIntegrityError("evidence is not canonicalizable") from exc


class Branch(str, enum.Enum):
    MAP = "MAP"
    EVALUATION = "EVALUATION"


class BranchStatus(str, enum.Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class BranchWorkItemV1:
    identity: FrameActionIdentityV1
    branch: Branch
    tail_output_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.identity) is FrameActionIdentityV1,
                 "work identity must be exactly FrameActionIdentityV1")
        _require(type(self.branch) is Branch, "branch must be a Branch")
        _sha256(self.tail_output_sha256, "tail_output_sha256")


@dataclass(frozen=True, slots=True)
class TailOutputPublicationV1:
    """One immutable output fan-out; ACK construction precedes both branches."""

    ack: TailOutputAckV1
    map_work: BranchWorkItemV1
    evaluation_work: BranchWorkItemV1

    def __post_init__(self) -> None:
        _require(type(self.ack) is TailOutputAckV1,
                 "publication ACK must be TailOutputAckV1")
        _require(self.map_work.branch is Branch.MAP,
                 "map work has the wrong branch")
        _require(self.evaluation_work.branch is Branch.EVALUATION,
                 "evaluation work has the wrong branch")
        for work in (self.map_work, self.evaluation_work):
            _require(work.identity == self.ack.identity,
                     "branch identity differs from ACK identity")
            _require(work.tail_output_sha256 == self.ack.tail_output_sha256,
                     "branch output digest differs from ACK")


@dataclass(frozen=True, slots=True)
class BranchResultV1:
    identity: FrameActionIdentityV1
    branch: Branch
    status: BranchStatus
    detail_code: str
    evidence_sha256: Optional[str]

    def __post_init__(self) -> None:
        _require(type(self.identity) is FrameActionIdentityV1,
                 "result identity must be exactly FrameActionIdentityV1")
        _require(type(self.branch) is Branch, "branch must be a Branch")
        _require(type(self.status) is BranchStatus,
                 "status must be a BranchStatus")
        _require(type(self.detail_code) is str
                 and bool(_SAFE_CODE_RE.fullmatch(self.detail_code)),
                 "detail_code is empty or unsafe")
        if self.evidence_sha256 is not None:
            _sha256(self.evidence_sha256, "evidence_sha256")
        if self.status is BranchStatus.SUCCEEDED:
            _require(self.evidence_sha256 is not None,
                     "successful branch completion requires evidence SHA-256")


@dataclass(slots=True)
class _PublicationState:
    publication: TailOutputPublicationV1
    results: dict[Branch, BranchResultV1]


class TailOutputBranchCoordinatorV1:
    """Pure coordination state; callers own actual queues and workers."""

    def __init__(self) -> None:
        self._states: dict[str, _PublicationState] = {}
        self._decision_index: dict[tuple[Any, ...], str] = {}

    def publish(self, identity: FrameActionIdentityV1,
                tail_output: bytes) -> TailOutputPublicationV1:
        _require(type(identity) is FrameActionIdentityV1,
                 "identity must be exactly FrameActionIdentityV1")
        _require(type(tail_output) is bytes and len(tail_output) > 0,
                 "tail_output must be non-empty bytes")
        digest = hashlib.sha256(tail_output).hexdigest()
        exact = identity.exact_sha256()
        decision = identity.decision_key()
        indexed = self._decision_index.get(decision)
        if indexed is not None and indexed != exact:
            raise PublicationConflict(
                "logical decision already maps to a different exact identity")
        if exact in self._states:
            existing = self._states[exact].publication
            if existing.ack.tail_output_sha256 != digest:
                raise PublicationConflict(
                    "exact identity was published with different output bytes")
            raise BranchEvidenceError("tail output was already published")
        ack = TailOutputAckV1.success(identity, tail_output)
        publication = TailOutputPublicationV1(
            ack=ack,
            map_work=BranchWorkItemV1(identity, Branch.MAP, digest),
            evaluation_work=BranchWorkItemV1(identity, Branch.EVALUATION, digest),
        )
        self._states[exact] = _PublicationState(publication, {})
        self._decision_index[decision] = exact
        return publication

    def complete(self, work: BranchWorkItemV1, *, status: BranchStatus,
                 detail_code: str,
                 evidence_sha256: Optional[str]) -> BranchResultV1:
        _require(type(work) is BranchWorkItemV1,
                 "work must be exactly BranchWorkItemV1")
        state = self._states.get(work.identity.exact_sha256())
        _require(state is not None, "unknown branch work item")
        expected = (state.publication.map_work if work.branch is Branch.MAP
                    else state.publication.evaluation_work)
        _require(work == expected, "branch work item differs from publication")
        result = BranchResultV1(work.identity, work.branch, status,
                                detail_code, evidence_sha256)
        prior = state.results.get(work.branch)
        if prior is not None:
            if prior == result:
                return prior
            raise BranchConflict("branch already has a different terminal result")
        state.results[work.branch] = result
        return result

    def result(self, identity: FrameActionIdentityV1,
               branch: Branch) -> Optional[BranchResultV1]:
        state = self._states.get(identity.exact_sha256())
        _require(state is not None, "unknown publication")
        _require(state.publication.ack.identity == identity,
                 "publication identity differs")
        return state.results.get(branch)

    def ack(self, identity: FrameActionIdentityV1) -> TailOutputAckV1:
        state = self._states.get(identity.exact_sha256())
        _require(state is not None, "unknown publication")
        _require(state.publication.ack.identity == identity,
                 "publication identity differs")
        return state.publication.ack


@dataclass(frozen=True, slots=True)
class PredictionEvidenceRecordV1:
    """Metadata for an exact, opaque tail-output artifact retained locally."""

    identity: FrameActionIdentityV1
    identity_sha256: str
    postrun_join_key: Mapping[str, Any]
    prediction_encoding: str
    prediction_sha256: str
    artifact_relative_path: str
    recorded_monotonic_raw_ns: int

    def __post_init__(self) -> None:
        _require(type(self.identity) is FrameActionIdentityV1,
                 "record identity must be exactly FrameActionIdentityV1",
                 EvidenceIntegrityError)
        _sha256(self.identity_sha256, "identity_sha256")
        _require(self.identity_sha256 == self.identity.exact_sha256(),
                 "identity SHA-256 mismatch", EvidenceIntegrityError)
        _require(dict(self.postrun_join_key) == self.identity.postrun_join_key(),
                 "post-run join key differs from identity", EvidenceIntegrityError)
        _require(self.prediction_encoding == PREDICTION_ENCODING,
                 "prediction encoding drift", EvidenceIntegrityError)
        _sha256(self.prediction_sha256, "prediction_sha256")
        _require(type(self.recorded_monotonic_raw_ns) is int
                 and self.recorded_monotonic_raw_ns >= 0,
                 "recorded_monotonic_raw_ns must be nonnegative",
                 EvidenceIntegrityError)
        path = PurePosixPath(self.artifact_relative_path)
        _require(not path.is_absolute() and ".." not in path.parts
                 and path.parts == ("artifacts", f"{self.identity_sha256}.bin"),
                 "artifact path is not the registered identity path",
                 EvidenceIntegrityError)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PredictionEvidenceRecordV1":
        fields = {
            "schema", "identity", "identity_sha256", "postrun_join_key",
            "prediction_encoding", "prediction_sha256",
            "artifact_relative_path", "recorded_monotonic_raw_ns",
        }
        _require(isinstance(raw, Mapping) and set(raw) == fields,
                 "prediction record fields are incomplete or foreign",
                 EvidenceIntegrityError)
        _require(raw["schema"] == PREDICTION_RECORD_SCHEMA,
                 "prediction record schema drift", EvidenceIntegrityError)
        return cls(
            identity=FrameActionIdentityV1.from_mapping(raw["identity"]),
            identity_sha256=raw["identity_sha256"],
            postrun_join_key=raw["postrun_join_key"],
            prediction_encoding=raw["prediction_encoding"],
            prediction_sha256=raw["prediction_sha256"],
            artifact_relative_path=raw["artifact_relative_path"],
            recorded_monotonic_raw_ns=raw["recorded_monotonic_raw_ns"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PREDICTION_RECORD_SCHEMA,
            "identity": self.identity.as_dict(),
            "identity_sha256": self.identity_sha256,
            "postrun_join_key": dict(self.postrun_join_key),
            "prediction_encoding": self.prediction_encoding,
            "prediction_sha256": self.prediction_sha256,
            "artifact_relative_path": self.artifact_relative_path,
            "recorded_monotonic_raw_ns": self.recorded_monotonic_raw_ns,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical(self.as_dict()) + b"\n"


class PredictionEvidenceStoreV1:
    """Create-only per-prediction artifacts and exact join metadata."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.records = self.root / "records"
        self.artifacts = self.root / "artifacts"

    @classmethod
    def create(cls, root: Path) -> "PredictionEvidenceStoreV1":
        store = cls(root)
        try:
            store.root.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise CreateOnlyError(f"evidence root already exists: {store.root}") from exc
        store.records.mkdir()
        store.artifacts.mkdir()
        return store

    @classmethod
    def open_existing(cls, root: Path) -> "PredictionEvidenceStoreV1":
        store = cls(root)
        _require(store.root.is_dir() and store.records.is_dir()
                 and store.artifacts.is_dir(),
                 "evidence store is incomplete", EvidenceIntegrityError)
        return store

    @staticmethod
    def _write_exclusive(path: Path, payload: bytes) -> None:
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise CreateOnlyError(f"create-only evidence exists: {path}") from exc

    def write(self, publication: TailOutputPublicationV1,
              prediction_bytes: bytes,
              recorded_monotonic_raw_ns: int) -> PredictionEvidenceRecordV1:
        _require(type(publication) is TailOutputPublicationV1,
                 "publication must be TailOutputPublicationV1")
        _require(type(prediction_bytes) is bytes and len(prediction_bytes) > 0,
                 "prediction_bytes must be non-empty bytes")
        prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
        _require(prediction_sha == publication.ack.tail_output_sha256,
                 "prediction bytes differ from acknowledged tail output",
                 EvidenceIntegrityError)
        identity = publication.ack.identity
        identity_sha = identity.exact_sha256()
        artifact_rel = f"artifacts/{identity_sha}.bin"
        record = PredictionEvidenceRecordV1(
            identity=identity,
            identity_sha256=identity_sha,
            postrun_join_key=identity.postrun_join_key(),
            prediction_encoding=PREDICTION_ENCODING,
            prediction_sha256=prediction_sha,
            artifact_relative_path=artifact_rel,
            recorded_monotonic_raw_ns=recorded_monotonic_raw_ns,
        )
        artifact_path = self.root / artifact_rel
        record_path = self.records / f"{identity_sha}.json"
        if artifact_path.exists() or record_path.exists():
            raise CreateOnlyError("prediction identity already exists")
        self._write_exclusive(artifact_path, prediction_bytes)
        self._write_exclusive(record_path, record.canonical_bytes())
        return record

    def verify_all(self) -> tuple[PredictionEvidenceRecordV1, ...]:
        _require(self.root.is_dir() and self.records.is_dir()
                 and self.artifacts.is_dir(),
                 "evidence store is incomplete", EvidenceIntegrityError)
        _require(not self.root.is_symlink() and not self.records.is_symlink()
                 and not self.artifacts.is_symlink(),
                 "evidence store directories may not be symlinks",
                 EvidenceIntegrityError)
        foreign = [path for path in self.root.iterdir()
                   if path.name not in {"records", "artifacts"}]
        _require(not foreign, "foreign path in evidence root",
                 EvidenceIntegrityError)
        records: list[PredictionEvidenceRecordV1] = []
        expected_artifacts: set[str] = set()
        for path in sorted(self.records.iterdir()):
            _require(path.is_file() and not path.is_symlink()
                     and path.suffix == ".json",
                     "foreign record entry", EvidenceIntegrityError)
            try:
                raw_bytes = path.read_bytes()
                raw = json.loads(raw_bytes.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EvidenceIntegrityError("prediction record is invalid JSON") from exc
            record = PredictionEvidenceRecordV1.from_mapping(raw)
            _require(path.name == f"{record.identity_sha256}.json",
                     "record filename differs from identity",
                     EvidenceIntegrityError)
            _require(raw_bytes == record.canonical_bytes(),
                     "prediction record is not canonical",
                     EvidenceIntegrityError)
            artifact = self.root / record.artifact_relative_path
            _require(artifact.is_file() and not artifact.is_symlink(),
                     "prediction artifact is missing or a symlink",
                     EvidenceIntegrityError)
            _require(hashlib.sha256(artifact.read_bytes()).hexdigest()
                     == record.prediction_sha256,
                     "prediction artifact digest mismatch",
                     EvidenceIntegrityError)
            expected_artifacts.add(artifact.name)
            records.append(record)
        actual_artifacts = set()
        for path in self.artifacts.iterdir():
            _require(path.is_file() and not path.is_symlink()
                     and path.suffix == ".bin",
                     "foreign artifact entry", EvidenceIntegrityError)
            actual_artifacts.add(path.name)
        _require(actual_artifacts == expected_artifacts,
                 "orphan or missing prediction artifact",
                 EvidenceIntegrityError)
        return tuple(records)
