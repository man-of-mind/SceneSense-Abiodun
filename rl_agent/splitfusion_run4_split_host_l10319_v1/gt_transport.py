"""Bounded, digest-checked Run-4 GT transport for the two-host seam.

This module transports the *existing* Phase-6 object JSON, semantic NPY and
semantic sidecar bytes.  It does not re-encode, reinterpret, or replace the
authoritative ``gt_evidence`` producer/reader.  A receiver may store a bundle
only after the remote edge has registered the exact reward-ticket identity.

The module deliberately does not bind, listen, connect, launch a thread, or
touch the filesystem at import time.  Live lifecycle code supplies an already
connected stream socket and an already-created per-attempt evidence directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import threading
import time
from typing import Any, Mapping, Optional
import uuid

REQUEST_SCHEMA = "scenesense.run4.split_host.gt_transfer_request.v1"
ACK_SCHEMA = "scenesense.run4.split_host.gt_transfer_ack.v1"
RECEIPT_SCHEMA = "scenesense.run4.split_host.gt_transfer_receipt.v1"
REQUEST_MAGIC = b"R4GTRQ1\n"
ACK_MAGIC = b"R4GTAK1\n"
HEADER = struct.Struct("!8sI")
COMPONENT_NAMES = ("objects.json", "semantic.npy", "semantic.json")
MAX_MANIFEST_BYTES = 16 * 1024
MAX_COMPONENT_BYTES = {
    "objects.json": 4 * 1024 * 1024,
    "semantic.npy": 4 * 1024 * 1024,
    "semantic.json": 64 * 1024,
}
MAX_BUNDLE_BYTES = sum(MAX_COMPONENT_BYTES.values())
MAX_SOCKET_TIMEOUT_S = 5.0
MAX_EXPECTATION_TIMEOUT_S = 1.0
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@+-]{1,192}$")


class GtTransportError(RuntimeError):
    """Base class for a fail-closed transport refusal."""


class ProtocolError(GtTransportError):
    pass


class IdentityError(GtTransportError):
    pass


class UnauthorizedTicketError(GtTransportError):
    pass


class DuplicateReplayError(GtTransportError):
    """An already-committed, byte-identical ticket was replayed."""


class IdentityConflictError(GtTransportError):
    pass


class StorageError(GtTransportError):
    pass


def _authoritative_gt_evidence():
    """Import the unchanged GT authority only when GT bytes are processed.

    The L10319 startup qualifier runs under the system Python before the CUDA
    container exists.  Deferring this import avoids pulling the authority's
    Torch-backed scoring dependency into host-only preparation while keeping
    every GT schema, reader, and filename decision authoritative.
    """
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence
    return gt_evidence


def _require(condition: bool, message: str,
             error: type[GtTransportError] = ProtocolError) -> None:
    if not condition:
        raise error(message)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _strict_json(payload: bytes) -> Mapping[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON") from exc
    _require(isinstance(value, Mapping), "JSON document must be an object")
    return value


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_id(value: Any, field: str) -> str:
    result = str(value)
    _require(bool(SAFE_ID_RE.fullmatch(result)), f"unsafe or empty {field}", IdentityError)
    _require(".." not in result and "/" not in result and "\\" not in result,
             f"path syntax is forbidden in {field}", IdentityError)
    return result


def _nonnegative(value: Any, field: str) -> int:
    _require(type(value) is int and value >= 0, f"{field} must be a nonnegative integer",
             IdentityError)
    return value


@dataclass(frozen=True)
class GtTransportIdentityV1:
    """Full control identity; GT's historical seven fields are a strict subset."""

    run_id: str
    cell_id: str
    stream_id: str
    session_uuid: str
    controller_lineage_sha256: str
    decision_seq: int
    ticket_seq: int
    frame_id: int
    tensor_seq: int
    capture_timestamp_ns: int
    action_id: Optional[int]
    profile_id: Optional[str]

    def __post_init__(self) -> None:
        for field in ("run_id", "cell_id", "stream_id"):
            object.__setattr__(self, field, _safe_id(getattr(self, field), field))
        try:
            canonical_uuid = str(uuid.UUID(str(self.session_uuid)))
        except (ValueError, AttributeError) as exc:
            raise IdentityError("session_uuid is not a UUID") from exc
        _require(canonical_uuid == self.session_uuid, "session_uuid is not canonical",
                 IdentityError)
        _require(bool(SHA256_RE.fullmatch(self.controller_lineage_sha256)),
                 "controller lineage SHA-256 is invalid", IdentityError)
        for field in ("decision_seq", "ticket_seq", "frame_id", "tensor_seq"):
            _nonnegative(getattr(self, field), field)
        _require(type(self.capture_timestamp_ns) is int and self.capture_timestamp_ns > 0,
                 "capture_timestamp_ns must be a positive integer", IdentityError)
        paired = (self.action_id is None, self.profile_id is None)
        _require(paired[0] == paired[1], "action_id and profile_id must both be null or present",
                 IdentityError)
        if self.action_id is not None:
            _require(type(self.action_id) is int and 0 <= self.action_id < 72,
                     "anchor action_id is outside [0,71]", IdentityError)
            object.__setattr__(self, "profile_id", _safe_id(self.profile_id, "profile_id"))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GtTransportIdentityV1":
        fields = {
            "run_id", "cell_id", "stream_id", "session_uuid",
            "controller_lineage_sha256", "decision_seq", "ticket_seq", "frame_id",
            "tensor_seq", "capture_timestamp_ns", "action_id", "profile_id",
        }
        _require(isinstance(raw, Mapping) and set(raw) == fields,
                 "transport identity fields are incomplete or foreign", IdentityError)
        return cls(**{field: raw[field] for field in fields})

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "cell_id": self.cell_id,
            "stream_id": self.stream_id, "session_uuid": self.session_uuid,
            "controller_lineage_sha256": self.controller_lineage_sha256,
            "decision_seq": self.decision_seq, "ticket_seq": self.ticket_seq,
            "frame_id": self.frame_id, "tensor_seq": self.tensor_seq,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "action_id": self.action_id, "profile_id": self.profile_id,
        }

    def gt_identity(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "cell_id": self.cell_id,
            "stream_id": self.stream_id, "frame_id": self.frame_id,
            "action_id": self.action_id, "profile_id": self.profile_id,
            "capture_timestamp_ns": self.capture_timestamp_ns,
        }

    def logical_key(self) -> tuple[Any, ...]:
        return (self.run_id, self.cell_id, self.session_uuid,
                self.controller_lineage_sha256, self.decision_seq, self.ticket_seq,
                self.frame_id, self.tensor_seq)

    def exact_digest(self) -> str:
        return _digest(_canonical(self.as_dict()))


def identity_from_phase6(*, run_id: str, cell_id: str, envelope: Any,
                         context: Any, gt_identity: Mapping[str, Any]) -> GtTransportIdentityV1:
    """Build the transport identity from the already-verified SFD4 ticket.

    The caller must invoke this only after Phase-6 frame verification.  Every
    overlapping context/envelope/GT field is checked here before an identity is
    returned.
    """
    _require(str(gt_identity.get("run_id")) == str(run_id), "GT run identity drift",
             IdentityError)
    _require(str(gt_identity.get("cell_id")) == str(cell_id), "GT cell identity drift",
             IdentityError)
    agreements = {
        "stream_id": (str(context.stream_id), str(gt_identity.get("stream_id"))),
        "frame_id": (int(envelope.frame_id), int(context.frame_id),
                     int(gt_identity.get("frame_id", -1))),
        "capture_timestamp_ns": (int(envelope.capture_timestamp_ns),
                                 int(context.capture_timestamp_ns),
                                 int(gt_identity.get("capture_timestamp_ns", -1))),
        "action_id": (envelope.anchor_action_id, gt_identity.get("action_id")),
    }
    for field, values in agreements.items():
        _require(len(set(values)) == 1, f"Phase-6 {field} identity drift", IdentityError)
    return GtTransportIdentityV1(
        run_id=str(run_id), cell_id=str(cell_id), stream_id=str(context.stream_id),
        session_uuid=str(envelope.session_uuid),
        controller_lineage_sha256=str(envelope.controller_lineage_sha256),
        decision_seq=int(envelope.decision_seq), ticket_seq=int(envelope.ticket_seq),
        frame_id=int(envelope.frame_id), tensor_seq=int(envelope.tensor_seq),
        capture_timestamp_ns=int(envelope.capture_timestamp_ns),
        action_id=gt_identity.get("action_id"), profile_id=gt_identity.get("profile_id"),
    )


@dataclass(frozen=True)
class GtBundleV1:
    identity: GtTransportIdentityV1
    components: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        _require(tuple(name for name, _ in self.components) == COMPONENT_NAMES,
                 "GT components are missing, reordered, or foreign")
        total = 0
        for name, payload in self.components:
            _require(type(payload) is bytes, f"{name} is not bytes")
            _require(0 < len(payload) <= MAX_COMPONENT_BYTES[name],
                     f"{name} exceeds its size bound")
            total += len(payload)
        _require(total <= MAX_BUNDLE_BYTES, "GT bundle exceeds its size bound")
        _validate_phase6_bytes(self.identity, dict(self.components))

    @property
    def descriptors(self) -> list[dict[str, Any]]:
        return [{"name": name, "length": len(payload), "sha256": _digest(payload)}
                for name, payload in self.components]

    def manifest_core(self) -> dict[str, Any]:
        return {"schema": REQUEST_SCHEMA, "identity": self.identity.as_dict(),
                "ground_truth_identity": self.identity.gt_identity(),
                "components": self.descriptors}

    @property
    def bundle_sha256(self) -> str:
        return _digest(_canonical(self.manifest_core()))

    def manifest(self) -> dict[str, Any]:
        return self.manifest_core() | {"bundle_sha256": self.bundle_sha256}

    def to_wire(self) -> bytes:
        manifest = _canonical(self.manifest())
        _require(len(manifest) <= MAX_MANIFEST_BYTES, "GT manifest exceeds its size bound")
        return HEADER.pack(REQUEST_MAGIC, len(manifest)) + manifest + b"".join(
            payload for _name, payload in self.components)

    @classmethod
    def from_wire(cls, packet: bytes) -> "GtBundleV1":
        _require(type(packet) is bytes and len(packet) >= HEADER.size,
                 "truncated GT request")
        magic, manifest_size = HEADER.unpack(packet[:HEADER.size])
        _require(magic == REQUEST_MAGIC, "GT request magic drift")
        _require(0 < manifest_size <= MAX_MANIFEST_BYTES, "invalid GT manifest length")
        end = HEADER.size + manifest_size
        _require(len(packet) >= end, "truncated GT manifest")
        manifest_bytes = packet[HEADER.size:end]
        manifest = _strict_json(manifest_bytes)
        _require(_canonical(manifest) == manifest_bytes, "GT manifest is not canonical")
        _require(set(manifest) == {"schema", "identity", "ground_truth_identity",
                                  "components", "bundle_sha256"},
                 "GT manifest fields are incomplete or foreign")
        _require(manifest["schema"] == REQUEST_SCHEMA, "GT request schema drift")
        identity = GtTransportIdentityV1.from_mapping(manifest["identity"])
        _require(manifest["ground_truth_identity"] == identity.gt_identity(),
                 "GT subset identity drift", IdentityError)
        descriptors = manifest["components"]
        _require(isinstance(descriptors, list) and len(descriptors) == len(COMPONENT_NAMES),
                 "GT descriptor count drift")
        offset, components = end, []
        for expected_name, descriptor in zip(COMPONENT_NAMES, descriptors):
            _require(isinstance(descriptor, Mapping)
                     and set(descriptor) == {"name", "length", "sha256"},
                     "GT descriptor fields are incomplete or foreign")
            _require(descriptor["name"] == expected_name, "GT descriptor order drift")
            length = descriptor["length"]
            _require(type(length) is int and 0 < length <= MAX_COMPONENT_BYTES[expected_name],
                     f"invalid {expected_name} length")
            digest = descriptor["sha256"]
            _require(isinstance(digest, str) and bool(SHA256_RE.fullmatch(digest)),
                     f"invalid {expected_name} digest")
            payload = packet[offset:offset + length]
            _require(len(payload) == length, f"truncated {expected_name}")
            _require(_digest(payload) == digest, f"{expected_name} digest mismatch")
            components.append((expected_name, payload))
            offset += length
        _require(offset == len(packet), "trailing bytes after GT request")
        core = {key: manifest[key] for key in (
            "schema", "identity", "ground_truth_identity", "components")}
        _require(_digest(_canonical(core)) == manifest["bundle_sha256"],
                 "GT bundle digest mismatch")
        return cls(identity=identity, components=tuple(components))


def bundle_from_phase6_paths(identity: GtTransportIdentityV1,
                             paths: Mapping[str, Path]) -> GtBundleV1:
    """Read only the three paths returned by the existing Phase-6 writers."""
    _require(set(paths) == set(COMPONENT_NAMES), "GT path set is incomplete or foreign")
    GE = _authoritative_gt_evidence()
    stem = GE._stem(identity.stream_id, identity.frame_id)
    components = []
    for name in COMPONENT_NAMES:
        path = Path(paths[name])
        _require(path.name == f"{stem}.{name}", f"unexpected Phase-6 path for {name}")
        _require(path.is_file() and not path.is_symlink(), f"missing or symlinked {name}")
        components.append((name, path.read_bytes()))
    return GtBundleV1(identity=identity, components=tuple(components))


def _validate_phase6_bytes(identity: GtTransportIdentityV1,
                           components: Mapping[str, bytes]) -> None:
    """Verify the exact existing writer formats without re-encoding them."""
    GE = _authoritative_gt_evidence()
    objects = _strict_json(components["objects.json"])
    semantic = _strict_json(components["semantic.json"])
    _require(_canonical(objects) == components["objects.json"],
             "object GT bytes are not the authoritative canonical encoding")
    _require(_canonical(semantic) == components["semantic.json"],
             "semantic sidecar bytes are not the authoritative canonical encoding")
    _require(objects.get("schema") == GE.GT_OBJECT_SCHEMA, "object GT schema drift")
    _require(semantic.get("schema") == GE.GT_SEMANTIC_SCHEMA, "semantic GT schema drift")
    expected = identity.gt_identity()
    for document, label in ((objects, "object"), (semantic, "semantic")):
        for key, value in expected.items():
            _require(document.get(key) == value, f"{label} GT {key} identity drift",
                     IdentityError)
        _require(document.get("frozen_carla_frame_id") == identity.frame_id,
                 f"{label} GT snapshot/frame drift", IdentityError)
    _require(isinstance(objects.get("objects"), list), "object GT rows are not a list")
    import numpy as np

    source = io.BytesIO(components["semantic.npy"])
    try:
        mask = np.load(source, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - normalize a hostile NPY failure
        raise ProtocolError("semantic NPY is unreadable") from exc
    _require(source.tell() == len(components["semantic.npy"]),
             "semantic NPY has trailing bytes")
    _require(mask.dtype == np.uint8 and str(mask.dtype) == semantic.get("dtype"),
             "semantic NPY dtype drift")
    _require(list(mask.shape) == semantic.get("shape"), "semantic NPY shape drift")
    _require(_digest(mask.tobytes()) == semantic.get("sha256"),
             "semantic NPY content digest drift")


@dataclass(frozen=True)
class GtAckV1:
    status: str
    identity_sha256: str
    bundle_sha256: str
    receipt_sha256: Optional[str]
    error_code: Optional[str] = None

    def __post_init__(self) -> None:
        _require(self.status in {"STORED", "DUPLICATE_IDENTICAL", "REJECTED"},
                 "invalid GT ACK status")
        for value, label in ((self.identity_sha256, "identity"),
                             (self.bundle_sha256, "bundle")):
            _require(bool(SHA256_RE.fullmatch(value)), f"invalid ACK {label} digest")
        if self.status in {"STORED", "DUPLICATE_IDENTICAL"}:
            _require(isinstance(self.receipt_sha256, str)
                     and bool(SHA256_RE.fullmatch(self.receipt_sha256)),
                     "successful GT ACK lacks a receipt digest")
            _require(self.error_code is None, "successful GT ACK carries an error")
        else:
            _require(self.receipt_sha256 is None, "rejected GT ACK carries a receipt")
            _require(bool(self.error_code and SAFE_ID_RE.fullmatch(self.error_code)),
                     "rejected GT ACK lacks a safe error code")

    def as_dict(self) -> dict[str, Any]:
        return {"schema": ACK_SCHEMA, "status": self.status,
                "identity_sha256": self.identity_sha256,
                "bundle_sha256": self.bundle_sha256,
                "receipt_sha256": self.receipt_sha256,
                "error_code": self.error_code}

    def to_wire(self) -> bytes:
        payload = _canonical(self.as_dict())
        return HEADER.pack(ACK_MAGIC, len(payload)) + payload

    @classmethod
    def from_wire(cls, packet: bytes) -> "GtAckV1":
        _require(type(packet) is bytes and len(packet) >= HEADER.size, "truncated GT ACK")
        magic, size = HEADER.unpack(packet[:HEADER.size])
        _require(magic == ACK_MAGIC and 0 < size <= MAX_MANIFEST_BYTES,
                 "invalid GT ACK header")
        _require(len(packet) == HEADER.size + size, "GT ACK length mismatch")
        payload = packet[HEADER.size:]
        raw = _strict_json(payload)
        _require(_canonical(raw) == payload, "GT ACK is not canonical")
        _require(set(raw) == {"schema", "status", "identity_sha256", "bundle_sha256",
                              "receipt_sha256", "error_code"},
                 "GT ACK fields are incomplete or foreign")
        _require(raw["schema"] == ACK_SCHEMA, "GT ACK schema drift")
        return cls(**{key: raw[key] for key in (
            "status", "identity_sha256", "bundle_sha256", "receipt_sha256", "error_code")})


class ExpectedTicketRegistryV1:
    """Edge-owned exact allow-list; unknown/future identities are never stored."""

    def __init__(self, *, run_id: str, cell_id: str, max_tickets: int = 4096) -> None:
        self.run_id = _safe_id(run_id, "run_id")
        self.cell_id = _safe_id(cell_id, "cell_id")
        _require(type(max_tickets) is int and 1 <= max_tickets <= 100_000,
                 "invalid ticket registry bound")
        self.max_tickets = max_tickets
        self._condition = threading.Condition()
        self._expected: dict[tuple[Any, ...], GtTransportIdentityV1] = {}
        self._completed: dict[tuple[Any, ...], tuple[str, str]] = {}

    def _scope(self, identity: GtTransportIdentityV1) -> None:
        _require((identity.run_id, identity.cell_id) == (self.run_id, self.cell_id),
                 "foreign run/cell identity", UnauthorizedTicketError)

    def authorize(self, identity: GtTransportIdentityV1) -> None:
        self._scope(identity)
        key = identity.logical_key()
        with self._condition:
            existing = self._expected.get(key)
            if existing is not None and existing != identity:
                raise IdentityConflictError("logical ticket identity conflict")
            _require(existing is not None or len(self._expected) < self.max_tickets,
                     "ticket registry is full", UnauthorizedTicketError)
            self._expected[key] = identity
            self._condition.notify_all()

    def await_authorized(self, identity: GtTransportIdentityV1, timeout_s: float) -> None:
        self._scope(identity)
        _require(type(timeout_s) in (int, float) and 0 <= float(timeout_s)
                 <= MAX_EXPECTATION_TIMEOUT_S, "expectation timeout is outside its bound")
        key, deadline = identity.logical_key(), time.monotonic() + float(timeout_s)
        with self._condition:
            while key not in self._expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise UnauthorizedTicketError("unknown or future reward ticket")
                self._condition.wait(remaining)
            if self._expected[key] != identity:
                raise IdentityConflictError("authorized ticket fields conflict")

    def completed(self, identity: GtTransportIdentityV1) -> Optional[tuple[str, str]]:
        with self._condition:
            return self._completed.get(identity.logical_key())

    def commit(self, identity: GtTransportIdentityV1, bundle_sha256: str,
               receipt_sha256: str) -> None:
        with self._condition:
            key = identity.logical_key()
            _require(self._expected.get(key) == identity,
                     "ticket lost authorization before commit", UnauthorizedTicketError)
            previous = self._completed.get(key)
            if previous is not None and previous != (bundle_sha256, receipt_sha256):
                raise IdentityConflictError("completed ticket digest conflict")
            self._completed[key] = (bundle_sha256, receipt_sha256)


class GtIngressStoreV1:
    """Create-only atomic installation into the edge evaluator's evidence root."""

    def __init__(self, root: Path, registry: ExpectedTicketRegistryV1) -> None:
        self.root = Path(root)
        _require(self.root.is_dir() and not self.root.is_symlink(),
                 "GT evidence root must pre-exist as a real directory", StorageError)
        self.root = self.root.resolve(strict=True)
        self.registry = registry
        self._lock = threading.Lock()

    def accept(self, bundle: GtBundleV1, *, expectation_timeout_s: float) -> GtAckV1:
        self.registry.await_authorized(bundle.identity, expectation_timeout_s)
        with self._lock:
            previous = self.registry.completed(bundle.identity)
            if previous is not None:
                if previous[0] != bundle.bundle_sha256:
                    raise IdentityConflictError("replayed ticket has conflicting bytes")
                return GtAckV1("DUPLICATE_IDENTICAL", bundle.identity.exact_digest(),
                               bundle.bundle_sha256, previous[1])
            GE = _authoritative_gt_evidence()
            stem = GE._stem(bundle.identity.stream_id, bundle.identity.frame_id)
            for name, payload in bundle.components:
                self._create_identical_or_fail(self.root / f"{stem}.{name}", payload)
            # Reuse the unchanged authoritative reader before acknowledging storage.
            GE.read_ground_truth(self.root, expected_identity=bundle.identity.gt_identity(),
                                 timeout_s=0.05)
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "identity": bundle.identity.as_dict(),
                "identity_sha256": bundle.identity.exact_digest(),
                "bundle_sha256": bundle.bundle_sha256,
                "components": bundle.descriptors,
            }
            receipt_bytes = _canonical(receipt)
            receipt_digest = _digest(receipt_bytes)
            receipt_dir = self.root / ".gt_transport_receipts"
            if receipt_dir.exists():
                _require(receipt_dir.is_dir() and not receipt_dir.is_symlink(),
                         "GT receipt directory is not a real directory", StorageError)
            else:
                receipt_dir.mkdir(mode=0o700)
                self._fsync_directory(self.root)
            receipt_path = receipt_dir / f"{bundle.identity.exact_digest()}.json"
            self._create_identical_or_fail(receipt_path, receipt_bytes)
            self.registry.commit(bundle.identity, bundle.bundle_sha256, receipt_digest)
            return GtAckV1("STORED", bundle.identity.exact_digest(),
                           bundle.bundle_sha256, receipt_digest)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _create_identical_or_fail(self, path: Path, payload: bytes) -> None:
        parent = path.parent.resolve(strict=True)
        receipt_dir = self.root / ".gt_transport_receipts"
        allowed = parent == self.root
        if receipt_dir.exists() and not receipt_dir.is_symlink():
            allowed = allowed or parent == receipt_dir.resolve(strict=True)
        _require(allowed, "GT destination escapes the evidence root", StorageError)
        if path.exists() or path.is_symlink():
            _require(path.is_file() and not path.is_symlink(),
                     "GT destination is not a regular file", StorageError)
            _require(path.read_bytes() == payload, "conflicting create-only GT evidence",
                     IdentityConflictError)
            return
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        _require(not temporary.exists() and not temporary.is_symlink(),
                 "GT temporary path already exists", StorageError)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                _require(written > 0, "short GT evidence write", StorageError)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                _require(path.is_file() and not path.is_symlink()
                         and path.read_bytes() == payload,
                         "conflicting concurrent GT evidence", IdentityConflictError)
            self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _recv_exact(stream: Any, size: int) -> bytes:
    _require(type(size) is int and size >= 0, "invalid receive length")
    chunks, remaining = [], size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise ProtocolError("stream closed during framed message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_bundle(stream: Any) -> GtBundleV1:
    """Receive one bounded request from an already-connected timed stream."""
    header = _recv_exact(stream, HEADER.size)
    magic, manifest_size = HEADER.unpack(header)
    _require(magic == REQUEST_MAGIC and 0 < manifest_size <= MAX_MANIFEST_BYTES,
             "invalid GT request header")
    manifest_bytes = _recv_exact(stream, manifest_size)
    manifest = _strict_json(manifest_bytes)
    descriptors = manifest.get("components")
    _require(isinstance(descriptors, list) and len(descriptors) == len(COMPONENT_NAMES),
             "GT descriptor count drift")
    total = 0
    for descriptor, name in zip(descriptors, COMPONENT_NAMES):
        _require(isinstance(descriptor, Mapping) and descriptor.get("name") == name,
                 "GT descriptor order drift")
        length = descriptor.get("length")
        _require(type(length) is int and 0 < length <= MAX_COMPONENT_BYTES[name],
                 f"invalid {name} length")
        total += length
    _require(total <= MAX_BUNDLE_BYTES, "GT request exceeds bundle bound")
    return GtBundleV1.from_wire(header + manifest_bytes + _recv_exact(stream, total))


def recv_ack(stream: Any) -> GtAckV1:
    header = _recv_exact(stream, HEADER.size)
    magic, size = HEADER.unpack(header)
    _require(magic == ACK_MAGIC and 0 < size <= MAX_MANIFEST_BYTES,
             "invalid GT ACK header")
    return GtAckV1.from_wire(header + _recv_exact(stream, size))


class PersistentGtSenderV1:
    """W10275 side of a caller-owned persistent connected socket."""

    def __init__(self, stream: Any, *, timeout_s: float) -> None:
        _require(type(timeout_s) in (int, float) and 0 < float(timeout_s)
                 <= MAX_SOCKET_TIMEOUT_S, "socket timeout is outside its bound")
        self.stream = stream
        self.stream.settimeout(float(timeout_s))

    def send(self, bundle: GtBundleV1) -> GtAckV1:
        self.stream.sendall(bundle.to_wire())
        ack = recv_ack(self.stream)
        _require(ack.identity_sha256 == bundle.identity.exact_digest(),
                 "GT ACK identity mismatch")
        _require(ack.bundle_sha256 == bundle.bundle_sha256,
                 "GT ACK bundle mismatch")
        if ack.status == "REJECTED":
            raise UnauthorizedTicketError(f"remote GT refusal: {ack.error_code}")
        return ack


def serve_one(stream: Any, ingress: GtIngressStoreV1, *, socket_timeout_s: float,
              expectation_timeout_s: float) -> GtAckV1:
    """L10319 side: process one request; caller owns accept/loop/teardown."""
    _require(type(socket_timeout_s) in (int, float) and 0 < float(socket_timeout_s)
             <= MAX_SOCKET_TIMEOUT_S, "socket timeout is outside its bound")
    stream.settimeout(float(socket_timeout_s))
    bundle = recv_bundle(stream)
    try:
        ack = ingress.accept(bundle, expectation_timeout_s=expectation_timeout_s)
    except UnauthorizedTicketError:
        ack = GtAckV1("REJECTED", bundle.identity.exact_digest(),
                      bundle.bundle_sha256, None, "UNKNOWN_OR_FUTURE_TICKET")
    stream.sendall(ack.to_wire())
    return ack
