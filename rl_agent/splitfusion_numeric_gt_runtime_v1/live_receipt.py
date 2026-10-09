"""Exact durable receipt and postreceipt evidence for a scored live archive.

The ArchiveClient writes the immutable archive batch. This module checks that
child's authenticated COMPLETE receipt against every live archive byte, observes
the client-side submission/completion on CLOCK_MONOTONIC_RAW, and durably creates
a separate completion sidecar. It never stamps READY or reconstructs old AVO
pre-score chronology.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import time
from types import MappingProxyType

from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient, MAX_FILES
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C

from .live_archive import LiveArchive, verify_live_archive
from .live_clock import (ASYNC_COMPLETION_SCHEMA, ASYNC_HANDOFF_ID, ASYNC_SCHEMA,
                         CLOCK_DOMAIN, COMPLETION_SCHEMA, SCHEMA as CLOCK_SCHEMA,
                         read_boot_id)


SCHEMA = "splitfusion.numeric_gt.live_receipt.v1"
PROOF_SCHEMA = "splitfusion.numeric_gt.live_completion_durability_proof.v1"
SIDECAR_NAME = "live_archive_completion.json"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


class LiveReceiptError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise LiveReceiptError(message)


def _prefix(value):
    _require(type(value) is str and value.endswith("/") and "\\" not in value
             and "\x00" not in value, "exact relative archive prefix required")
    path = PurePosixPath(value[:-1])
    _require(bool(path.parts) and not path.is_absolute() and str(path) + "/" == value
             and all(part not in (".", "..") for part in path.parts),
             "safe normalized relative archive prefix required")
    return value


def _relative(value):
    _require(type(value) is str and value and "\\" not in value and "\x00" not in value,
             "safe relative archive file required")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and str(path) == value
             and all(part not in (".", "..") for part in path.parts),
             "safe normalized relative archive file required")
    return value


def _basename(value):
    _require(type(value) is str and value not in ("", ".", "..") and len(value) <= 255
             and "/" not in value and "\\" not in value and "\x00" not in value,
             "safe flat sidecar name required")
    return value


def _archive_binding(archive):
    _require(type(archive) is LiveArchive, "exact live archive required")
    verify_live_archive(archive)
    manifest = C._parse(archive.manifest_bytes)
    _require(manifest.get("historical_avo_document") is False
             and manifest.get("old_prepared_digest_claim") is False
             and manifest.get("ready_manifest_claim") is False,
             "historical AVO or READY claim forbidden in live archive")
    return C._parse(archive.ready_identity_bytes)["capture_identity"]


def archive_from_scorer_reply(key: bytes, reply: dict) -> LiveArchive:
    """Rebuild the scorer's immutable archive from exact primitive IPC values."""
    _require(type(key) is bytes and len(key) <= C.MAX_METADATA_BYTES
             and C.canonical(C._parse(key)) == key, "canonical full scorer key required")
    _require(type(reply) is dict and set(reply) == {
        "files", "sha256", "archive_manifest_sha256", "ready_semantic_sha256",
        "identity_sha256", "terminal", "clock_descriptor_bytes", "live_qualified"},
        "exact live scorer archive reply required")
    files = reply["files"]
    _require(type(files) is dict and type(reply["sha256"]) is dict
             and set(reply["sha256"]) == set(files)
             and all(type(name) is str and type(raw) is bytes for name, raw in files.items())
             and all(reply["sha256"][name] == C.sha256(raw) for name, raw in files.items()),
             "scorer archive file digest inventory differs")
    _require(reply["identity_sha256"] == C.sha256(key)
             and C.is_digest(reply["ready_semantic_sha256"])
             and type(reply["clock_descriptor_bytes"]) is bytes
             and reply["live_qualified"] is False,
             "foreign scorer identity, semantic or live qualification")
    manifest_bytes = files.get("live_archive.json")
    _require(type(manifest_bytes) is bytes
             and reply["archive_manifest_sha256"] == C.sha256(manifest_bytes),
             "exact live archive manifest bytes required")
    archive = LiveArchive(MappingProxyType(dict(files)), manifest_bytes,
        reply["archive_manifest_sha256"], key, reply["ready_semantic_sha256"],
        reply["clock_descriptor_bytes"])
    _archive_binding(archive)
    clock = C._parse(archive.clock_descriptor_bytes)
    _require(reply["terminal"] == clock.get("deadline", {}).get("terminal"),
             "scorer live terminal differs from its observed clock descriptor")
    return archive


def _expected(archive, relative_prefix, root_index):
    prefix = _prefix(relative_prefix)
    _require(type(root_index) is int and root_index >= 0, "explicit archive root index required")
    return {prefix + name: dict(root=root_index, relative=prefix + name,
                                sha256=C.sha256(raw), bytes=len(raw), ready=False)
            for name, raw in archive.files.items()}


def _check_rows(rows, expected):
    _require(type(rows) is list and 0 < len(rows) <= MAX_FILES,
             "bounded archive file inventory required")
    seen = {}
    for row in rows:
        _require(type(row) is dict and set(row) == {"root", "relative", "sha256", "bytes", "ready"},
                 "exact archive file receipt fields required")
        relative = _relative(row["relative"])
        _require(relative not in seen and type(row["root"]) is int and row["root"] >= 0
                 and C.is_digest(row["sha256"]) and type(row["bytes"]) is int
                 and row["bytes"] >= 0 and row["ready"] is False,
                 "duplicate, invalid or READY archive file receipt")
        seen[relative] = row
    prefix = next(iter(expected)).rsplit("/", 1)[0] + "/"
    selected = {name: row for name, row in seen.items() if name.startswith(prefix)}
    _require(selected == expected, "live archive receipt has missing, extra or changed files under prefix")
    return selected


def _open_directory(path, *, owned):
    """Walk every component with O_NOFOLLOW, including an owned final directory."""
    path = Path(path)
    _require(path.is_absolute() and ".." not in path.parts, "explicit absolute directory required")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if owned:
            _require(os.fstat(descriptor).st_uid == os.getuid(),
                     "sidecar directory must belong to this user")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _readback(archive_root, selected, relative_prefix):
    root = _open_directory(archive_root, owned=False)
    directory = root
    try:
        for part in PurePosixPath(relative_prefix[:-1]).parts:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory)
            if directory != root:
                os.close(directory)
            directory = child
        for relative, row in selected.items():
            name = relative[len(relative_prefix):]
            _basename(name)
            descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory)
            try:
                info = os.fstat(descriptor)
                _require(stat.S_ISREG(info.st_mode) and info.st_size == row["bytes"],
                         "durable archive readback type/size differs: " + name)
                digest = hashlib.sha256()
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                _require(digest.hexdigest() == row["sha256"],
                         "durable archive readback digest differs: " + name)
            finally:
                os.close(descriptor)
    finally:
        if directory != root:
            os.close(directory)
        os.close(root)


def verify_live_receipt(archive: LiveArchive, receipt: dict, *, relative_prefix: str,
                        root_index: int, archive_root: Path) -> dict:
    """Check the complete receipt and symlink-safe readback for this live archive."""
    capture = _archive_binding(archive)
    expected = _expected(archive, relative_prefix, root_index)
    value = C._json_owned(receipt)
    _require(type(value) is dict and value.get("status") == "COMPLETE"
             and value.get("durable") is True and value.get("error") is None
             and value.get("identity") == capture
             and type(value.get("job_id")) is int and value["job_id"] > 0
             and type(value.get("pid")) is int and value["pid"] > 0
             and value.get("finalized_ready") == [],
             "foreign, incomplete or READY archive receipt")
    selected = _check_rows(value.get("written"), expected)
    _readback(archive_root, selected, relative_prefix)
    return dict(schema=SCHEMA, job_id=value["job_id"], root_index=root_index,
                relative_prefix=relative_prefix, archive_manifest_sha256=archive.manifest_sha256,
                files={name: selected[name] for name in sorted(selected)},
                archive_receipt_sha256=C.sha256(C.canonical(value)))


def _client_root(client, root_index):
    _require(type(client) is ArchiveClient and not client.failed and client.process is not None,
             "healthy actual ArchiveClient required")
    _require(type(root_index) is int and 0 <= root_index < len(client.config["roots"]),
             "configured archive root index required")
    return Path(client.config["roots"][root_index]["path"])


def archive_files_for_submit(client: ArchiveClient, archive: LiveArchive, *,
                             relative_prefix: str, root_index: int) -> dict:
    """Return exact non-READY paths for one ArchiveClient.submit call."""
    _archive_binding(archive)
    root = _client_root(client, root_index)
    _expected(archive, relative_prefix, root_index)
    return {root / relative_prefix / name: raw for name, raw in archive.files.items()}


def _client_event(archive, name):
    return dict(role="archive_client", name=name,
        raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        pid=os.getpid(), boot_id=read_boot_id(), clock_domain=CLOCK_DOMAIN,
        ready_identity=C._parse(archive.ready_identity_bytes),
        ready_identity_sha256=C.sha256(archive.ready_identity_bytes),
        semantic_sha256=archive.semantic_sha256)


def _check_event(archive, event, name):
    value = C._json_owned(event)
    _require(type(value) is dict and set(value) == {
        "role", "name", "raw_ns", "pid", "boot_id", "clock_domain",
        "ready_identity", "ready_identity_sha256", "semantic_sha256"}
        and value["role"] == "archive_client" and value["name"] == name
        and value["clock_domain"] == CLOCK_DOMAIN
        and type(value["raw_ns"]) is int and value["raw_ns"] > 0
        and type(value["pid"]) is int and value["pid"] > 0
        and value["ready_identity"] == C._parse(archive.ready_identity_bytes)
        and value["ready_identity_sha256"] == C.sha256(archive.ready_identity_bytes)
        and value["semantic_sha256"] == archive.semantic_sha256,
        "foreign or incomplete archive client event")
    return value


def _completion_version(archive):
    """Require the descriptor's exact registered clock/handoff pair."""
    descriptor = C._parse(archive.clock_descriptor_bytes)
    if descriptor.get("schema") == CLOCK_SCHEMA and "handoff_id" not in descriptor:
        return COMPLETION_SCHEMA, None
    if (descriptor.get("schema") == ASYNC_SCHEMA
            and descriptor.get("handoff_id") == ASYNC_HANDOFF_ID):
        return ASYNC_COMPLETION_SCHEMA, ASYNC_HANDOFF_ID
    raise LiveReceiptError("mixed or unknown live clock descriptor version")


def observe_archive_submission(client: ArchiveClient, job_id: int, archive: LiveArchive,
                               *, relative_prefix: str, root_index: int) -> dict:
    """Return owner-stamped primitive evidence just after ArchiveClient.submit."""
    _archive_binding(archive)
    _client_root(client, root_index)
    expected = _expected(archive, relative_prefix, root_index)
    _require(type(job_id) is int and job_id in client.pending
             and 0 < job_id <= len(client.submissions),
             "actual pending ArchiveClient job required")
    submission = client.submissions[job_id - 1]
    _require(submission["job_id"] == job_id
             and submission["ownership"] == "SEALED_KERNEL_TRANSFER_NOT_DURABLE"
             and submission["identity"] == C._parse(archive.ready_identity_bytes)["capture_identity"],
             "foreign or untransferred live archive submission")
    _check_rows(submission["files"], expected)
    _require(len(submission["files"]) == len(expected),
             "live submission contains files outside its exact archive")
    return _client_event(archive, "archive_submitted")


def _authenticated_completion(client, receipt, archive, relative_prefix, root_index):
    root = _client_root(client, root_index)
    _require(type(receipt) is dict and type(receipt.get("job_id")) is int
             and receipt in client.completions and receipt.get("pid") == client.process.pid,
             "receipt was not authenticated by this ArchiveClient")
    job_id = receipt["job_id"]
    _require(0 < job_id <= len(client.submissions)
             and client.submissions[job_id - 1]["job_id"] == job_id,
             "foreign archive submission job")
    expected = _expected(archive, relative_prefix, root_index)
    _check_rows(client.submissions[job_id - 1]["files"], expected)
    _require(len(client.submissions[job_id - 1]["files"]) == len(expected),
             "authenticated submission has files outside the live archive")
    return verify_live_receipt(archive, receipt, relative_prefix=relative_prefix,
                               root_index=root_index, archive_root=root)


def observe_durable_completion(client: ArchiveClient, receipt: dict, archive: LiveArchive,
                               *, relative_prefix: str, root_index: int) -> dict:
    """Return owner-stamped primitive evidence after exact COMPLETE verification."""
    _archive_binding(archive)
    _authenticated_completion(client, receipt, archive, relative_prefix, root_index)
    return _client_event(archive, "durable_completion_observed")


def _write_sidecar(directory_fd, name, data):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=directory_fd)
    try:
        view = memoryview(data)
        position = 0
        while position < len(view):
            written = os.write(descriptor, view[position:])
            _require(written > 0, "completion sidecar write stalled")
            position += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def persist_live_completion(client: ArchiveClient, receipt: dict, archive: LiveArchive,
                            scorer_confirmation: dict, archive_submitted_event: dict,
                            durable_completion_event: dict, *, relative_prefix: str,
                            root_index: int, sidecar_directory: Path,
                            sidecar_name: str = SIDECAR_NAME) -> dict:
    """Fsync the scorer's authoritative clock completion in an owned directory."""
    _archive_binding(archive)
    _basename(sidecar_name)
    verified = _authenticated_completion(client, receipt, archive, relative_prefix, root_index)
    submit = _check_event(archive, archive_submitted_event, "archive_submitted")
    completed = _check_event(archive, durable_completion_event, "durable_completion_observed")
    _require((submit["boot_id"], submit["pid"]) == (completed["boot_id"], completed["pid"])
             and (submit["boot_id"], submit["pid"]) == (read_boot_id(), os.getpid())
             and submit["raw_ns"] <= completed["raw_ns"],
             "archive client event owner or RAW order changed")
    _require(type(scorer_confirmation) is dict and set(scorer_confirmation) == {
        "status", "terminal", "archive_manifest_sha256", "clock_completion_bytes",
        "clock_completion_sha256", "live_qualified"},
        "exact scorer durable confirmation required")
    completion_bytes = scorer_confirmation["clock_completion_bytes"]
    _require(scorer_confirmation["status"] == "COMPLETION_SIDECAR_PENDING_DURABILITY"
             and scorer_confirmation["live_qualified"] is False
             and scorer_confirmation["archive_manifest_sha256"] == archive.manifest_sha256
             and type(completion_bytes) is bytes
             and C.sha256(completion_bytes) == scorer_confirmation["clock_completion_sha256"],
             "foreign or unverified scorer completion bytes")
    completion = C._parse(completion_bytes)
    completion_schema, handoff_id = _completion_version(archive)
    _require(C.canonical(completion) == completion_bytes
             and completion.get("schema") == completion_schema
             and (("handoff_id" not in completion) if handoff_id is None else
                  completion.get("handoff_id") == handoff_id)
             and completion.get("ready_identity") == C._parse(archive.ready_identity_bytes)
             and completion.get("ready_identity_sha256") == C.sha256(archive.ready_identity_bytes)
             and completion.get("semantic_sha256") == archive.semantic_sha256
             and completion.get("clock_descriptor_sha256") == C.sha256(archive.clock_descriptor_bytes)
             and completion.get("archive_receipt_sha256") == verified["archive_receipt_sha256"]
             and completion.get("archive_scope") == "SCORER_NUMERIC_ARCHIVE"
             and completion.get("archive_submit_event") == submit
             and completion.get("archive_completion_event") == completed
             and completion.get("receipt_status") == "COMPLETE"
             and completion.get("durable") is True
             and completion.get("written_files") == len(receipt["written"]),
             "scorer completion does not bind exact live receipt and owner events")
    clock = C._parse(archive.clock_descriptor_bytes)
    _require(scorer_confirmation["terminal"] == clock["deadline"]["terminal"],
             "scorer terminal differs from archived clock")
    directory = _open_directory(sidecar_directory, owned=True)
    try:
        try:
            os.stat(sidecar_name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise LiveReceiptError("completion sidecar must be create-only")
        document = dict(verified, archive_receipt=C._json_owned(receipt),
                        clock_completion=completion)
        data = C.canonical(document)
        _require(len(data) <= C.MAX_METADATA_BYTES, "bounded live receipt sidecar required")
        _write_sidecar(directory, sidecar_name, data)
        descriptor = os.open(sidecar_name, _FILE_FLAGS, dir_fd=directory)
        try:
            info = os.fstat(descriptor)
            _require(stat.S_ISREG(info.st_mode) and info.st_size == len(data),
                     "durable sidecar readback size/type differs")
            chunks = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            readback = b"".join(chunks)
            _require(readback == data,
                     "durable sidecar readback digest differs")
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)
    return dict(path=str(Path(sidecar_directory) / sidecar_name), bytes=len(data),
                sha256=C.sha256(readback), job_id=receipt["job_id"], schema=PROOF_SCHEMA,
                receipt_sha256=verified["archive_receipt_sha256"],
                archive_manifest_sha256=archive.manifest_sha256,
                clock_completion_sha256=C.sha256(completion_bytes),
                clock_completion_bytes=len(completion_bytes), readback_bytes=readback,
                created_exclusive=True, file_fsynced=True, directory_fsynced=True,
                readback_verified=True)


__all__ = ["SCHEMA", "PROOF_SCHEMA", "SIDECAR_NAME", "LiveReceiptError", "archive_from_scorer_reply",
           "archive_files_for_submit", "observe_archive_submission", "verify_live_receipt",
           "observe_durable_completion", "persist_live_completion"]
