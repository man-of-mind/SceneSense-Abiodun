"""Durable, create-only complete-v010 GT bundles; no producer or live authority.

The READY manifest is published atomically only after every payload is fsynced.
Readers wait only for that manifest, with a bounded timeout, then fail closed on
any missing, foreign, unsafe or inconsistent input. Historical/AVO-only bundles
cannot masquerade as newly collected complete targets. No rewards are changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any, Mapping

import cv2
import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as targets

SCHEMA = "splitfusion.prepared_live_v010_transport.v1"
LIVE_SCOPE = "PROSPECTIVE_LIVE_V010_CONSERVATIVE_CONFLICTS_V1"
IDENTITY_KEYS = frozenset(("run_id", "cell_id", "stream_id", "frame_id", "action_id",
                           "profile_id", "capture_timestamp_ns"))
PAYLOAD_NAMES = frozenset(("inputs.json", "depth_bgra.npy", "raw_semantic.npy",
                          "original_v3_mask.npy", "target.json", "segmentation.png",
                          "object_ignore.png"))
READY_NAME = "READY.json"
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_WAIT_SECONDS = 3.0


class TargetTransportError(ValueError):
    pass


class TargetNotReadyError(TargetTransportError):
    pass


@dataclass(frozen=True)
class TargetBundle:
    identity: Mapping[str, Any]
    inputs: targets.CompleteTargetInputs | None
    prepared: Any
    manifest_sha256: str
    directory: Path
    manifest: Mapping[str, Any]
    native_inputs_verified: bool


def _canonical(value):
    return targets.canonical(targets._json_value(value))


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_identity(identity):
    if not isinstance(identity, Mapping) or set(identity) != IDENTITY_KEYS:
        raise TargetTransportError("GT identity must have exactly the seven MP.gt_identity fields")
    owned = json.loads(_canonical(identity))
    for key in ("run_id", "cell_id", "stream_id"):
        if not isinstance(owned[key], str) or not owned[key] or len(owned[key]) > 512:
            raise TargetTransportError(f"invalid GT identity {key}")
    for key in ("frame_id", "capture_timestamp_ns"):
        if type(owned[key]) is not int or owned[key] < 0:
            raise TargetTransportError(f"invalid GT identity {key}")
    action, profile = owned["action_id"], owned["profile_id"]
    if (action is None) != (profile is None):
        raise TargetTransportError("anchor action/profile must both be null or both real")
    if action is not None and (type(action) is not int or action < 0
            or not isinstance(profile, str) or not profile):
        raise TargetTransportError("invalid GT anchor identity")
    return owned


def bundle_path(directory, identity):
    """Identity is hashed, never interpolated into a filesystem path."""
    return Path(directory) / ("target_" + _hash(_canonical(validate_identity(identity))))


def _validate_metadata(metadata, identity):
    if set(metadata) != {"frame", "split", "actor_boxes", "actor_visibility", "static_candidates", "quarantine", "provenance"}:
        raise TargetTransportError("incomplete/unexpected native input metadata")
    provenance, frame = metadata["provenance"], metadata["frame"]
    if provenance.get("scope") != LIVE_SCOPE:
        raise TargetTransportError("new prospective live scope required; historical sample cannot be live")
    if provenance.get("capture_identity") != identity:
        raise TargetTransportError("complete-input capture identity differs")
    if int(frame["frame_id"]) != identity["frame_id"]:
        raise TargetTransportError("complete-input frame differs from ticket")
    if provenance.get("schema") != targets.SCHEMA or provenance.get("sample_id") != frame.get("sample_id"):
        raise TargetTransportError("complete-input schema/sample mismatch")
    if any(provenance.get("complete", {}).get(key) is not True for key in targets.COMPLETENESS_KEYS):
        raise TargetTransportError("incomplete target producer declarations")
    evidence = provenance.get("evidence_sha256", {})
    if not evidence or any(not _digest(value) for value in evidence.values()):
        raise TargetTransportError("native inputs need explicit source evidence digests")
    if not _digest(provenance.get("input_content_sha256")):
        raise TargetTransportError("native input content digest missing")
    if set(metadata["quarantine"]) != set(targets.QUARANTINE_KEYS):
        raise TargetTransportError("incomplete quarantine declarations")
    for key in ("sample_id", "experiment_id"):
        if str(frame[key]).startswith("canonical_v3_"):
            raise TargetTransportError("historical corpus sample identity cannot be relabelled live")
    for rows in (metadata["actor_boxes"], metadata["actor_visibility"], metadata["static_candidates"]):
        if any(int(row.get("frame_id", -1)) != identity["frame_id"]
               or row.get("sample_id") != frame["sample_id"] for row in rows):
            raise TargetTransportError("target input row has foreign/missing frame identity")


def _validate_inputs(inputs, identity):
    if not isinstance(inputs, targets.CompleteTargetInputs):
        raise TargetTransportError("complete target inputs required; AVO sidecar is insufficient")
    targets._validate(inputs)
    _validate_metadata(_input_metadata(inputs), identity)


def _input_metadata(inputs):
    return {name: targets._json_value(getattr(inputs, name)) for name in (
        "frame", "split", "actor_boxes", "actor_visibility", "static_candidates", "quarantine", "provenance")}


def _array_bytes(array):
    stream = io.BytesIO()
    np.save(stream, np.array(array, copy=True, order="C"), allow_pickle=False)
    return stream.getvalue()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_x(path, payload):
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise TargetTransportError("target payload exceeds bounded size")
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_bundle(directory, *, identity, inputs, prepared):
    """Write an exact prepared target plus all native reconstruction inputs.

    Completeness is a producer assertion, independently qualified elsewhere.
    This function provides integrity/identity, not live scientific qualification.
    Failed writes remain as incomplete evidence; existing bundles are never reused.
    """
    from .evaluation import restore_prepared_target
    identity = validate_identity(identity)
    _validate_inputs(inputs, identity)
    owned = restore_prepared_target(frame_id=prepared.frame_id, sample_id=prepared.sample_id,
        target=prepared.target, content_sha256=prepared.content_sha256)
    if owned.frame_id != identity["frame_id"] or owned.sample_id != inputs.frame["sample_id"]:
        raise TargetTransportError("prepared target frame/sample differs from native inputs")
    if _canonical(owned.target["provenance"]) != _canonical(inputs.provenance):
        raise TargetTransportError("prepared target provenance differs from native inputs")
    metadata = {key: value for key, value in owned.target.items()
                if key not in ("segmentation", "object_ignore", "segmentation_png", "object_ignore_png")}
    payloads = {
        "inputs.json": _canonical(_input_metadata(inputs)),
        "depth_bgra.npy": _array_bytes(inputs.depth_bgra),
        "raw_semantic.npy": _array_bytes(inputs.raw_semantic),
        "original_v3_mask.npy": _array_bytes(inputs.original_v3_mask),
        "target.json": _canonical(metadata),
        "segmentation.png": owned.target["segmentation_png"],
        "object_ignore.png": owned.target["object_ignore_png"],
    }
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise TargetTransportError("target bundle root must be a real directory")
    destination = bundle_path(root, identity)
    destination.mkdir(exist_ok=False)
    for name, payload in payloads.items():
        _write_x(destination / name, payload)
    manifest = {
        "schema": SCHEMA, "identity": identity,
        "frame_id": owned.frame_id, "sample_id": owned.sample_id,
        "input_content_sha256": inputs.provenance["input_content_sha256"],
        "prepared_content_sha256": owned.content_sha256,
        "scope": LIVE_SCOPE, "live_qualified": False,
        "clock_domain": "CLOCK_MONOTONIC_RAW", "ready_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        "ready_wall_ns": time.time_ns(),
        "payloads": {name: {"filename": name, "sha256": _hash(payload), "bytes": len(payload)}
                     for name, payload in payloads.items()},
    }
    encoded = _canonical(manifest)
    temporary = destination / (".READY." + uuid.uuid4().hex + ".tmp")
    _write_x(temporary, encoded)
    # link is atomic and create-only. A reader cannot observe a partial JSON.
    os.link(temporary, destination / READY_NAME)
    temporary.unlink()
    _fsync_directory(destination)
    _fsync_directory(root)
    return {"directory": str(destination), "manifest_sha256": _hash(encoded),
            "frame_id": owned.frame_id, "prepared_content_sha256": owned.content_sha256,
            "input_content_sha256": inputs.provenance["input_content_sha256"]}


def _read(path, *, maximum=MAX_PAYLOAD_BYTES):
    if path.is_symlink() or not path.is_file():
        raise TargetTransportError(f"missing or symbolic target file: {path.name}")
    if path.stat().st_size > maximum:
        raise TargetTransportError("target file exceeds bounded size")
    return path.read_bytes()


def _restore_inputs(metadata, payloads):
    if set(metadata) != {"frame", "split", "actor_boxes", "actor_visibility", "static_candidates", "quarantine", "provenance"}:
        raise TargetTransportError("incomplete/unexpected native input metadata")
    arrays = {name: np.load(io.BytesIO(payloads[name + ".npy"]), allow_pickle=False)
              for name in ("depth_bgra", "raw_semantic", "original_v3_mask")}
    quarantine = metadata["quarantine"]
    if set(quarantine) != set(targets.QUARANTINE_KEYS):
        raise TargetTransportError("incomplete quarantine collections")
    quarantine = dict(quarantine)
    for key in ("confirmed_secondary", "manual_static_ids"):
        quarantine[key] = set(quarantine[key])
    for key in ("dynamic_static_keys", "dynamic_actor_keys", "panel_static_keys"):
        if any(not isinstance(item, list) or len(item) != 2 for item in quarantine[key]):
            raise TargetTransportError("invalid quarantine frame/identity pair")
        quarantine[key] = {tuple(item) for item in quarantine[key]}
    result = targets.CompleteTargetInputs(**{**metadata, "quarantine": quarantine}, **arrays)
    for value in arrays.values():
        value.flags.writeable = False
    return result


def read_bundle(directory, *, expected_identity, timeout_s, expected_manifest_sha256=None,
                verify_native_inputs=False):
    """Read prepared scoring inputs, waiting at most0..3seconds for READY.

    The live default verifies prepared PNGs/tables and native metadata, retaining
    but not reading the large raw NPY payloads. ``verify_full_inputs`` is the
    mandatory offline provenance check; default ``inputs`` is therefore None.
    """
    from .evaluation import restore_prepared_target
    expected = validate_identity(expected_identity)
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or not 0 <= timeout <= MAX_WAIT_SECONDS:
        raise TargetTransportError("target wait must be finite and within0..3seconds")
    if expected_manifest_sha256 is not None and not _digest(expected_manifest_sha256):
        raise TargetTransportError("invalid expected manifest digest")
    destination = bundle_path(directory, expected)
    ready = destination / READY_NAME
    deadline = time.monotonic() + timeout
    while not ready.exists():
        if destination.is_symlink() or ready.is_symlink():
            raise TargetTransportError("symbolic target path refused")
        left = deadline - time.monotonic()
        if left <= 0:
            raise TargetNotReadyError("complete v010 target READY manifest not available")
        time.sleep(min(.001, left))
    if destination.is_symlink() or Path(directory).is_symlink():
        raise TargetTransportError("symbolic target directory refused")
    encoded = _read(ready, maximum=MAX_MANIFEST_BYTES)
    digest = _hash(encoded)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise TargetTransportError("complete target manifest digest differs")
    manifest = json.loads(encoded)
    fields = {"schema", "identity", "frame_id", "sample_id", "input_content_sha256",
              "prepared_content_sha256", "scope", "live_qualified", "clock_domain", "ready_raw_ns", "ready_wall_ns", "payloads"}
    if set(manifest) != fields or manifest["schema"] != SCHEMA or manifest["scope"] != LIVE_SCOPE:
        raise TargetTransportError("incomplete/foreign target manifest schema")
    if validate_identity(manifest["identity"]) != expected or manifest["frame_id"] != expected["frame_id"]:
        raise TargetTransportError("foreign target identity/frame")
    if manifest["live_qualified"] is not False or manifest["clock_domain"] != "CLOCK_MONOTONIC_RAW":
        raise TargetTransportError("invalid transport qualification or clock assertion")
    for name in ("ready_raw_ns", "ready_wall_ns"):
        if type(manifest[name]) is not int or manifest[name] < 0:
            raise TargetTransportError("invalid ready clock")
    if set(manifest["payloads"]) != PAYLOAD_NAMES:
        raise TargetTransportError("incomplete or unexpected target payload names")
    payloads = {}
    for name, item in manifest["payloads"].items():
        if not isinstance(item, Mapping) or set(item) != {"filename", "sha256", "bytes"} or item["filename"] != name:
            raise TargetTransportError("unsafe or foreign payload filename")
        if not _digest(item["sha256"]) or type(item["bytes"]) is not int or not 0 <= item["bytes"] <= MAX_PAYLOAD_BYTES:
            raise TargetTransportError("invalid target payload binding")
        path = destination / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise TargetTransportError(f"missing, symbolic or wrong-sized target payload: {name}")
        if name.endswith(".npy") and not verify_native_inputs:
            continue
        payload = _read(path)
        if len(payload) != item["bytes"] or _hash(payload) != item["sha256"]:
            raise TargetTransportError(f"target payload hash/size differs: {name}")
        payloads[name] = payload
    input_metadata = json.loads(payloads["inputs.json"])
    _validate_metadata(input_metadata, expected)
    inputs = None
    if verify_native_inputs:
        inputs = _restore_inputs(input_metadata, payloads)
        _validate_inputs(inputs, expected)
    if input_metadata["provenance"]["input_content_sha256"] != manifest["input_content_sha256"]:
        raise TargetTransportError("target input content digest differs")
    target = json.loads(payloads["target.json"])
    if _canonical(target.get("provenance")) != _canonical(input_metadata["provenance"]):
        raise TargetTransportError("target/native provenance differs")
    for name in ("segmentation", "object_ignore"):
        raw = payloads[name + ".png"]
        target[name + "_png"] = raw
        target[name] = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    prepared = restore_prepared_target(frame_id=manifest["frame_id"], sample_id=manifest["sample_id"],
        target=target, content_sha256=manifest["prepared_content_sha256"])
    if prepared.sample_id != input_metadata["frame"]["sample_id"]:
        raise TargetTransportError("target/native sample identity differs")
    return TargetBundle(expected, inputs, prepared, digest, destination, manifest, bool(verify_native_inputs))


def verify_full_inputs(directory, *, expected_identity, expected_manifest_sha256=None):
    """Offline complete native-array/hash/input-content verification; no wait."""
    return read_bundle(directory, expected_identity=expected_identity, timeout_s=0,
                       expected_manifest_sha256=expected_manifest_sha256, verify_native_inputs=True)
