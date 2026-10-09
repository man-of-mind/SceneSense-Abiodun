"""Bounded CPU-input retention and post-drain, create-only evidence.

The live observer retains the already prepared CPU tensor, not a reconstruction
from XYZ-only radar evidence. No clone, D2H copy, tensor scan, or disk operation
is performed by ``observe_input``. This requires the source-bound packing
function in ``scope.py``: its fresh padded storage is only read by the H2D
transfer and is never reused in later frames. Offline selected-payload and
prediction parity remains mandatory before any dense comparison is admitted.

Retaining host memory can change allocation pressure. This is a quality
diagnostic, not an uninstrumented performance measurement or GT timing fix.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch

SHAPE = (1, 7, 448, 768)
INPUT_BYTES = int(np.prod(SHAPE)) * 4
SCHEMA = "splitfusion.paired_quality_input_archive.v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_gt_context(gt: dict, envelope: Any, context: Any) -> None:
    if (set(gt) != {"run_id", "cell_id", "stream_id", "frame_id", "action_id",
                   "profile_id", "capture_timestamp_ns"}
            or not gt["run_id"] or not gt["cell_id"]
            or gt["stream_id"] != context.stream_id
            or gt["frame_id"] != envelope.frame_id
            or gt["capture_timestamp_ns"] != envelope.capture_timestamp_ns
            or gt["action_id"] != envelope.anchor_action_id
            or (envelope.anchor_action_id is None) != (gt["profile_id"] is None)):
        raise ValueError("GT/wire context mismatch")


def check_sources(source_binding: dict) -> None:
    from .scope import PINS
    root = Path(__file__).resolve().parents[2]
    if source_binding != PINS or any(sha(root/name) != digest for name, digest in PINS.items()):
        raise ValueError("bound capture source drift")


def verified_drain(collectors: list[Any]) -> dict:
    """Both front/preparation and GT workers must have actually stopped."""
    if len(collectors) != 1:
        raise ValueError("exactly one owned collector required")
    collector = collectors[0]
    evidence = {}
    for name in ("worker", "evaluation_worker"):
        worker = getattr(collector, name, None)
        if worker is None or not callable(getattr(worker, "is_alive", None)):
            raise ValueError(f"missing owned {name}")
        if worker.is_alive() is not False:
            raise ValueError(f"{name} has not drained")
        evidence[name] = {"alive": False}
    return evidence


class InputArchive:
    def __init__(self, *, max_frames: int = 50,
                 max_bytes: int = 512 * 1024 * 1024) -> None:
        if type(max_frames) is not int or not 1 <= max_frames <= 100:
            raise ValueError("explicit frame bound must be in [1,100]")
        if type(max_bytes) is not int or not INPUT_BYTES <= max_bytes <= 2**30:
            raise ValueError("host retention must be bounded at <=1GiB")
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.records: list[dict] = []
        self.errors: list[str] = []
        self.error_count = self.policy_opportunities = self.retained_bytes = 0
        self._active = 0
        self._local = threading.local()
        self._owner: int | None = None
        self._keys: set[tuple] = set()
        self._closed = False

    def fault(self, exc: Exception) -> None:
        self.error_count += 1
        if len(self.errors) < 32:
            self.errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")

    def begin(self, planned: Any) -> dict | None:
        if not planned.reward_requested:
            return None
        kind = getattr(getattr(planned.plan, "kind", None), "value", None)
        if kind != "POLICY_DECISION":
            raise ValueError("reward request does not belong to a policy decision")
        if self._closed or self._active:
            raise ValueError("archive closed or reentrant materialization")
        tid = threading.get_ident()
        if self._owner not in (None, tid):
            raise ValueError("capture requires the single owned frame worker")
        self._owner = tid
        self.policy_opportunities += 1
        if self.error_count:
            return None  # bounded observer failure: disable further retention
        key = (planned.session_uuid, planned.tensor_seq, planned.frame_id)
        if key in self._keys:
            raise ValueError("duplicate capture identity")
        if len(self.records) >= self.max_frames:
            raise ValueError("input frame bound exhausted; cannot select a subset post-hoc")
        self._keys.add(key)
        record = {"planned": planned, "input": None, "prepared": None,
                  "transfer_device": None}
        self.records.append(record)
        self._active = 1
        self._local.current = record
        return record

    def observe_input(self, tensor: torch.Tensor, device: Any) -> torch.Tensor:
        """Observer failures never enter production preparation/timeout logic."""
        record = getattr(self._local, "current", None)
        if record is None:
            return tensor  # warm-up, holds, fallback or unrelated thread
        try:
            if (record["input"] is not None or not isinstance(tensor, torch.Tensor)
                    or tuple(tensor.shape) != SHAPE or tensor.dtype != torch.float32
                    or tensor.device.type != "cpu" or tensor.requires_grad
                    or not tensor.is_contiguous()):
                raise ValueError("expected one exact contiguous CPU float32 model input")
            if self.retained_bytes + INPUT_BYTES > self.max_bytes:
                raise ValueError("input host-memory bound exhausted")
            record["input"] = tensor
            record["transfer_device"] = str(torch.device(device))
            self.retained_bytes += INPUT_BYTES
        except Exception as exc:
            self.fault(exc)
        return tensor

    def end(self, record: dict | None, prepared: Any = None) -> None:
        if record is None:
            return
        try:
            if getattr(self._local, "current", None) is not record:
                raise ValueError("input observer context changed")
            if prepared is None or record["input"] is None:
                raise ValueError("incomplete prepared policy frame")
            if not isinstance(prepared.wire, bytes) or not prepared.wire:
                raise ValueError("missing immutable SFD4 wire")
            if self.retained_bytes + len(prepared.wire) > self.max_bytes:
                raise ValueError("input plus wire host-memory bound exhausted")
            self.retained_bytes += len(prepared.wire)
            # Do not retain the envelope's second inner-payload allocation.
            record["prepared"] = {"wire": prepared.wire,
                "identity": dict(prepared.identity), "gt_identity": dict(prepared.gt_identity)}
        except Exception as exc:
            self.fault(exc)
        finally:
            self._local.current = None
            self._active = 0

    def flush(self, directory: Path, *, collectors: list[Any],
              source_binding: dict[str, str]) -> dict:
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
        drain = verified_drain(collectors)
        if self._active or self._closed or self.error_count:
            raise ValueError(f"capture incomplete/closed: {self.error_count} observer faults")
        if not self.records or self.policy_opportunities != len(self.records):
            raise ValueError("empty or incomplete policy input population")
        check_sources(source_binding)
        directory = Path(directory)
        if any(p.is_symlink() for p in (directory, *directory.parents)):
            raise ValueError("archive path must not use symlinks")
        directory.mkdir(parents=False, exist_ok=False)
        frames, hashes = [], {}
        for index, record in enumerate(self.records):
            tensor, prepared, planned = record["input"], record["prepared"], record["planned"]
            if record["transfer_device"] not in ("cuda", "cuda:0"):
                raise ValueError("diagnostic requires the original CUDA UE inference path")
            envelope, context = W.unpack_sfd4(prepared["wire"])
            identity = MP.run4_identity(envelope)
            MP.validate_run4_identity(identity)
            if (identity != prepared["identity"] or not envelope.reward_requested
                    or envelope.frame_id != planned.frame_id
                    or envelope.tensor_seq != planned.tensor_seq
                    or envelope.session_uuid != planned.session_uuid
                    or envelope.decision_seq != planned.decision_seq
                    or envelope.ticket_seq != planned.ticket_seq
                    or envelope.capture_timestamp_ns != planned.capture_wall_ns
                    or envelope.mode_id != planned.mode_id or envelope.q_e4 != planned.q_e4):
                raise ValueError("input/wire/plan identity mismatch")
            check_gt_context(prepared["gt_identity"], envelope, context)
            array = tensor.numpy()
            if not np.isfinite(array).all():
                raise ValueError("retained model input is non-finite")
            stem = f"{index:04d}__frame{envelope.frame_id}__tensor{envelope.tensor_seq}"
            npy, wire = stem + ".npy", stem + ".sfd4"
            with (directory / npy).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            with (directory / wire).open("xb") as stream:
                stream.write(prepared["wire"])
                stream.flush()
                os.fsync(stream.fileno())
            hashes[npy], hashes[wire] = sha(directory/npy), sha(directory/wire)
            frames.append({"index": index, "identity": identity,
                "gt_identity": prepared["gt_identity"],
                "input_file": npy, "wire_file": wire, "shape": list(SHAPE),
                "dtype": "float32", "input_bytes": INPUT_BYTES,
                "carla_timestamp": planned.carla_timestamp,
                "radar_window_sha256": planned.radar_window_sha256,
                "transfer_device": record["transfer_device"],
                "stream_id": context.stream_id,
                "status": "PREPARED__TRANSMISSION_AND_PREDICTION_JOIN_REQUIRED"})
        manifest = {"schema": SCHEMA, "complete": True, "frames": frames,
            "max_frames": self.max_frames, "max_bytes": self.max_bytes,
            "retained_bytes": self.retained_bytes, "policy_opportunities": self.policy_opportunities,
            "drain": drain, "source_binding": dict(source_binding), "sha256": hashes,
            "claim_scope": "SAME_INPUT_QUALITY_DIAGNOSTIC_ONLY",
            "live_disk_writes": 0, "extra_gpu_copies": 0,
            "retention_overhead_not_yet_measured": True,
            "selected_replay_parity_required": True, "no_gt_timing_fix_claim": True}
        write_json(directory / "MANIFEST.json", manifest)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self._closed = True
        self.records.clear()  # release host references only after durable export
        return manifest


def verify_archive(directory: Path, *, expected_manifest_sha256: str) -> dict:
    directory = Path(directory)
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("unsafe archive path")
    manifest_path = directory / "MANIFEST.json"
    if (manifest_path.is_symlink() or not manifest_path.is_file()
            or manifest_path.stat().st_size > 1024 * 1024
            or sha(manifest_path) != expected_manifest_sha256):
        raise ValueError("unbound or modified archive manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("foreign/incomplete input archive")
    check_sources(manifest["source_binding"])
    if (type(manifest["max_frames"]) is not int or not 1 <= manifest["max_frames"] <= 100
            or type(manifest["max_bytes"]) is not int
            or not INPUT_BYTES <= manifest["max_bytes"] <= 2**30
            or not 1 <= len(manifest["frames"]) <= manifest["max_frames"]
            or manifest.get("drain") != {"worker": {"alive": False}, "evaluation_worker": {"alive": False}}):
        raise ValueError("invalid archive bounds or drain")
    hashes = manifest["sha256"]
    if set(p.name for p in directory.iterdir()) != set(hashes) | {"MANIFEST.json"}:
        raise ValueError("input archive inventory mismatch")
    for name, digest in hashes.items():
        path = directory / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or path.stat().st_size > manifest["max_bytes"] + 4096
                or sha(path) != digest):
            raise ValueError("unsafe or modified input archive member")
    if len(manifest["frames"]) != manifest["policy_opportunities"]:
        raise ValueError("missing policy input records")
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    referenced, identities, retained_bytes = set(), set(), 0
    for index, frame in enumerate(manifest["frames"]):
        npy, wire = frame["input_file"], frame["wire_file"]
        if (npy not in hashes or wire not in hashes or npy == wire
                or npy in referenced or wire in referenced):
            raise ValueError("unbound frame members")
        referenced.update((npy, wire))
        if not INPUT_BYTES < (directory/npy).stat().st_size <= INPUT_BYTES + 4096:
            raise ValueError("invalid NPY size before deserialization")
        array = np.load(directory/npy, allow_pickle=False, mmap_mode="r")
        if array.shape != SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("invalid exact model input")
        data = (directory/wire).read_bytes()
        envelope, context = W.unpack_sfd4(data)
        identity = MP.run4_identity(envelope)
        MP.validate_run4_identity(identity)
        key = (envelope.session_uuid, envelope.tensor_seq, envelope.frame_id)
        if identity != frame["identity"] or not envelope.reward_requested or key in identities:
            raise ValueError("duplicate or mismatched policy input identity")
        check_gt_context(frame["gt_identity"], envelope, context)
        if (frame["index"] != index or frame["stream_id"] != context.stream_id
                or frame["transfer_device"] not in ("cuda", "cuda:0")
                or frame["shape"] != list(SHAPE) or frame["dtype"] != "float32"
                or frame["input_bytes"] != INPUT_BYTES):
            raise ValueError("invalid frame context or input metadata")
        retained_bytes += INPUT_BYTES + len(data)
        identities.add(key)
    if referenced != set(hashes):
        raise ValueError("unreferenced archive data")
    if retained_bytes != manifest["retained_bytes"] or retained_bytes > manifest["max_bytes"]:
        raise ValueError("input/wire byte budget mismatch")
    return manifest
