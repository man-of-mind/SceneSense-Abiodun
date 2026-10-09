"""Off-decision-path persistence for retained records (post-drain or one background writer thread).

Nothing here is called from ``observe_input``/``begin``/``end``. ``persist_record`` performs every
check that v1 performed at flush (SFD4 unpack, plan/wire/GT identity agreement, finite input) and then
writes the exact prepared tensor (.npy) and the executed wire (.sfd4) create-only with fsync, after
checking the configured storage cap and the free-space reserve. References are released afterwards.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import queue
import shutil
import threading
from typing import Any

import numpy as np

from .plan import INPUT_BYTES, INPUT_SHAPE

NPY_HEADER_MAX = 256


class RetentionWriteError(ValueError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def context_document(context: Any) -> dict:
    """Same field mapping as ``splitfusion_live_parity_runtime_v1.pose.context_document``."""
    return {"stream_id": context.stream_id, "frame_id": context.frame_id,
            "sequence_id": context.sequence_id, "capture_timestamp_ns": context.capture_timestamp_ns,
            "ego_world": list(context.ego_world.values()),
            "camera_model_sha256": context.camera_model_sha256,
            "camera_mount_sha256": context.camera_mount_sha256}


def check_gt_context(gt: dict, envelope: Any, context: Any) -> None:
    """v1 semantics (capture.check_gt_context), raising a classified write error."""
    if (set(gt) != {"run_id", "cell_id", "stream_id", "frame_id", "action_id", "profile_id",
                    "capture_timestamp_ns"}
            or not gt["run_id"] or not gt["cell_id"]
            or gt["stream_id"] != context.stream_id or gt["frame_id"] != envelope.frame_id
            or gt["capture_timestamp_ns"] != envelope.capture_timestamp_ns
            or gt["action_id"] != envelope.anchor_action_id
            or (envelope.anchor_action_id is None) != (gt["profile_id"] is None)):
        raise RetentionWriteError("INCOMPLETE_GT_WIRE_IDENTITY_MISMATCH")


def check_record_identity(rec: dict) -> tuple[Any, Any, dict]:
    """Join planned <-> wire <-> run4 identity <-> GT identity. Returns (envelope, context, identity)."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    planned, prepared = rec["planned"], rec["prepared"]
    try:
        envelope, context = W.unpack_sfd4(prepared["wire"])
        identity = MP.run4_identity(envelope)
        MP.validate_run4_identity(identity)
    except Exception as exc:
        raise RetentionWriteError("INCOMPLETE_WIRE_UNPARSEABLE", f"{type(exc).__name__}: {exc}") from exc
    if (identity != prepared["identity"] or not envelope.reward_requested
            or identity["frame_kind"] != "POLICY_DECISION"
            or envelope.frame_id != planned.frame_id or envelope.tensor_seq != planned.tensor_seq
            or envelope.session_uuid != planned.session_uuid
            or envelope.decision_seq != planned.decision_seq
            or envelope.ticket_seq != planned.ticket_seq
            or envelope.capture_timestamp_ns != planned.capture_wall_ns
            or envelope.mode_id != planned.mode_id or envelope.q_e4 != planned.q_e4
            or context.sequence_id != envelope.tensor_seq):
        raise RetentionWriteError("INCOMPLETE_PLAN_WIRE_IDENTITY_MISMATCH")
    check_gt_context(prepared["gt_identity"], envelope, context)
    return envelope, context, identity


class StorageBudget:
    def __init__(self, directory: Path, max_storage_bytes: int, min_free_disk_bytes: int):
        self.directory, self.max_bytes, self.reserve = Path(directory), max_storage_bytes, min_free_disk_bytes
        self.used = 0

    def claim(self, amount: int) -> None:
        if self.used + amount > self.max_bytes:
            raise RetentionWriteError("INCOMPLETE_STORAGE_CAP", f"{self.used}+{amount}>{self.max_bytes}")
        if shutil.disk_usage(self.directory).free - amount < self.reserve:
            raise RetentionWriteError("INCOMPLETE_FREE_DISK_RESERVE")
        self.used += amount


def _write_create_only(path: Path, raw_writer) -> None:
    with path.open("xb") as stream:
        raw_writer(stream)
        stream.flush()
        os.fsync(stream.fileno())


def persist_record(rec: dict, directory: Path, budget: StorageBudget, *, require_cuda: bool) -> dict:
    """Validate and durably write one held record. Raises RetentionWriteError with a reason."""
    if rec.get("input") is None or rec.get("prepared") is None:
        raise RetentionWriteError("INCOMPLETE_NOT_HELD")
    if require_cuda and rec["device"] not in ("cuda", "cuda:0"):
        raise RetentionWriteError("INCOMPLETE_NOT_ORIGINAL_CUDA_TRANSFER", str(rec["device"]))
    envelope, context, identity = check_record_identity(rec)
    tensor = rec["input"]
    array = tensor.numpy()          # CPU tensor -> zero-copy view, off the decision path
    if array.shape != INPUT_SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
        raise RetentionWriteError("INCOMPLETE_NONFINITE_OR_MALFORMED_INPUT")
    wire = rec["prepared"]["wire"]
    stem = f"r{rec['ordinal']:05d}__frame{envelope.frame_id}__tensor{envelope.tensor_seq}"
    npy, sfd4 = Path(directory) / (stem + ".npy"), Path(directory) / (stem + ".sfd4")
    budget.claim(INPUT_BYTES + NPY_HEADER_MAX + len(wire))
    try:
        _write_create_only(npy, lambda s: np.save(s, array, allow_pickle=False))
        _write_create_only(sfd4, lambda s: s.write(wire))
    except OSError as exc:
        raise RetentionWriteError("INCOMPLETE_WRITE_FAILED", f"{type(exc).__name__}: {exc}") from exc
    planned = rec["planned"]
    return {
        "ordinal": rec["ordinal"], "status": "RETAINED_COMPLETE",
        "identity": identity, "gt_identity": dict(rec["prepared"]["gt_identity"]),
        "join_key": {"session_uuid": envelope.session_uuid, "tensor_seq": envelope.tensor_seq,
                     "frame_id": envelope.frame_id, "capture_timestamp_ns": envelope.capture_timestamp_ns,
                     "decision_ordinal": rec["ordinal"]},
        "action": {"mode_id": envelope.mode_id, "q_e4": envelope.q_e4,
                   "anchor_action_id": envelope.anchor_action_id, "kind": identity["frame_kind"],
                   "execution_bundle_sha256": envelope.execution_bundle_sha256},
        "frame_context": context_document(context),
        "carla_timestamp": getattr(planned, "carla_timestamp", None),
        "radar_window_sha256": getattr(planned, "radar_window_sha256", None),
        "transfer_device": rec["device"],
        "input_file": npy.name, "input_sha256": sha256_file(npy), "input_file_bytes": npy.stat().st_size,
        "input_shape": list(INPUT_SHAPE), "input_dtype": "float32", "input_bytes": INPUT_BYTES,
        "wire_file": sfd4.name, "wire_sha256": hashlib.sha256(wire).hexdigest(), "wire_bytes": len(wire),
        "input_binding": "OBSERVED_INSIDE_THE_SAME_MATERIALIZE_CALL_ON_THE_OWNED_FRAME_WORKER",
    }


def release(rec: dict) -> None:
    rec["input"] = None
    rec["prepared"] = None
    rec["planned"] = None


class BackgroundWriter:
    """One daemon writer thread fed by a bounded queue; never blocks the producer."""

    def __init__(self, directory: Path, *, max_storage_bytes: int, min_free_disk_bytes: int,
                 queue_records: int, require_cuda: bool):
        self.directory = Path(directory)
        self.queue: queue.Queue = queue.Queue(maxsize=queue_records)
        self.require_cuda = require_cuda
        self._lock = threading.Lock()
        self.results: dict[int, dict] = {}
        self._released = 0
        self._thread: threading.Thread | None = None
        self._budget_args = (max_storage_bytes, min_free_disk_bytes)
        self.budget: StorageBudget | None = None
        self.thread_ident: int | None = None

    def start(self) -> None:
        """Called at scope entry (before the route), never on the decision path."""
        if self._thread is not None:
            raise RuntimeError("writer already started")
        if any(p.is_symlink() for p in (self.directory, *self.directory.parents)):
            raise ValueError("retention path must not use symlinks")
        self.directory.mkdir(parents=False, exist_ok=False)
        self.budget = StorageBudget(self.directory, *self._budget_args)
        self._thread = threading.Thread(target=self._run, name="paired-retention-v2-writer", daemon=True)
        self._thread.start()

    def submit(self, rec: dict) -> bool:
        try:
            self.queue.put_nowait(rec)
            return True
        except queue.Full:
            return False

    def released_bytes(self) -> int:
        with self._lock:
            return self._released

    def _run(self) -> None:
        self.thread_ident = threading.get_ident()
        while True:
            rec = self.queue.get()
            if rec is None:
                return
            try:
                result = persist_record(rec, self.directory, self.budget, require_cuda=self.require_cuda)
            except RetentionWriteError as exc:
                result = {"ordinal": rec["ordinal"], "status": exc.reason, "detail": str(exc)[:300]}
            except Exception as exc:      # never let the writer die silently
                result = {"ordinal": rec["ordinal"], "status": "INCOMPLETE_WRITE_FAILED",
                          "detail": f"{type(exc).__name__}: {exc}"[:300]}
            reserved = rec.get("reserved_bytes", 0)
            release(rec)
            with self._lock:
                self.results[result["ordinal"]] = result
                self._released += reserved

    def close(self, timeout_s: float = 900.0) -> dict[int, dict]:
        if self._thread is None:
            return {}
        self.queue.put(None)
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("retention writer did not drain")
        with self._lock:
            return dict(self.results)
