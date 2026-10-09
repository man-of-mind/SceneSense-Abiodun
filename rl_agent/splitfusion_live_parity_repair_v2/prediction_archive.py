"""Native calibrated prediction ownership; unchanged labels and scoring rules.

The default is the production camera's exact 720x1280 grid, not the model or
target grid. A separately identified tiny-grid contract is available only for
explicit CPU fixtures and cannot pass the reader's native default. Memory is
allocated lazily; all capacity failures are explicit and retain prior items.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    StaticCameraRegistry, STATIC_CAMERA_MODEL_SHA256, STATIC_CAMERA_MOUNT_SHA256,
)
from rl_agent.splitfusion_live_parity_repair_v1 import prediction_archive as V1

SCHEMA = "splitfusion.parity_repair.prediction_archive.v2"
CONTRACT_SCHEMA = "splitfusion.parity_repair.prediction_contract.v2"
MAX_FRAMES = 1000
MAX_MEMORY_BYTES = 2_000_000_000
canonical = V1.canonical


@dataclass(frozen=True)
class PredictionContract:
    kind: str
    height: int
    width: int
    camera_model_sha256: str | None
    camera_mount_sha256: str | None

    def __post_init__(self):
        if (type(self.height) is not int or type(self.width) is not int
                or not 0 < self.height <= 720 or not 0 < self.width <= 1280):
            raise ValueError("invalid bounded prediction dimensions")
        if self.kind == "NATIVE_CALIBRATED":
            spec = StaticCameraRegistry.audited().resolve(self.camera_model_sha256, self.camera_mount_sha256)
            if (self.height, self.width) != (spec.source_height, spec.source_width):
                raise ValueError("prediction shape differs from audited camera calibration")
        elif self.kind != "CPU_ONLY_SYNTHETIC_FIXTURE" or self.camera_model_sha256 is not None or self.camera_mount_sha256 is not None:
            raise ValueError("unknown/nonexplicit prediction contract")

    @property
    def shape(self):
        return self.height, self.width

    @property
    def label_bytes(self):
        return self.height * self.width

    def document(self):
        return dict(schema=CONTRACT_SCHEMA, kind=self.kind, shape=list(self.shape),
                    dtype="uint8", labels=[0, 1, 2],
                    camera_model_sha256=self.camera_model_sha256,
                    camera_mount_sha256=self.camera_mount_sha256,
                    transformation="NONE__PRESERVE_ACTUAL_NATIVE_LABELS")

    @classmethod
    def cpu_fixture(cls, shape):
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("explicit CPU fixture height/width tuple required")
        return cls("CPU_ONLY_SYNTHETIC_FIXTURE", *shape, None, None)


@lru_cache(maxsize=1)
def native_prediction_contract():
    spec = StaticCameraRegistry.audited().resolve(STATIC_CAMERA_MODEL_SHA256, STATIC_CAMERA_MOUNT_SHA256)
    return PredictionContract("NATIVE_CALIBRATED", spec.source_height, spec.source_width,
                              spec.camera_model_sha256, spec.camera_mount_sha256)


def contract_or_native(contract=None):
    contract = native_prediction_contract() if contract is None else contract
    if not isinstance(contract, PredictionContract):
        raise ValueError("explicit immutable PredictionContract required")
    return contract


def memory_budget(max_frames, max_bytes=None):
    if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
        raise ValueError("explicit 1..1000 frame bound required")
    budget = max(512_000_000, max_frames * 2_000_000) if max_bytes is None else max_bytes
    if type(budget) is not int or not 1 <= budget <= MAX_MEMORY_BYTES:
        raise ValueError("explicit bounded archive memory required (at most 2 GB)")
    return budget


def budget_document(max_frames, max_bytes, contract):
    contract = contract_or_native(contract)
    budget = memory_budget(max_frames, max_bytes)
    return dict(max_frames=max_frames, max_bytes=budget,
                raw_label_bytes_per_frame=contract.label_bytes,
                raw_label_bytes_at_frame_bound=max_frames * contract.label_bytes,
                includes_owned_records_identity_evaluation=True,
                excludes_temporary_copies_and_python_object_overhead=True)


def validate_labels(labels, contract=None):
    contract = contract_or_native(contract)
    if (not isinstance(labels, np.ndarray) or labels.dtype != np.uint8
            or labels.ndim != 2 or labels.shape != contract.shape
            or not np.isin(labels, (0, 1, 2)).all()):
        shape, dtype = getattr(labels, "shape", None), getattr(labels, "dtype", None)
        raise ValueError(f"invalid dense prediction: shape={shape} dtype={dtype}; expected {contract.shape} uint8 labels 0/1/2")
    return labels


class PredictionArchive(V1.PredictionArchive):
    """Retain exact native output; capacity is a bound, never a drop policy."""
    def __init__(self, *, max_frames: int, max_bytes: int | None = None,
                 prediction_contract: PredictionContract | None = None):
        self.prediction_contract = contract_or_native(prediction_contract)
        self.max_frames, self.max_bytes = max_frames, memory_budget(max_frames, max_bytes)
        self._lock = threading.Lock()
        self._items, self._bytes, self._closed = {}, 0, False

    def add(self, ticket):
        identity = self.identity(ticket)
        if not identity["action_identity"]["reward_requested"]:
            raise ValueError("archive takes reward tickets, not held-frame pseudo-tickets")
        contract = self.prediction_contract
        if contract.kind == "NATIVE_CALIBRATED" and (
                ticket.context.camera_model_sha256 != contract.camera_model_sha256
                or ticket.context.camera_mount_sha256 != contract.camera_mount_sha256):
            raise ValueError("prediction ticket camera calibration mismatch")
        labels = validate_labels(ticket.predicted_mask, contract)
        if len(ticket.records) > 4096:
            raise ValueError("prediction record bound exceeded")
        records = canonical(list(ticket.records))
        labels = labels.copy(order="C")
        labels.flags.writeable = False
        key = self.key(identity)
        digest = hashlib.sha256(labels.tobytes() + records).hexdigest()
        cost = labels.nbytes + len(records) + len(canonical(identity))
        with self._lock:
            if self._closed:
                raise ValueError("prediction archive already closed")
            if key in self._items:
                if self._items[key]["prediction_digest"] != digest:
                    raise ValueError("conflicting duplicate prediction")
                return key
            if len(self._items) >= self.max_frames or self._bytes + cost > self.max_bytes:
                raise ValueError("prediction archive capacity exceeded; never silently drop")
            self._items[key] = {"identity": identity, "labels": labels, "records": records,
                "prediction_digest": digest, "snapshot_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                "evaluation": None}
            self._bytes += cost
        return key

    def flush(self, directory: Path):
        """Exact old payload format with a new bound native-contract manifest."""
        directory = Path(directory)
        with self._lock:
            if self._closed:
                raise ValueError("prediction archive already flushed")
            directory.mkdir(parents=True, exist_ok=False)
            self._closed = True
            manifest = {}
            for key, item in self._items.items():
                buffer = io.BytesIO()
                np.save(buffer, item["labels"], allow_pickle=False)
                payloads = {f"{key}.labels.npy": buffer.getvalue(),
                    f"{key}.records.json": item["records"],
                    f"{key}.identity.json": canonical({"schema": SCHEMA, **item["identity"],
                        "prediction_digest": item["prediction_digest"],
                        "snapshot_raw_ns": item["snapshot_raw_ns"]})}
                if item["evaluation"] is not None:
                    payloads[f"{key}.evaluation.json"] = item["evaluation"]
                for name, payload in payloads.items():
                    with (directory / name).open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    manifest[name] = hashlib.sha256(payload).hexdigest()
            result = {"schema": SCHEMA, "frames": len(self._items), "owned_bytes": self._bytes,
                "evaluated_frames": sum(v["evaluation"] is not None for v in self._items.values()),
                "sha256": manifest, "includes_gt_unavailable_tickets": True,
                "flush_after_feedback": True, "quality_and_terminals_unchanged": True,
                "prediction_contract": self.prediction_contract.document(),
                "memory_budget": budget_document(self.max_frames, self.max_bytes, self.prediction_contract)}
            with (directory / "MANIFEST.json").open("xb") as stream:
                stream.write(canonical(result))
                stream.flush()
                os.fsync(stream.fileno())
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return result
