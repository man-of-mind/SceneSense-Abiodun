"""Qualified SFD1-v2 live-pilot UE bridge and isolated edge service.

The UE owns only the frozen front/ranker/encoders. The separately started
``oai-perception-rx`` service owns only the frozen tail/decoders. Both
directions use the existing production ``!IHH`` fragmentation header; neither
direction adds feature compression beyond the mandatory inner zstd level 1.

Phase-15 real-time recovery (see
``experiments/splitfusion_phase15_retry4_latency_audit_v1/20260906_root_cause_audit``):
the dense 720x1280 evaluation label map no longer rides the radio return path.
It is persisted atomically on the edge's own evidence mount and only compact
object/service records plus a compact terminal ACK travel back to the UE. Both
endpoints run a bounded latest-frame-first pending slot instead of an implicit
FIFO backlog.  The frozen 100 ms capture-to-install service target remains
distinct from the 500 ms feedback/processing horizon: a result between them is
late-accepted, while work older than the feedback horizon is discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import socket
import threading
import time
import zlib
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np
import torch

from phase2_map_sharing.transport import ChunkReassembler, chunk_payload
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    guards,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    load_frozen_perception,
)

from .context_tail import ContextualFrozenP025TailAdapter
from .edge_runtime import PreloadedSplitEdgeRuntime
from .envelope import unpack_envelope
from .frame_context import StaticCameraRegistry, build_frame_context_v1
from .registry import SplitActionRegistry
from .transport import ProductionSplitCodec
from .ue_runtime import PreloadedSplitUERuntime


EDGE_RESULT_SCHEMA = "splitfusion_edge_result.v2"
OBJECT_MAP_UPDATE_SCHEMA = "splitfusion_object_map_update.v1"
EDGE_TERMINAL_ACK_SCHEMA = "splitfusion_edge_terminal_ack.v1"
EDGE_COUNTERS_SCHEMA = "splitfusion_edge_counters.v1"
EVIDENCE_SIDECAR_SCHEMA = "splitfusion_segmentation_evidence.v1"

# These are deliberately different contracts.  The service target classifies
# capture-to-install timeliness; the longer ACK timeout bounds useful work and
# declares missing feedback.  Both endpoints derive both absolute instants
# from the capture timestamp already carried by SFD1-v2.
SERVICE_DEADLINE_CONFIG_KEY = "service_deadline_ms"
ACK_TIMEOUT_CONFIG_KEY = "ack_timeout_ms"

UE_STAGE_AFTER_PREPARATION = "UE_AFTER_PREPARATION"
UE_STAGE_BEFORE_SEND = "UE_BEFORE_SEND"
UE_STAGE_BEFORE_MAP_PUBLICATION = "UE_BEFORE_MAP_PUBLICATION"
EDGE_STAGE_AFTER_REASSEMBLY = "EDGE_AFTER_REASSEMBLY"
EDGE_STAGE_BEFORE_DECODE = "EDGE_BEFORE_DECODE"
EDGE_STAGE_BEFORE_TAIL = "EDGE_BEFORE_TAIL"
EDGE_STAGE_BEFORE_PUBLICATION = "EDGE_BEFORE_PUBLICATION"
DEADLINE_STAGES = (
    UE_STAGE_AFTER_PREPARATION,
    UE_STAGE_BEFORE_SEND,
    EDGE_STAGE_AFTER_REASSEMBLY,
    EDGE_STAGE_BEFORE_DECODE,
    EDGE_STAGE_BEFORE_TAIL,
    EDGE_STAGE_BEFORE_PUBLICATION,
    UE_STAGE_BEFORE_MAP_PUBLICATION,
)


class LivePilotRuntimeError(RuntimeError):
    """The live dispatch, result identity, or map handoff was invalid."""


class DeadlineExpired(RuntimeError):
    """The capture-based processing/feedback horizon expired at one stage."""

    def __init__(self, stage: str, *, capture_timestamp_ns: int, age_ms: float) -> None:
        super().__init__(
            f"processing horizon expired at {stage} (age {age_ms:.1f} ms)"
        )
        self.stage = str(stage)
        self.capture_timestamp_ns = int(capture_timestamp_ns)
        self.age_ms = float(age_ms)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LivePilotRuntimeError(message)


def service_deadline_s(campaign: Mapping[str, Any]) -> float:
    """Return the frozen desired capture-to-install service target."""

    value = float(campaign["cell"][SERVICE_DEADLINE_CONFIG_KEY]) / 1000.0
    _require(value > 0.0, "configured service deadline must be positive")
    return value


def ack_timeout_s(campaign: Mapping[str, Any]) -> float:
    """Return the capture-based feedback timeout and processing horizon."""

    service = service_deadline_s(campaign)
    value = float(campaign["cell"][ACK_TIMEOUT_CONFIG_KEY]) / 1000.0
    _require(value > service, "ACK timeout must exceed the service deadline")
    return value


def deadline_at_s(capture_timestamp_ns: int, deadline_s: float) -> float:
    """The absolute wall-clock instant this capture must be installed by."""

    return int(capture_timestamp_ns) / 1_000_000_000.0 + float(deadline_s)


def check_deadline(
    stage: str, capture_timestamp_ns: int, deadline_s: float, *, now_s: float | None = None
) -> float:
    """Raise :class:`DeadlineExpired` when the supplied horizon has elapsed.

    ``capture_timestamp_ns`` is the original CARLA capture instant on the one
    physical host wall clock, so the same comparison is valid in the UE process
    and inside the edge container.
    """

    observed = time.time() if now_s is None else float(now_s)
    limit = deadline_at_s(capture_timestamp_ns, deadline_s)
    age_ms = (observed - int(capture_timestamp_ns) / 1_000_000_000.0) * 1000.0
    if observed > limit:
        raise DeadlineExpired(
            stage, capture_timestamp_ns=int(capture_timestamp_ns), age_ms=age_ms
        )
    return age_ms


class _Counters:
    """Thread-safe named counters reconciled into the cell evidence."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def bump(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[str(name)] += int(amount)

    def set_max(self, name: str, value: int) -> None:
        with self._lock:
            if int(value) > self._counts[str(name)]:
                self._counts[str(name)] = int(value)

    def set_value(self, name: str, value: int) -> None:
        with self._lock:
            self._counts[str(name)] = int(value)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


class FastStationaryTrackAccumulator:
    """Point-order-equivalent stationary tracking grouped by spatial cell.

    The dataset implementation walks every radar return in Python.  A live
    two-sweep window contains about 37k returns, making that loop the dominant
    avoidable preparation cost.  Keys are independent, so this implementation
    groups them in NumPy while preserving the exact within-cell point order and
    reset behavior of the reference accumulator.
    """

    def __init__(
        self,
        stationary_velocity_mps: float = 0.35,
        parked_threshold_s: float = 5.0,
        association_grid_m: float = 1.5,
        max_stale_s: float = 2.0,
    ) -> None:
        self.stationary_velocity_mps = float(stationary_velocity_mps)
        self.parked_threshold_s = float(parked_threshold_s)
        self.association_grid_m = float(association_grid_m)
        self.max_stale_s = float(max_stale_s)
        self._keys = np.zeros((0,), dtype=np.uint64)
        self._ages = np.zeros((0,), dtype=np.float64)
        self._last_seen = np.zeros((0,), dtype=np.float64)
        self._x = np.zeros((0,), dtype=np.float64)
        self._y = np.zeros((0,), dtype=np.float64)

    def _key(self, x: float, y: float) -> tuple[int, int]:
        scale = max(0.05, self.association_grid_m)
        return int(round(float(x) / scale)), int(round(float(y) / scale))

    @staticmethod
    def _pack_keys(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x64 = np.rint(x.astype(np.float64, copy=False)).astype(np.int64)
        y64 = np.rint(y.astype(np.float64, copy=False)).astype(np.int64)
        limit = np.iinfo(np.int32)
        _require(
            bool(np.all((x64 >= limit.min) & (x64 <= limit.max)))
            and bool(np.all((y64 >= limit.min) & (y64 <= limit.max))),
            "stationary tracker grid key exceeds int32",
        )
        xbits = x64.astype(np.int32).view(np.uint32).astype(np.uint64)
        ybits = y64.astype(np.int32).view(np.uint32).astype(np.uint64)
        return (xbits << np.uint64(32)) | ybits

    @staticmethod
    def _unpack_key(value: np.uint64) -> tuple[int, int]:
        raw = int(value)
        x = np.array([raw >> 32], dtype=np.uint32).view(np.int32)[0]
        y = np.array([raw & 0xFFFFFFFF], dtype=np.uint32).view(np.int32)[0]
        return int(x), int(y)

    def tracks_snapshot(self) -> dict[tuple[int, int], dict[str, float]]:
        """Expose reference-shaped state for qualification, never hot-path use."""

        return {
            self._unpack_key(key): {
                "age_s": float(self._ages[index]),
                "last_seen_s": float(self._last_seen[index]),
                "x": float(self._x[index]),
                "y": float(self._y[index]),
            }
            for index, key in enumerate(self._keys)
        }

    def update(self, world_velocity_points: np.ndarray, frame_time_s: float) -> np.ndarray:
        points = np.asarray(world_velocity_points)
        if points.size == 0:
            return np.zeros((0,), dtype=np.float32)
        _require(
            points.ndim == 2 and points.shape[1] >= 4,
            "stationary tracker input must be [N,>=4]",
        )
        now = float(frame_time_s)
        scale = max(0.05, self.association_grid_m)
        packed = self._pack_keys(
            points[:, 0].astype(np.float64, copy=False) / scale,
            points[:, 1].astype(np.float64, copy=False) / scale,
        )
        unique_keys, inverse = np.unique(packed, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=len(unique_keys))
        starts = np.cumsum(counts) - counts
        stationary = (
            np.abs(points[:, 3].astype(np.float64, copy=False))
            <= self.stationary_velocity_mps
        )
        ages = np.zeros((points.shape[0],), dtype=np.float32)

        previous_age = np.zeros(len(unique_keys), dtype=np.float64)
        previous_seen = np.full(len(unique_keys), now, dtype=np.float64)
        if self._keys.size:
            prior_positions = np.searchsorted(self._keys, unique_keys)
            in_bounds = prior_positions < len(self._keys)
            matched = np.zeros(len(unique_keys), dtype=bool)
            matched[in_bounds] = (
                self._keys[prior_positions[in_bounds]] == unique_keys[in_bounds]
            )
            previous_age[matched] = self._ages[prior_positions[matched]]
            previous_seen[matched] = self._last_seen[prior_positions[matched]]
        dt = np.maximum(0.0, now - previous_seen)
        first_indices = order[starts]
        first_age = np.where(
            stationary[first_indices],
            np.minimum(self.parked_threshold_s * 3.0, previous_age + dt),
            0.0,
        )

        # The first moving return resets a key. Subsequent returns for that key
        # see dt=0 in the reference loop and therefore retain zero age.
        group_ids = inverse[order]
        positions_in_group = np.arange(len(order), dtype=np.int64) - starts[group_ids]
        first_moving = counts.astype(np.int64, copy=True)
        moving_positions = np.flatnonzero(~stationary[order])
        if moving_positions.size:
            np.minimum.at(
                first_moving,
                group_ids[moving_positions],
                positions_in_group[moving_positions],
            )
        prefix = positions_in_group < first_moving[group_ids]
        ages[order[prefix]] = first_age[group_ids[prefix]].astype(np.float32)
        final_age = np.where(first_moving == counts, first_age, 0.0)
        last_indices = order[starts + counts - 1]

        retain = np.zeros(len(self._keys), dtype=bool)
        if self._keys.size:
            current_positions = np.searchsorted(unique_keys, self._keys)
            current_bounds = current_positions < len(unique_keys)
            is_current = np.zeros(len(self._keys), dtype=bool)
            is_current[current_bounds] = (
                unique_keys[current_positions[current_bounds]]
                == self._keys[current_bounds]
            )
            stale_after = max(self.max_stale_s, self.association_grid_m)
            retain = (~is_current) & ((now - self._last_seen) <= stale_after)
        merged_keys = np.concatenate((self._keys[retain], unique_keys))
        merged_order = np.argsort(merged_keys, kind="stable")
        self._keys = merged_keys[merged_order]
        self._ages = np.concatenate((self._ages[retain], final_age))[merged_order]
        self._last_seen = np.concatenate(
            (self._last_seen[retain], np.full(len(unique_keys), now, dtype=np.float64))
        )[merged_order]
        self._x = np.concatenate(
            (self._x[retain], points[last_indices, 0].astype(np.float64, copy=False))
        )[merged_order]
        self._y = np.concatenate(
            (self._y[retain], points[last_indices, 1].astype(np.float64, copy=False))
        )[merged_order]
        return ages


class LatestFramePendingSlot:
    """Bounded latest-frame-first pending work, at most one frame per stream.

    This replaces the implicit FIFO backlog that the retry4 audit measured. A
    newer complete opportunity replaces an older pending opportunity and the
    replaced item is returned so the caller can record it with an exact reason.
    A frame already handed to a worker is never interrupted.
    """

    def __init__(self, *, capacity_per_stream: int = 1) -> None:
        if int(capacity_per_stream) != 1:
            raise ValueError("exactly one pending frame per stream is supported")
        self._pending: "OrderedDict[str, Any]" = OrderedDict()
        self._condition = threading.Condition()
        self._closed = False

    def offer(
        self, stream_id: str, item: Any, *, sequence: int
    ) -> tuple[bool, Any | None]:
        """Try to admit ``item``.

        Returns ``(admitted, displaced)``. ``displaced`` is the older pending
        item this admission replaced, if any, so the caller can record it with
        an exact reason. When ``admitted`` is false the offered item itself was
        refused, either because the slot is closed or because pending work is
        already fresher; the caller owns recording that drop.
        """

        with self._condition:
            if self._closed:
                return False, None
            previous = self._pending.get(str(stream_id))
            if previous is not None and int(previous[0]) >= int(sequence):
                # An out-of-order arrival must never displace fresher pending
                # work; the stale arrival is itself the dropped opportunity.
                return False, None
            self._pending[str(stream_id)] = (int(sequence), item)
            self._pending.move_to_end(str(stream_id))
            self._condition.notify()
            return True, (None if previous is None else previous[1])

    def take(self, timeout: float) -> tuple[str, Any] | None:
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while not self._pending:
                if self._closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)
            stream_id, (_sequence, item) = self._pending.popitem(last=False)
            return str(stream_id), item

    def depth(self) -> int:
        with self._condition:
            return len(self._pending)

    def close(self) -> list[Any]:
        with self._condition:
            self._closed = True
            dropped = [item for _sequence, item in self._pending.values()]
            self._pending.clear()
            self._condition.notify_all()
            return dropped


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish ``payload`` so no reader can ever observe a partial file."""

    staging = path.with_suffix(path.suffix + ".tmp")
    try:
        with staging.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    os.replace(staging, path)


def _write_decoded_evidence(evidence: Path, mask: np.ndarray) -> None:
    """Publish an evaluation-only mask so no reader can observe a partial file.

    The adapter's segmentation evaluator polls for this exact name and loads it
    as soon as it exists, so the array is staged under a temporary name, flushed
    and fsynced, then renamed into place.
    """

    if evidence.exists():
        raise FileExistsError(f"duplicate decoded segmentation evidence: {evidence}")
    staging = evidence.with_suffix(evidence.suffix + ".tmp")
    try:
        with staging.open("wb") as handle:
            np.save(handle, mask, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    os.replace(staging, evidence)


def segmentation_evidence_name(stream_id: str, frame_id: int) -> str:
    """The exact evidence name the adapter's segmentation evaluator polls for."""

    stream_digest = hashlib.sha256(str(stream_id).encode("utf-8")).hexdigest()[:16]
    return f"{stream_digest}_{int(frame_id)}.npy"


class EdgeEvaluationEvidenceWriter:
    """Bounded, non-blocking edge-side sink for evaluation-only label maps.

    The label map is evaluation evidence, not deployment feedback, so it is
    persisted on the edge's own writable per-cell mount and never base64-encoded
    into the radio return path. Writing runs on this dedicated thread so the
    receive, tail and result-transmission paths never block on a ~900 KB write.
    """

    def __init__(self, output_dir: Path, counters: _Counters, *, depth: int = 8) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counters = counters
        self._queue: "queue.Queue[tuple[Path, np.ndarray, dict[str, Any]] | None]" = (
            queue.Queue(maxsize=int(depth))
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="edge-evaluation-evidence", daemon=True
        )
        self._thread.start()

    def submit(self, mask: np.ndarray, sidecar: Mapping[str, Any]) -> str:
        """Queue one mask; returns the recorded installation status."""

        path = self.output_dir / str(sidecar["evidence_name"])
        try:
            self._queue.put_nowait((path, mask, dict(sidecar)))
        except queue.Full:
            self._counters.bump("evaluation_masks_dropped_writer_backpressure")
            return "EVALUATION_EVIDENCE_WRITER_SATURATED"
        self._counters.bump("evaluation_masks_submitted")
        return "EVALUATION_EVIDENCE_WRITE_SUBMITTED"

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if item is None:
                return
            path, mask, sidecar = item
            try:
                digest = hashlib.sha256(
                    np.ascontiguousarray(mask).tobytes()
                ).hexdigest()
                verified = digest == str(sidecar.get("sha256") or "")
                # The sidecar is published first: the consumer polls for the
                # .npy name, so the binding metadata and digest must already be
                # readable by the time that name appears.
                _atomic_write_bytes(
                    path.with_suffix(".json"),
                    json.dumps(
                        {**sidecar, "hash_verified": bool(verified)},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                )
                _write_decoded_evidence(path, mask)
                self._counters.bump("evaluation_masks_persisted")
                if verified:
                    self._counters.bump("evaluation_masks_hash_verified")
                else:
                    self._counters.bump("evaluation_masks_hash_mismatched")
            except Exception:
                self._counters.bump("evaluation_masks_write_failed")
            finally:
                self._queue.task_done()

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        deadline = time.monotonic() + float(timeout)
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        self._thread.join(timeout=1.0)


class _DeadlineGuardedTail:
    """Refuse frozen-tail inference for a capture that can no longer be timely.

    ``edge_runtime`` and ``context_tail`` are SHA-256 pinned by the SFD1-v2
    frame-context authority, so the pre-tail deadline gate is installed as this
    callable wrapper around the unmodified frozen tail rather than as an edit to
    either pinned module. The wrapped object is only ever asked for inference;
    serialization and snapshot consumption still go to the real tail.
    """

    def __init__(
        self,
        tail: Callable[..., Any],
        guard: Callable[[str, int], None],
    ) -> None:
        self._tail = tail
        self._guard = guard

    def __call__(self, c2: Any, metadata: Any) -> Any:
        self._guard(EDGE_STAGE_BEFORE_TAIL, int(metadata.capture_timestamp_ns))
        return self._tail(c2, metadata)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"configuration is not an object: {path}")
    return value


def _finite_tree(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), "non-finite frozen-tail output")
    elif isinstance(value, Mapping):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_tree(child)


def _trace_ns(trace: Any) -> dict[str, int]:
    return {
        str(boundary.name): int(boundary.finished_monotonic_ns - boundary.started_monotonic_ns)
        for boundary in trace.boundaries
    }


class _Ledger(_Counters):
    """Frozen-module call ledger; identical accounting to :class:`_Counters`."""


class _Front:
    def __init__(self, model: torch.nn.Module, ledger: _Ledger) -> None:
        self._model, self._ledger = model, ledger

    def __call__(self, input_7ch: torch.Tensor) -> torch.Tensor:
        self._ledger.bump("front")
        return self._model.encode_front(input_7ch)


class _Ranker:
    def __init__(self, model: torch.nn.Module, ledger: _Ledger) -> None:
        self._model, self._ledger = model, ledger

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        self._ledger.bump("ranker")
        return self._model.score_cells(c2)


class _AE:
    def __init__(self, family: str, model: Any, ledger: _Ledger) -> None:
        self.family = str(family)
        self.family_id = int(model.family_id)
        self.bottleneck = int(model.bottleneck)
        self.routing_tag = int(model.routing_tag)
        self._model, self._ledger = model, ledger

    def encode(self, c2: torch.Tensor) -> torch.Tensor:
        self._ledger.bump(f"ae_encoder_{self.family}")
        return self._model.encode(c2)

    def decode(self, latent: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        self._ledger.bump(f"ae_decoder_{self.family}")
        return self._model.decode(latent, keep_mask)


def _preload_ue(device: torch.device) -> tuple[PreloadedSplitUERuntime, _Ledger, list[Any]]:
    registry = SplitActionRegistry.from_runtime_binding()
    model, _base, _binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    ranker = phase11b._load_ranker(device)
    autoencoders = {}
    for family, _family_id, bottleneck in phase11b.FAMILIES:
        if bottleneck is None:
            continue
        item = phase11b.FROZEN_INPUTS[family]
        payload = torch.load(
            phase11b._repository_path(item["path"]),
            map_location="cpu", weights_only=False,
        )
        autoencoders[family] = phase11b._load_selected_autoencoder(
            family, bottleneck, item, payload, device,
        )
        del payload
    guards.require_frozen_perception([model, ranker, *autoencoders.values()])
    guards.require_eval_mode([model, ranker, *autoencoders.values()])
    ledger = _Ledger()
    wrapped = {family: _AE(family, ae, ledger) for family, ae in autoencoders.items()}
    runtime = PreloadedSplitUERuntime(
        registry, front=_Front(model, ledger), ranker=_Ranker(ranker, ledger),
        ae_encoders=wrapped, device=device, codec=ProductionSplitCodec(),
        prepare_modules=False, startup_model_load_operations=5,
        startup_model_construction_operations=5,
    )
    return runtime, ledger, [model, ranker, *autoencoders.values()]


def _preload_edge(
    device: torch.device,
    *,
    deadline_guard: Callable[[str, int], None] | None = None,
) -> tuple[PreloadedSplitEdgeRuntime, ContextualFrozenP025TailAdapter, _Ledger, list[Any]]:
    registry = SplitActionRegistry.from_runtime_binding()
    model, base, _binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    autoencoders = {}
    for family, _family_id, bottleneck in phase11b.FAMILIES:
        if bottleneck is None:
            continue
        item = phase11b.FROZEN_INPUTS[family]
        payload = torch.load(
            phase11b._repository_path(item["path"]),
            map_location="cpu", weights_only=False,
        )
        autoencoders[family] = phase11b._load_selected_autoencoder(
            family, bottleneck, item, payload, device,
        )
        del payload
    guards.require_frozen_perception([model, *autoencoders.values()])
    guards.require_eval_mode([model, *autoencoders.values()])
    ledger = _Ledger()
    wrapped = {family: _AE(family, ae, ledger) for family, ae in autoencoders.items()}
    tail = ContextualFrozenP025TailAdapter(
        model=model, base=base, camera_registry=StaticCameraRegistry.audited(),
        device=device, ledger=ledger,
    )
    # The deadline gate wraps only the inference call. Serialization and
    # snapshot consumption still go to the unmodified frozen tail, so the
    # scientific model path and its pinned identity are unchanged.
    dispatch_tail = tail if deadline_guard is None else _DeadlineGuardedTail(tail, deadline_guard)
    runtime = PreloadedSplitEdgeRuntime(
        registry, frozen_p025_tail=dispatch_tail, ae_decoders=wrapped, tail_device=device,
        codec=ProductionSplitCodec(), output_serializer=tail.serialize,
        prepare_modules=False, startup_model_load_operations=4,
        startup_model_construction_operations=4, camera_registry=StaticCameraRegistry.audited(),
        require_frame_context=True,
    )
    return runtime, tail, ledger, [model, *autoencoders.values()]


def _prepare_live_input(frame_bgr: np.ndarray, radar_tensor: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (768, 448), interpolation=cv2.INTER_LINEAR)
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    rgb_tensor = rgb_tensor.to(device=device, dtype=torch.float32).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    radar = torch.from_numpy(np.ascontiguousarray(np.stack([
        cv2.resize(channel, (768, 448), interpolation=(cv2.INTER_NEAREST if index == 0 else cv2.INTER_LINEAR))
        for index, channel in enumerate(radar_tensor)
    ], axis=0))).unsqueeze(0)
    return torch.cat(((rgb_tensor - mean) / std, radar.to(device=device, dtype=torch.float32)), dim=1)


class LivePilotCellRuntime:
    """The UE half of one cell; its peer is the separately preloaded edge service."""

    def __init__(self, *, campaign: Mapping[str, Any], cell: Mapping[str, Any], attempt_dir: Path,
                 map_host: str, map_port: int, evidence_dir: Path) -> None:
        runtime = campaign["runtime"]
        _require(int(runtime["sfd1_protocol_version"]) == 2 and bool(runtime["frame_context_required"]), "SFD1 v2 frame context is required")
        _require(runtime["udp_fragment_header"] == "!IHH" and not bool(runtime["retransmission"]), "fragment contract drift")
        _require(bool(runtime["no_secondary_feature_compression"]), "secondary feature compression forbidden")
        _require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090", "expected CUDA device unavailable")
        self.campaign, self.cell, self.device = campaign, cell, torch.device("cuda:0")
        self.registry = SplitActionRegistry.from_runtime_binding()
        self.profile = self.registry.resolve(int(cell["action_id"]))
        _require(self.profile.profile_id == str(cell["profile_id"]), "cell/catalog identity mismatch")
        qualification = campaign.get("_qualification")
        self.allowed_action_ids = (
            tuple(int(value) for value in qualification["action_ids"])
            if isinstance(qualification, Mapping)
            else (self.profile.action_id,)
        )
        _require(
            len(self.allowed_action_ids) == len(set(self.allowed_action_ids))
            and all(0 <= value < 72 for value in self.allowed_action_ids),
            "live action allowlist is invalid",
        )
        self.allowed_profiles = {
            action_id: self.registry.resolve(action_id)
            for action_id in self.allowed_action_ids
        }
        self.ue, self._ledger, self._models = _preload_ue(self.device)
        self.attempt_dir, self.evidence_dir = Path(attempt_dir), Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        request = int(runtime["socket_buffer_request_bytes"])
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, request)
        self.sender.bind((str(runtime["ue_bind_host"]), 0))
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, request)
        self.receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.receiver.bind((str(runtime["ue_bind_host"]), int(runtime["camera_result_port"])))
        self.receiver.settimeout(0.1)
        self.map_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.remote = (str(runtime["edge_remote_host"]), int(runtime["edge_receive_port"]))
        self.map_remote, self.chunk_bytes = (str(map_host), int(map_port)), int(runtime["udp_chunk_bytes"])
        self.reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.metrics: dict[int, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.sent = self.completed = 0
        self.service_deadline_s = service_deadline_s(campaign)
        self.ack_timeout_s = ack_timeout_s(campaign)
        # Expensive work is abandoned at the feedback horizon.  Crossing the
        # earlier service target is reported as LATE_ACCEPTED, not hidden.
        self.deadline_s = self.ack_timeout_s
        self.counters = _Counters()
        self._published_frames: set[int] = set()
        self._stale_frames: dict[int, dict[str, Any]] = {}
        self.thread = threading.Thread(target=self._result_loop, name="splitfusion-sfd1-result", daemon=True)
        self.thread.start()

    def _record_stale(
        self,
        expired: DeadlineExpired,
        *,
        frame_id: int,
        capture_id: str,
        stream_id: str,
        profile: Any,
    ) -> dict[str, Any]:
        """Account a capture that expired before the UE would have sent it."""

        self.counters.bump("stale_before_send")
        self.counters.bump(f"deadline_drop_{expired.stage}")
        record = {
            "sent": False,
            "stale_stage": expired.stage,
            "stale_age_ms": expired.age_ms,
            "prepare_status": "STALE_BEFORE_SEND",
            "front_ms": "",
            "payload_bytes": "",
            "payload_bytes_uncompressed": "",
            "payload_chunks": "",
        }
        with self.lock:
            self._stale_frames[int(frame_id)] = {
                "capture_id": str(capture_id),
                "frame_id": int(frame_id),
                "stream_id": str(stream_id),
                "action_id": profile.action_id,
                "profile_id": profile.profile_id,
                "deadline_expiry_stage": expired.stage,
                "deadline_expiry_age_ms": expired.age_ms,
            }
        return record

    def socket_buffer_report(self) -> dict[str, int]:
        return {"requested_bytes": int(self.campaign["runtime"]["socket_buffer_request_bytes"]),
                "ue_send_reported_bytes": int(self.sender.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)),
                "ue_result_receive_reported_bytes": int(self.receiver.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))}

    def submit(self, *, frame_bgr: np.ndarray, radar_tensor: np.ndarray, frame_id: int,
               capture_timestamp_ns: int, ego_pose: tuple[float, float, float, float, float, float],
               stream_id: str, carla_timestamp: float, capture_id: str,
               action_id: int | None = None,
               on_commit: Callable[[], None] | None = None) -> dict[str, Any]:
        """Prepare and transmit one capture, or refuse it as already stale.

        ``on_commit`` runs exactly once, after the pre-transmission deadline
        gate has passed and before the first datagram leaves. A capture refused
        as stale therefore never reaches it, so the caller can register its
        feedback obligation there and never has to retract one.
        """

        _require(not self.errors, self.errors[0] if self.errors else "edge service failed")
        _require(self.thread.is_alive(), "result service exited")
        selected_action = self.profile.action_id if action_id is None else int(action_id)
        _require(selected_action in self.allowed_profiles, "action is outside the live allowlist")
        profile = self.allowed_profiles[selected_action]
        # The capture is already stale before any encode work: do not spend the
        # front/ranker/AE path or the radio on a frame that can never install.
        try:
            check_deadline(
                UE_STAGE_AFTER_PREPARATION, capture_timestamp_ns, self.deadline_s
            )
        except DeadlineExpired as expired:
            return self._record_stale(
                expired, frame_id=frame_id, capture_id=capture_id,
                stream_id=stream_id, profile=profile,
            )
        started = time.perf_counter_ns()
        input_7ch = _prepare_live_input(frame_bgr, radar_tensor, self.device)
        context = build_frame_context_v1(
            stream_id=stream_id, frame_id=int(frame_id), sequence_id=int(frame_id),
            capture_timestamp_ns=int(capture_timestamp_ns), ego_world_x=ego_pose[0], ego_world_y=ego_pose[1],
            ego_world_z=ego_pose[2], ego_world_pitch=ego_pose[3], ego_world_yaw=ego_pose[4], ego_world_roll=ego_pose[5],
        )
        with torch.inference_mode():
            prepared = self.ue.prepare(profile.action_id, input_7ch, sequence_id=context.sequence_id,
                                       capture_timestamp_ns=context.capture_timestamp_ns, frame_context=context)
        chunks = chunk_payload(prepared.wire_bytes, message_id=int(frame_id), chunk_bytes=self.chunk_bytes)
        sent_started = time.perf_counter_ns()
        # Immediately before transmission: encode is done, but if the capture
        # expired meanwhile the radio must not carry work that cannot install.
        try:
            check_deadline(UE_STAGE_BEFORE_SEND, capture_timestamp_ns, self.deadline_s)
        except DeadlineExpired as expired:
            return self._record_stale(
                expired, frame_id=frame_id, capture_id=capture_id,
                stream_id=stream_id, profile=profile,
            )
        if on_commit is not None:
            on_commit()
        with self.lock:
            self.metrics[int(frame_id)] = {
                "capture_id": str(capture_id), "frame_id": int(frame_id), "stream_id": str(stream_id),
                "action_id": profile.action_id, "profile_id": profile.profile_id,
                "model_family": profile.family, "quantizer": profile.quantizer,
                "q_e4": profile.q_e4, "routing_tag": profile.routing_tag,
                "carla_timestamp": float(carla_timestamp), "capture_started_ns": started,
                "ue_prepare_finished_ns": sent_started,
                "scientific_inner_bytes": int(prepared.inner_payload_bytes),
                "sfd1_overhead_bytes": int(prepared.outer_envelope_bytes), "sfd1_bytes": int(prepared.total_transmitted_bytes),
                "datagrams": len(chunks), "udp_application_bytes": sum(map(len, chunks)),
                "estimated_wire_bytes": sum(len(chunk) + 28 for chunk in chunks), "front_timing_ns": _trace_ns(prepared.timing),
            }
        for chunk in chunks:
            self.sender.sendto(chunk, self.remote)
            self.counters.bump("feature_datagrams_transmitted")
        sent_finished = time.perf_counter_ns()
        with self.lock:
            self.metrics[int(frame_id)]["send_finished_ns"] = sent_finished
            self.metrics[int(frame_id)]["service_deadline_at"] = deadline_at_s(
                capture_timestamp_ns, self.service_deadline_s
            )
            self.metrics[int(frame_id)]["ack_timeout_at"] = deadline_at_s(
                capture_timestamp_ns, self.ack_timeout_s
            )
            self.sent += 1
        self.counters.bump("feature_messages_transmitted")
        return {"sent": True, "front_ms": (sent_started - started) / 1e6,
                "payload_bytes": len(prepared.wire_bytes),
                "payload_bytes_uncompressed": prepared.inner_payload_bytes,
                "payload_chunks": len(chunks)}

    def _result_loop(self) -> None:
        """Ingest compact edge results; never block on a dense payload.

        The result message no longer carries the 720x1280 label map, so this
        loop performs only small-JSON work and returns to ``recvfrom``. The
        evaluation label map is persisted by the edge on its own evidence mount
        and is never required here to determine that the edge installed a map.
        """

        expired_seen = 0
        while not self.stop_event.is_set():
            try:
                datagram, address = self.receiver.recvfrom(65535)
            except socket.timeout:
                self.reassembler.expire(time.monotonic())
                if self.reassembler.expired_messages != expired_seen:
                    self.counters.bump(
                        "result_incomplete_reassemblies_expired",
                        self.reassembler.expired_messages - expired_seen,
                    )
                    expired_seen = self.reassembler.expired_messages
                continue
            except OSError:
                return
            received_ns = time.perf_counter_ns()
            received_wall = time.time()
            self.counters.bump("result_datagrams_received")
            try:
                complete = self.reassembler.ingest(str(address), datagram, received_at_s=time.monotonic())
                if complete is None:
                    continue
                self.counters.bump("result_messages_reassembled")
                value = json.loads(complete.payload.decode("utf-8"))
                _require(value.get("schema") == EDGE_RESULT_SCHEMA, "edge result schema drift")
                _require(int(value["frame_id"]) == complete.message_id, "result chunk/frame identity drift")
                frame_id = int(value["frame_id"])
                metric = self.metrics.get(frame_id)
                _require(metric is not None, "edge result has no transmitted UE frame")
                _require(int(value["action_id"]) == int(metric["action_id"]), "edge action identity drift")
                _require(str(value["profile_id"]) == str(metric["profile_id"]), "edge profile identity drift")
                _require(str(value["stream_id"]) == str(metric["stream_id"]), "edge stream identity drift")
                _require(
                    "semantic_labels_b64" not in value,
                    "dense evaluation label map must not ride the radio return path",
                )
                update = value["object_map_update"]
                _require(
                    update.get("schema") == OBJECT_MAP_UPDATE_SCHEMA
                    and int(update["frame_id"]) == frame_id
                    and str(update["stream_id"]) == str(metric["stream_id"]),
                    "object map update schema/identity drift",
                )
                terminal = value["edge_terminal_ack"]
                _require(
                    terminal.get("schema") == EDGE_TERMINAL_ACK_SCHEMA
                    and int(terminal["frame_id"]) == frame_id,
                    "edge terminal ACK schema/identity drift",
                )
                with self.lock:
                    duplicate = frame_id in self._published_frames
                if duplicate:
                    # A late or duplicate edge terminal must never reinstall an
                    # obsolete map for a frame already published.
                    self.counters.bump("duplicate_result_messages")
                    continue
                capture_timestamp_ns = int(value["capture_timestamp_ns"])
                try:
                    check_deadline(
                        UE_STAGE_BEFORE_MAP_PUBLICATION,
                        capture_timestamp_ns,
                        self.deadline_s,
                        now_s=received_wall,
                    )
                except DeadlineExpired as expired:
                    # FEATURE_RECEIVED is diagnostic only; an expired capture is
                    # never published, so it can never install an obsolete map.
                    self.counters.bump(f"deadline_drop_{expired.stage}")
                    self.counters.bump("results_expired_before_map_publication")
                    with self.lock:
                        metric.update(self._receipt_fields(
                            value, complete, received_ns, received_wall, terminal
                        ))
                        metric["map_publication_status"] = "EXPIRED_BEFORE_PUBLICATION"
                        metric["deadline_expiry_stage"] = expired.stage
                        metric["deadline_expiry_age_ms"] = expired.age_ms
                        self.completed += 1
                    continue
                published = {
                    "schema": "fusion_object_spatial_map.v1", "stream_id": value["stream_id"],
                    "frame_id": frame_id, "capture_id": metric["capture_id"],
                    "capture_timestamp": capture_timestamp_ns / 1_000_000_000.0,
                    "action_id": str(value["action_id"]), "carla_timestamp": metric["carla_timestamp"],
                    "objects": update["records"],
                    # Segmentation remains part of the edge spatial-map install;
                    # only its dense evaluation evidence left the radio payload.
                    "segmentation": {
                        "available": True,
                        "evidence": dict(terminal.get("evidence") or {}),
                        "installation_status": str(terminal.get("installation_status") or ""),
                    },
                    "timing": {"t_edge_recv_perf": float(value["edge_received_ns"]) / 1e9,
                               "t_tail_done_perf": float(value["tail_finished_ns"]) / 1e9,
                               "t_map_publish_perf": time.perf_counter()},
                }
                self.map_socket.sendto(zlib.compress(json.dumps(published, allow_nan=False, separators=(",", ":")).encode("utf-8"), level=1), self.map_remote)
                self.counters.bump("results_published_to_map")
                with self.lock:
                    self._published_frames.add(frame_id)
                    metric.update(self._receipt_fields(
                        value, complete, received_ns, received_wall, terminal
                    ))
                    metric["map_publication_status"] = "PUBLISHED"
                    self.completed += 1
            except Exception as exc:
                with self.lock:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                return

    @staticmethod
    def _receipt_fields(
        value: Mapping[str, Any],
        complete: Any,
        received_ns: int,
        received_wall: float,
        terminal: Mapping[str, Any],
    ) -> dict[str, Any]:
        """FEATURE_RECEIVED diagnostics plus the compact edge terminal ACK."""

        return {
            "edge_result_received_ns": received_ns,
            "feature_received_at": received_wall,
            "edge_timing_ns": value["edge_timing_ns"],
            "duplicate_datagrams": int(complete.duplicate_chunks),
            "decoded": True,
            "finite": bool(value["finite"]),
            "decoder_identity": str(value["decoder_identity"]),
            "edge_result_datagrams": int(complete.chunk_count),
            "feature_received_datagrams": int(value["feature_received_datagrams"]),
            "feature_duplicate_datagrams": int(value["feature_duplicate_datagrams"]),
            "reconstructed_device": str(value["reconstructed_device"]),
            "frame_context_valid": bool(value["frame_context_valid"]),
            "camera_pose_reconstruct_ns": int(value["camera_pose_reconstruct_ns"]),
            "finite_output_tensor_count": int(value["finite_output_tensor_count"]),
            "service_record_count": int(value["service_record_count"]),
            "edge_call_ledger": dict(value["edge_call_ledger"]),
            "edge_counters": dict(value["edge_counters"]),
            "edge_receipt_wall_s": terminal.get("edge_receipt_wall_s", ""),
            "edge_tail_complete_wall_s": terminal.get("tail_complete_wall_s", ""),
            "edge_evidence_install_wall_s": terminal.get("evidence_install_wall_s", ""),
            "edge_evidence_installation_status": str(terminal.get("installation_status") or ""),
            "edge_evidence_sha256": str((terminal.get("evidence") or {}).get("sha256") or ""),
            "edge_terminal_reason": str(terminal.get("terminal_reason") or ""),
        }

    def take_metric(self, frame_id: int) -> dict[str, Any] | None:
        with self.lock:
            value = self.metrics.get(int(frame_id))
            return dict(value) if value else None

    def close(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=10.0)
        buffers = self.socket_buffer_report()
        for item in (self.sender, self.receiver, self.map_socket):
            try:
                item.close()
            except OSError:
                pass
        with self.lock:
            stale = {int(key): dict(value) for key, value in self._stale_frames.items()}
            published = len(self._published_frames)
        return {"sent": self.sent, "edge_completed": self.completed, "result_thread_alive": self.thread.is_alive(),
                "errors": list(self.errors), "socket_buffers": buffers, "call_ledger": self._ledger.snapshot(),
                "ue_counters": self.ue.counters.__dict__,
                "service_deadline_s": self.service_deadline_s,
                "ack_timeout_s": self.ack_timeout_s,
                "processing_expiry_s": self.deadline_s,
                "results_published_to_map": published,
                "stale_before_send_frames": stale,
                "transport_counters": self.counters.snapshot()}

    def stale_frames(self) -> dict[int, dict[str, Any]]:
        with self.lock:
            return {int(key): dict(value) for key, value in self._stale_frames.items()}


def run_edge_service(*, config_path: Path, action_id: int, allowed_action_ids: tuple[int, ...],
                     ready_file: Path, edge_port: int, result_host: str, result_port: int,
                     evidence_dir: Path | None = None, run_id: str = "", cell_id: str = "") -> int:
    """Serve one cell's frozen tail with a bounded, deadline-enforced pipeline.

    The receive/reassembly path runs on its own thread and never blocks on tail
    inference or result transmission. Admitted frames wait in an explicit
    bounded latest-frame-first slot, so the kernel receive buffer can no longer
    act as a hidden byte-bounded FIFO whose depth in frames grows as the payload
    shrinks. Every expensive stage re-checks the 500 ms capture-based
    processing horizon; the distinct 100 ms service target is still carried
    and reported for timeliness classification.
    """

    campaign = _load_json(config_path)
    runtime = campaign["runtime"]
    _require(int(runtime["sfd1_protocol_version"]) == 2 and runtime["udp_fragment_header"] == "!IHH", "edge protocol binding drift")
    _require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090", "edge CUDA device unavailable")
    device = torch.device("cuda:0")
    registry = SplitActionRegistry.from_runtime_binding()
    _require(
        allowed_action_ids and len(allowed_action_ids) == len(set(allowed_action_ids)),
        "edge action allowlist is empty or duplicated",
    )
    profiles = {value: registry.resolve(value) for value in allowed_action_ids}
    _require(int(action_id) in profiles, "edge fixed action is outside its allowlist")
    service_s = service_deadline_s(campaign)
    deadline_s = ack_timeout_s(campaign)
    counters = _Counters()

    def guard(stage: str, capture_timestamp_ns: int) -> None:
        check_deadline(stage, capture_timestamp_ns, deadline_s)
        if stage == EDGE_STAGE_BEFORE_TAIL:
            # Counted only once the gate has passed, so "tail starts" means
            # inference actually began and the shortfall against process starts
            # is exactly the pre-tail deadline refusals.
            counters.bump("tail_starts")

    edge, tail, ledger, models = _preload_edge(device, deadline_guard=guard)
    evidence: EdgeEvaluationEvidenceWriter | None = None
    if evidence_dir is not None:
        evidence = EdgeEvaluationEvidenceWriter(Path(evidence_dir), counters)
    request = int(runtime["socket_buffer_request_bytes"])
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, request)
    receiver.bind(("0.0.0.0", int(edge_port)))
    receiver.settimeout(0.25)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, request)
    reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
    pending = LatestFramePendingSlot()
    stop_event = threading.Event()
    failures: list[str] = []
    counters_path = Path(ready_file).parent / "edge_counters.json"
    chunk_bytes = int(runtime["udp_chunk_bytes"])

    def publish_counters() -> None:
        """Persist edge counters outside the result path.

        The retry4 audit could not separate uplink loss from edge buffering
        because every edge counter rode inside a returned result. These
        counters are now durable on the edge's own mount regardless of whether
        any result survives the downlink.
        """

        try:
            _atomic_write_bytes(
                counters_path,
                json.dumps(
                    {
                        "schema": EDGE_COUNTERS_SCHEMA,
                        "run_id": str(run_id),
                        "cell_id": str(cell_id),
                        "action_id": int(action_id),
                        "service_deadline_s": service_s,
                        "ack_timeout_s": deadline_s,
                        "processing_expiry_s": deadline_s,
                        "pending_depth": pending.depth(),
                        "incomplete_reassemblies_expired": int(reassembler.expired_messages),
                        "reassembly_pending_messages": len(reassembler.pending),
                        "counters": counters.snapshot(),
                        "edge_operation_counters": dict(edge.counters.__dict__),
                        "call_ledger": ledger.snapshot(),
                        "failures": failures[:8],
                        "updated_at_unix_s": time.time(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        except Exception:
            counters.bump("edge_counter_publication_failed")

    expired_seen = 0

    def reconcile_expiries() -> None:
        """Attribute incomplete feature reassemblies the audit could not see.

        ``ChunkReassembler`` has no capacity cap, so a buffered partial message
        is only ever released by its timeout: eviction and expiry are the same
        event and are counted under both registered names.
        """

        nonlocal expired_seen
        observed = int(reassembler.expired_messages)
        if observed != expired_seen:
            delta = observed - expired_seen
            counters.bump("incomplete_reassemblies_expired", delta)
            counters.bump("reassembly_buffer_evictions", delta)
            expired_seen = observed

    def receive_loop() -> None:
        """Drain the socket and admit only fresh, complete feature messages."""

        while not stop_event.is_set():
            try:
                datagram, address = receiver.recvfrom(65535)
            except socket.timeout:
                reassembler.expire(time.monotonic())
                reconcile_expiries()
                continue
            except OSError:
                return
            counters.bump("feature_datagrams_received")
            try:
                complete = reassembler.ingest(
                    str(address), datagram, received_at_s=time.monotonic()
                )
            except ValueError:
                counters.bump("feature_datagrams_malformed")
                continue
            reconcile_expiries()
            counters.set_max("reassembly_pending_high_water", len(reassembler.pending))
            if complete is None:
                continue
            received_wall = time.time()
            received_ns = time.perf_counter_ns()
            counters.bump("feature_messages_reassembled")
            counters.bump("feature_datagrams_duplicate", int(complete.duplicate_chunks))
            try:
                outer = unpack_envelope(complete.payload)
            except Exception:
                counters.bump("feature_envelope_rejected")
                continue
            if outer.action_id not in profiles:
                counters.bump("feature_action_outside_allowlist")
                continue
            context = outer.frame_context
            if context is None:
                counters.bump("feature_missing_frame_context")
                continue
            try:
                check_deadline(
                    EDGE_STAGE_AFTER_REASSEMBLY,
                    outer.capture_timestamp_ns,
                    deadline_s,
                    now_s=received_wall,
                )
            except DeadlineExpired as expired:
                counters.bump(f"deadline_drop_{expired.stage}")
                continue
            item = {
                "payload": complete.payload,
                "action_id": int(outer.action_id),
                "message_id": int(complete.message_id),
                "capture_timestamp_ns": int(outer.capture_timestamp_ns),
                "chunk_count": int(complete.chunk_count),
                "duplicate_chunks": int(complete.duplicate_chunks),
                "edge_received_ns": received_ns,
                "edge_received_wall_s": received_wall,
            }
            admitted, displaced = pending.offer(
                str(context.stream_id), item, sequence=int(outer.sequence_id)
            )
            if not admitted:
                counters.bump("edge_admission_refused_not_freshest")
                continue
            counters.bump("edge_queue_admissions")
            counters.set_max("edge_pending_depth_high_water", pending.depth())
            if displaced is not None:
                counters.bump("edge_pending_replacements")

    def process_loop() -> None:
        """Decode, run the frozen tail and return one compact result."""

        while not stop_event.is_set():
            taken = pending.take(timeout=0.1)
            if taken is None:
                continue
            _stream_id, item = taken
            capture_timestamp_ns = int(item["capture_timestamp_ns"])
            try:
                check_deadline(
                    EDGE_STAGE_BEFORE_DECODE, capture_timestamp_ns, deadline_s
                )
            except DeadlineExpired as expired:
                counters.bump(f"deadline_drop_{expired.stage}")
                continue
            counters.bump("edge_process_starts")
            try:
                result = edge.process(
                    item["payload"], transmitted_action_id=int(item["action_id"])
                )
            except DeadlineExpired as expired:
                # Refused before inference, so no tail snapshot was produced.
                counters.bump(f"deadline_drop_{expired.stage}")
                continue
            except Exception as exc:
                counters.bump("edge_processing_failed")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            tail_finished_ns = time.perf_counter_ns()
            tail_finished_wall = time.time()
            counters.bump("tail_completions")
            # The frozen tail holds exactly one unconsumed snapshot and refuses
            # the next frame until it is taken, so it is consumed first and
            # unconditionally -- including when this result is then discarded.
            try:
                snapshot = tail.take_snapshot()
            except Exception as exc:
                counters.bump("edge_snapshot_failed")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            try:
                _finite_tree(result.perception)
            except Exception as exc:
                counters.bump("edge_nonfinite_perception")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            context = result.metadata.frame_context
            if context is None or context.frame_id != int(item["message_id"]):
                counters.bump("edge_frame_identity_rejected")
                failures.append("SFD1/chunk frame identity drift")
                continue
            profile = profiles[int(item["action_id"])]
            try:
                check_deadline(
                    EDGE_STAGE_BEFORE_PUBLICATION, capture_timestamp_ns, deadline_s
                )
            except DeadlineExpired as expired:
                counters.bump(f"deadline_drop_{expired.stage}")
                continue
            records = list(snapshot.records or ())
            installation_status = "EVALUATION_EVIDENCE_NOT_CONFIGURED"
            evidence_meta: dict[str, Any] = {}
            evidence_install_wall = ""
            if evidence is not None:
                labels = (
                    snapshot.semantic_labels.detach()
                    .to(device="cpu", dtype=torch.uint8)
                    .contiguous()
                    .numpy()
                )
                digest = hashlib.sha256(labels.tobytes()).hexdigest()
                evidence_meta = {
                    "evidence_name": segmentation_evidence_name(
                        context.stream_id, context.frame_id
                    ),
                    "schema": EVIDENCE_SIDECAR_SCHEMA,
                    "run_id": str(run_id),
                    "cell_id": str(cell_id),
                    "action_id": int(profile.action_id),
                    "profile_id": str(profile.profile_id),
                    "stream_id": str(context.stream_id),
                    "frame_id": int(context.frame_id),
                    "capture_timestamp_ns": int(context.capture_timestamp_ns),
                    "shape": [int(value) for value in labels.shape],
                    "dtype": str(labels.dtype),
                    "bytes": int(labels.nbytes),
                    "sha256": digest,
                }
                installation_status = evidence.submit(labels, evidence_meta)
                evidence_install_wall = time.time()
            terminal_ack = {
                "schema": EDGE_TERMINAL_ACK_SCHEMA,
                "run_id": str(run_id),
                "cell_id": str(cell_id),
                "action_id": int(profile.action_id),
                "stream_id": str(context.stream_id),
                "frame_id": int(context.frame_id),
                "capture_timestamp_ns": int(context.capture_timestamp_ns),
                "capture_wall_s": int(context.capture_timestamp_ns) / 1_000_000_000.0,
                "edge_receipt_wall_s": float(item["edge_received_wall_s"]),
                "tail_complete_wall_s": tail_finished_wall,
                "evidence_install_wall_s": evidence_install_wall,
                "service_deadline_at": deadline_at_s(capture_timestamp_ns, service_s),
                "ack_timeout_at": deadline_at_s(capture_timestamp_ns, deadline_s),
                "installation_status": installation_status,
                "evidence": evidence_meta,
                "terminal_reason": "EDGE_SERVICE_COMPLETE",
            }
            value = {
                "schema": EDGE_RESULT_SCHEMA,
                "action_id": profile.action_id,
                "profile_id": profile.profile_id,
                "decoder_identity": profile.decoder_identity,
                "stream_id": context.stream_id,
                "frame_id": context.frame_id,
                "capture_timestamp_ns": context.capture_timestamp_ns,
                "finite": True,
                "frame_context_valid": True,
                "reconstructed_device": str(edge.tail_device),
                "camera_pose_reconstruct_ns": int(snapshot.camera_pose_reconstruct_ns),
                "finite_output_tensor_count": int(snapshot.output_tensor_count),
                "service_record_count": len(records),
                "feature_received_datagrams": int(item["chunk_count"]),
                "feature_duplicate_datagrams": int(item["duplicate_chunks"]),
                "edge_call_ledger": ledger.snapshot(),
                "edge_counters": {**dict(edge.counters.__dict__), **counters.snapshot()},
                "edge_received_ns": int(item["edge_received_ns"]),
                "tail_finished_ns": tail_finished_ns,
                "edge_timing_ns": _trace_ns(result.timing),
                # Compact world-frame object/track state, versioned separately
                # from the terminal ACK so neither depends on the other.
                "object_map_update": {
                    "schema": OBJECT_MAP_UPDATE_SCHEMA,
                    "stream_id": str(context.stream_id),
                    "frame_id": int(context.frame_id),
                    "capture_timestamp_ns": int(context.capture_timestamp_ns),
                    "action_id": int(profile.action_id),
                    "records": records,
                },
                "edge_terminal_ack": terminal_ack,
            }
            payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
            chunks = chunk_payload(
                payload, message_id=context.frame_id, chunk_bytes=chunk_bytes
            )
            for chunk in chunks:
                sender.sendto(chunk, (str(result_host), int(result_port)))
                counters.bump("result_datagrams_transmitted")
            counters.bump("compact_results_transmitted")
            counters.bump("result_bytes_transmitted", len(payload))
            counters.set_max("result_datagrams_per_message_high_water", len(chunks))
            # Counters are published by the supervising loop's heartbeat, not
            # here: an fsync per frame would sit on the processing path.

    ready_file.parent.mkdir(parents=True, exist_ok=True)
    with ready_file.open("x", encoding="utf-8") as handle:
        json.dump({"schema": "splitfusion_live_edge_ready.v1", "action_id": int(action_id),
                   "allowed_action_ids": list(allowed_action_ids),
                   "profiles": {str(key): value.profile_id for key, value in profiles.items()},
                   "tail_device": str(edge.tail_device),
                   "state_root": str(ready_file.parent), "state_root_writable": True,
                   "service_deadline_s": service_s,
                   "ack_timeout_s": deadline_s,
                   "processing_expiry_s": deadline_s,
                   "dense_label_map_on_radio": False,
                   "edge_result_schema": EDGE_RESULT_SCHEMA,
                   "evaluation_evidence_dir": str(evidence_dir) if evidence_dir else "",
                   "edge_receive_reported_bytes": receiver.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
                   "edge_send_reported_bytes": sender.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)}, handle, sort_keys=True)
    publish_counters()
    receiver_thread = threading.Thread(target=receive_loop, name="edge-feature-receive", daemon=True)
    processor_thread = threading.Thread(target=process_loop, name="edge-tail-process", daemon=True)
    receiver_thread.start()
    processor_thread.start()
    try:
        while receiver_thread.is_alive() and processor_thread.is_alive():
            time.sleep(1.0)
            publish_counters()
        return 1
    finally:
        stop_event.set()
        for dropped in pending.close():
            counters.bump("edge_pending_dropped_at_shutdown")
            del dropped
        receiver_thread.join(timeout=2.0)
        processor_thread.join(timeout=5.0)
        if evidence is not None:
            evidence.close()
        publish_counters()
        receiver.close()
        sender.close()
        del models, ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="qualified SFD1-v2 live edge")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--action-id", type=int)
    parser.add_argument("--allowed-action-ids")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--edge-port", type=int, default=51002)
    parser.add_argument("--result-host", default="10.0.0.2")
    parser.add_argument("--result-port", type=int, default=51004)
    parser.add_argument("--edge-segmentation-evidence-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--cell-id", default="")
    args, _ignored = parser.parse_known_args(argv)
    _require(args.edge and args.config and args.action_id is not None and args.ready_file, "edge mode and all qualified bindings are required")
    allowed_action_ids = tuple(
        int(value)
        for value in str(args.allowed_action_ids or args.action_id).split(",")
    )
    return run_edge_service(config_path=args.config.resolve(strict=True), action_id=int(args.action_id),
                            allowed_action_ids=allowed_action_ids,
                            ready_file=args.ready_file, edge_port=int(args.edge_port),
                            result_host=str(args.result_host), result_port=int(args.result_port),
                            evidence_dir=args.edge_segmentation_evidence_dir,
                            run_id=str(args.run_id), cell_id=str(args.cell_id))


if __name__ == "__main__":
    raise SystemExit(main())
