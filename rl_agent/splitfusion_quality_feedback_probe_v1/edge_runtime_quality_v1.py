#!/usr/bin/env python3
"""Add exact CARLA-quality progress feedback to the direct edge runtime.

This wrapper runs inside ``oai-perception-rx``.  It imports the qualified
direct edge service unchanged and installs two additive seams:

* an optional, lifetime-safe semantic-label branch launched after
  ``decode_tail`` on a dedicated CUDA stream; and
* a bounded privileged evaluator that emits one compact, record-free
  non-terminal progress datagram through :class:`UEControlSender`.

The production map document, records, label map and terminal map ledger are
not modified.  CARLA ground truth reaches this diagnostic only through the
per-cell shared evidence mount and is never placed on the radio.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rl_agent.splitfusion_direct_edge_map_v1 import (
    live_pilot_runtime_direct_v1 as direct_runtime,
)
from rl_agent.splitfusion_direct_edge_map_v1.direct_v3_edge import (
    preload_direct_v3_edge as qualified_preload,
)
from rl_agent.splitfusion_direct_edge_map_v1.edge_publisher import UEControlSender
from rl_agent.splitfusion_live_dispatch_v1.envelope import unpack_envelope
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry

from . import protocol
from .coordinator import BoundedQualityCoordinator
from .scoring import immutable_mask
from .signal_shutdown import run_with_shutdown_handlers


QUALITY_EDGE_REPORT_SCHEMA = "splitfusion_privileged_quality_edge_report.v1"


class QualityEdgeRuntimeError(RuntimeError):
    pass


QUALITY_CONFIG_PATH = Path("/work/torch_cache/quality_probe_config.json")


class _EarlySemanticTap:
    """Launch an owned semantic label-map copy without unsafe stream reuse."""

    def __init__(
        self,
        coordinator: BoundedQualityCoordinator,
        *,
        source_hw: tuple[int, int] = (720, 1280),
        enabled: bool,
    ) -> None:
        self.coordinator = coordinator
        self.source_hw = tuple(int(value) for value in source_hw)
        self.enabled = bool(enabled and torch.cuda.is_available())
        self._identity = threading.local()
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="early-semantic-copy"
        )
        self._observer_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="model-ready-observer"
        )
        self._stream = torch.cuda.Stream() if self.enabled else None
        self._handles: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.failures: list[str] = []

    @staticmethod
    def _key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(identity[name] for name in protocol.REQUIRED_IDENTITY)

    def enter(self, identity: Mapping[str, Any]) -> None:
        self._identity.value = dict(identity)

    def leave(self) -> None:
        self._identity.value = None

    def wrap_decode(self, original: Any) -> Any:
        def decode_tail(*args: Any, **kwargs: Any) -> Any:
            identity = getattr(self._identity, "value", None)
            start_wall_ns, start_monotonic_ns = time.time_ns(), time.monotonic_ns()
            outputs = original(*args, **kwargs)
            if identity is None:
                self.failures.append("decode_tail ran without frame identity")
                return outputs
            try:
                finish_event = (
                    torch.cuda.Event(enable_timing=True) if self.enabled else None
                )
                if finish_event is not None:
                    finish_event.record(torch.cuda.current_stream())
                if self.enabled:
                    future, copy_timing = self._launch_cuda_copy(
                        outputs["semantic_logits"], finish_event
                    )
                else:
                    future = Future()
                    copy_timing = {}
                model_ready_future = (
                    self._observer_pool.submit(self._observe_model_ready, finish_event)
                    if finish_event is not None else None
                )
                diagnostic_error = ""
            except Exception as exc:
                # The production decode result remains authoritative.
                future = Future()
                future.set_exception(exc)
                copy_timing = {}
                model_ready_future = None
                diagnostic_error = f"{type(exc).__name__}:{exc}"
                self.failures.append(diagnostic_error)
            handle = {
                "identity": dict(identity),
                "mask_future": future,
                "model_dispatch_start_wall_ns": start_wall_ns,
                "model_dispatch_start_monotonic_ns": start_monotonic_ns,
                # A separate executor prevents a full-label D2H copy from
                # delaying the model-ready observation stamp.
                "model_ready_future": model_ready_future,
                "model_ready_event": finish_event,
                "semantic_copy_timing": copy_timing,
                "diagnostic_error": diagnostic_error,
            }
            key = self._key(identity)
            with self._lock:
                if key in self._handles:
                    self.failures.append(f"duplicate model handle: {key}")
                else:
                    self._handles[key] = handle
            if self.enabled and not diagnostic_error:
                try:
                    self.coordinator.submit_early_segmentation(identity, future)
                except Exception as exc:
                    handle["diagnostic_error"] = f"{type(exc).__name__}:{exc}"
                    self.failures.append(handle["diagnostic_error"])
            return outputs

        return decode_tail

    @staticmethod
    def _observe_model_ready(event: torch.cuda.Event | None) -> tuple[int, int]:
        if event is not None:
            event.synchronize()
        return time.time_ns(), time.monotonic_ns()

    def _launch_cuda_copy(
        self,
        semantic_logits: torch.Tensor,
        finish_event: torch.cuda.Event,
    ) -> tuple[Future[np.ndarray], dict[str, int]]:
        assert self._stream is not None
        producer = torch.cuda.current_stream(device=semantic_logits.device)
        producer_done = torch.cuda.Event()
        producer_done.record(producer)
        cpu = torch.empty(self.source_hw, dtype=torch.uint8, pin_memory=True)
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(producer_done)
            # Explicit ownership on the side stream prevents the allocator
            # lifetime bug previously seen in the deferred finite validator.
            semantic_logits.record_stream(self._stream)
            labels = F.interpolate(
                semantic_logits.float(),
                size=self.source_hw,
                mode="bilinear",
                align_corners=False,
            ).argmax(1)[0].to(dtype=torch.uint8)
            labels.record_stream(self._stream)
            cpu.copy_(labels, non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record(self._stream)

        copy_timing: dict[str, int] = {
            "semantic_branch_submitted_wall_ns": time.time_ns()
        }

        def finish() -> np.ndarray:
            # Closure owns logits/labels/cpu until copy_done; this is as
            # important as the stream dependency itself.
            keepalive = (semantic_logits, labels, cpu, finish_event)
            copy_done.synchronize()
            copy_timing["semantic_branch_copy_ready_wall_ns"] = time.time_ns()
            del keepalive
            return immutable_mask(cpu.numpy())

        return self._pool.submit(finish), copy_timing

    def copy_production_mask(
        self, semantic_labels: torch.Tensor
    ) -> Future[np.ndarray]:
        """Own a bounded-sample copy of the authoritative production mask.

        This is deliberately called for the parity sample only.  It does not
        put a second full-label D2H copy on every measured frame.
        """

        if not self.enabled:
            result: Future[np.ndarray] = Future()
            try:
                value = (
                    semantic_labels.detach()
                    .to(device="cpu", dtype=torch.uint8)
                    .contiguous()
                    .numpy()
                )
                result.set_result(immutable_mask(value))
            except Exception as exc:
                result.set_exception(exc)
            return result

        assert self._stream is not None
        producer = torch.cuda.current_stream(device=semantic_labels.device)
        producer_done = torch.cuda.Event()
        producer_done.record(producer)
        cpu = torch.empty(
            tuple(int(value) for value in semantic_labels.shape[-2:]),
            dtype=torch.uint8,
            pin_memory=True,
        )
        labels = semantic_labels.detach()
        if labels.ndim == 3 and int(labels.shape[0]) == 1:
            labels = labels[0]
        if labels.ndim != 2:
            result = Future()
            result.set_exception(
                QualityEdgeRuntimeError(
                    f"production semantic labels must be HxW, got {tuple(labels.shape)}"
                )
            )
            return result
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(producer_done)
            labels.record_stream(self._stream)
            converted = labels.to(dtype=torch.uint8)
            converted.record_stream(self._stream)
            cpu.copy_(converted, non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record(self._stream)

        def finish() -> np.ndarray:
            keepalive = (semantic_labels, labels, converted, cpu)
            copy_done.synchronize()
            del keepalive
            return immutable_mask(cpu.numpy())

        return self._pool.submit(finish)

    def take(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        key = self._key(identity)
        with self._lock:
            handle = self._handles.pop(key, None)
        if handle is None:
            raise QualityEdgeRuntimeError(f"missing model handle: {key}")
        return handle

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)
        self._observer_pool.shutdown(wait=True, cancel_futures=False)


class _DecodeModelProxy:
    def __init__(self, model: Any, tap: _EarlySemanticTap) -> None:
        self._model = model
        self.decode_tail = tap.wrap_decode(model.decode_tail)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class _EdgeProxy:
    def __init__(
        self,
        delegate: Any,
        tap: _EarlySemanticTap,
        *,
        run_id: str,
        cell_id: str,
        registry: SplitActionRegistry,
    ) -> None:
        self._delegate = delegate
        self._tap = tap
        self._run_id = run_id
        self._cell_id = cell_id
        self._registry = registry
        self._identity_by_frame: dict[int, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def process(self, frame_bytes: Any, *, transmitted_action_id: int) -> Any:
        outer = unpack_envelope(frame_bytes)
        context = outer.frame_context
        if context is None:
            raise QualityEdgeRuntimeError("quality probe requires SFD1 frame context")
        profile = self._registry.resolve(int(transmitted_action_id))
        identity = {
            "run_id": self._run_id,
            "cell_id": self._cell_id,
            "stream_id": str(context.stream_id),
            "frame_id": int(context.frame_id),
            "action_id": int(profile.action_id),
            "profile_id": str(profile.profile_id),
            "capture_timestamp_ns": int(context.capture_timestamp_ns),
        }
        self._tap.enter(identity)
        try:
            result = self._delegate.process(
                frame_bytes, transmitted_action_id=transmitted_action_id
            )
        finally:
            self._tap.leave()
        result_context = result.metadata.frame_context
        if (
            result_context is None
            or str(result_context.stream_id) != identity["stream_id"]
            or int(result_context.frame_id) != identity["frame_id"]
            or int(result.metadata.action_id) != identity["action_id"]
        ):
            raise QualityEdgeRuntimeError("quality prediction identity drift")
        self._identity_by_frame[int(context.frame_id)] = identity
        return result

    def pop_identity(self, frame_id: int) -> dict[str, Any]:
        try:
            return self._identity_by_frame.pop(int(frame_id))
        except KeyError as exc:
            raise QualityEdgeRuntimeError(
                f"missing completed prediction identity: {frame_id}"
            ) from exc


class _TailProxy:
    def __init__(
        self,
        delegate: Any,
        edge: _EdgeProxy,
        tap: _EarlySemanticTap,
        coordinator: BoundedQualityCoordinator,
    ) -> None:
        self._delegate = delegate
        self._edge = edge
        self._tap = tap
        self._coordinator = coordinator
        self._parity_masks_scheduled = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def take_snapshot(self) -> Any:
        snapshot = self._delegate.take_snapshot()
        identity: dict[str, Any] | None = None
        try:
            context = snapshot.frame_context
            if context is None:
                raise QualityEdgeRuntimeError("final prediction lacks frame context")
            identity = self._edge.pop_identity(int(context.frame_id))
            handle = self._tap.take(identity)
            if handle.get("diagnostic_error"):
                raise QualityEdgeRuntimeError(str(handle["diagnostic_error"]))
            final_wall_ns, final_monotonic_ns = time.time_ns(), time.monotonic_ns()
            model_ready_future = handle.get("model_ready_future")
            if model_ready_future is None:
                raise QualityEdgeRuntimeError("model-ready CUDA event was unavailable")
            if model_ready_future.done():
                model_ready_wall_ns, model_ready_monotonic_ns = model_ready_future.result()
            elif handle["model_ready_event"].query():
                # A query is nonblocking. This is an observed-ready upper bound,
                # not a fabricated kernel-finish timestamp.
                model_ready_wall_ns, model_ready_monotonic_ns = (
                    time.time_ns(), time.monotonic_ns()
                )
            else:
                raise QualityEdgeRuntimeError(
                    "model completion was not observable without blocking map publication"
                )
            mask_future = handle["mask_future"]
            production_mask_for_parity = None
            if self._parity_masks_scheduled < self._coordinator.parity_sample_limit:
                production_mask_for_parity = self._tap.copy_production_mask(
                    snapshot.semantic_labels
                )
                self._parity_masks_scheduled += 1
            if not self._tap.enabled:
                value = (
                    snapshot.semantic_labels.detach()
                    .to(device="cpu", dtype=torch.uint8)
                    .contiguous()
                    .numpy()
                )
                mask_future.set_result(immutable_mask(value))
            self._coordinator.submit_final(
                identity=identity,
                frozen_carla_frame_id=int(context.frame_id),
                records=tuple(snapshot.records or ()),
                final_mask=mask_future,
                production_mask_for_parity=production_mask_for_parity,
                model_ready_wall_ns=model_ready_wall_ns,
                model_ready_monotonic_ns=model_ready_monotonic_ns,
                final_prediction_ready_wall_ns=final_wall_ns,
                final_prediction_ready_monotonic_ns=final_monotonic_ns,
                upstream_timing={
                    "model_dispatch_start_wall_ns": handle[
                        "model_dispatch_start_wall_ns"
                    ],
                    **dict(handle.get("semantic_copy_timing") or {}),
                },
            )
        except Exception as exc:
            # All diagnostic work is isolated after the production snapshot.
            # If full identity is available, emit an explicit failed progress
            # event; otherwise the edge report fails the live evidence gate.
            if identity is not None:
                try:
                    self._coordinator.submit_failure(
                        identity=identity,
                        frozen_carla_frame_id=int(identity["frame_id"]),
                        reason=f"TAIL_PROBE:{type(exc).__name__}:{exc}",
                    )
                except Exception:
                    pass
            self._coordinator.failures.append(
                f"TAIL_PROBE_ISOLATED:{type(exc).__name__}:{exc}"
            )
        return snapshot


class _QualityRuntime:
    def __init__(self) -> None:
        config_path = Path(
            os.environ.get("SPLITFUSION_QUALITY_CONFIG", str(QUALITY_CONFIG_PATH))
        )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QualityEdgeRuntimeError(
                f"cannot load quality probe config {config_path}: {exc}"
            ) from exc
        if config.get("schema") != "splitfusion_privileged_quality_probe_config.v1":
            raise QualityEdgeRuntimeError("quality probe config schema drift")
        self.run_id = str(config["run_id"])
        self.cell_id = str(config["cell_id"])
        self.evidence_dir = Path(str(config["evidence_dir"]))
        self.report_path = Path(str(config["report_path"]))
        ue_host = str(config["ue_host"])
        ue_port = int(config["ue_port"])
        self.sender = UEControlSender(ue_host=ue_host, ue_port=ue_port)
        self.sent_rows: list[dict[str, Any]] = []
        self.detail_rows: list[dict[str, Any]] = []
        self.validation_rows: list[dict[str, Any]] = []
        self._send_lock = threading.Lock()
        self.coordinator = BoundedQualityCoordinator(
            gt_directory=self.evidence_dir,
            emit_ack=self._emit,
            record_detail=self._record_detail,
            record_validation=self._record_validation,
            match_distance_m=float(config.get("match_distance_m", 3.0)),
            queue_depth=int(config.get("queue_depth", 64)),
            gt_timeout_s=float(config.get("gt_timeout_s", 2.0)),
            parity_sample_limit=int(config.get("parity_sample_limit", 8)),
        )
        self.tap = _EarlySemanticTap(
            self.coordinator,
            enabled=bool(config.get("early_segmentation", True)),
        )
        self.registry = SplitActionRegistry.from_runtime_binding()
        self.closed = False

    def _record_detail(self, document: Mapping[str, Any]) -> None:
        with self._send_lock:
            self.detail_rows.append(dict(document))

    def _record_validation(self, document: Mapping[str, Any]) -> None:
        with self._send_lock:
            self.validation_rows.append(dict(document))

    def _emit(self, document: Mapping[str, Any]) -> None:
        # Refresh the stamp immediately before validation/encoding/send.  The
        # remaining local encode-to-send interval is separately bounded by the
        # compact payload size and can be measured from edge instrumentation.
        value = dict(document)
        protocol.validate(value)
        encoded = protocol.canonical_bytes(value)
        socket_send_call_wall_ns = time.time_ns()
        socket_send_call_monotonic_ns = time.monotonic_ns()
        try:
            sent = self.sender.socket.sendto(encoded, self.sender.remote)
        except OSError as exc:
            with self.sender._lock:
                self.sender.counters["ue_control_send_failed"] += 1
            raise QualityEdgeRuntimeError(f"quality ACK send failed: {exc}") from exc
        if sent != len(encoded):
            raise QualityEdgeRuntimeError("quality ACK byte count drift")
        with self.sender._lock:
            self.sender.counters["ue_control_messages_sent"] += 1
            self.sender.counters["ue_control_bytes_sent"] += len(encoded)
        with self._send_lock:
            self.sent_rows.append(
                {
                    "identity": list(protocol.identity(value)),
                    "event": value["e"],
                    "bytes": sent,
                    "sha256": protocol.digest(value),
                    "detail_sha256": value["dh"],
                    "ack_emit_start_wall_ns": protocol.timing_dict(value)[
                        "ack_emit_start_wall_ns"
                    ],
                    "socket_send_call_wall_ns": socket_send_call_wall_ns,
                    "socket_send_call_monotonic_ns": socket_send_call_monotonic_ns,
                    "socket_local": list(self.sender.socket.getsockname()),
                    "socket_remote": list(self.sender.remote),
                }
            )

    def install(self, device: torch.device, *, deadline_guard: Any = None) -> Any:
        edge, tail, ledger, models = qualified_preload(
            device, deadline_guard=deadline_guard
        )
        # Replace only the model surface used by this diagnostic tail object.
        # Every attribute other than decode_tail delegates byte-for-byte.
        tail._tail._model = _DecodeModelProxy(tail._tail._model, self.tap)
        edge_proxy = _EdgeProxy(
            edge,
            self.tap,
            run_id=self.run_id,
            cell_id=self.cell_id,
            registry=self.registry,
        )
        tail_proxy = _TailProxy(tail, edge_proxy, self.tap, self.coordinator)
        return edge_proxy, tail_proxy, ledger, models

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        coordinator = self.coordinator.close(timeout_s=15.0)
        self.tap.close()
        report = {
            "schema": QUALITY_EDGE_REPORT_SCHEMA,
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "privileged_carla_ground_truth": True,
            "deployable_feedback": False,
            "early_segmentation_enabled": self.tap.enabled,
            "tap_failures": list(self.tap.failures),
            "coordinator": coordinator,
            "sender": self.sender.snapshot(),
            "messages": list(self.sent_rows),
            "details": list(self.detail_rows),
            "serial_validations": list(self.validation_rows),
            "source_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "scoring.py",
                    "gt_evidence.py",
                    "protocol.py",
                    "coordinator.py",
                )
            },
            "updated_at_unix_s": time.time(),
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.report_path)
        self.sender.close()


_RUNTIME: _QualityRuntime | None = None


def _preload(device: torch.device, *, deadline_guard: Any = None) -> Any:
    global _RUNTIME
    if _RUNTIME is not None:
        raise QualityEdgeRuntimeError("quality runtime preloaded twice")
    _RUNTIME = _QualityRuntime()
    return _RUNTIME.install(device, deadline_guard=deadline_guard)


def _run_with_close(**kwargs: Any) -> int:
    try:
        return _ORIGINAL_RUN(**kwargs)
    finally:
        if _RUNTIME is not None:
            _RUNTIME.close()


_ORIGINAL_RUN = direct_runtime.run_direct_edge_service
direct_runtime.preload_direct_v3_edge = _preload
direct_runtime.run_direct_edge_service = _run_with_close


def main(argv: Sequence[str] | None = None) -> int:
    # The handler raises on the main thread.  That unwinds the qualified
    # service and therefore executes _run_with_close before Docker exits.
    return run_with_shutdown_handlers(lambda: direct_runtime.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
