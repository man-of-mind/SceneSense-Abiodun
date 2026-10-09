"""Injected, offline same-frame replay with strict selected-branch parity gates.

This module loads no models, starts no services and chooses no device. Callers
own checkpoint/codec qualification and durable input hashes. The dense reference
is the *unmodified FP32 C2* from the frozen front: no AE, ranker, spatial drop or
quantization is applied to that branch. The frozen tail's ``dense=False`` option
is compatible with this definition; that option disables auxiliary depth only.

Callbacks receive private tensor copies. ``front`` accepts the saved CPU
``[1,7,448,768]`` input and returns C2. ``encode_selected(c2, envelope)`` must
reproduce the live inner codec bytes. ``decode_selected(inner, envelope)``
decodes the saved bytes. ``tail(c2, metadata)`` returns :class:`Prediction` with
native labels and complete contextual records. Codec/tail inputs are unbatched
``[256,112,192]`` FP32 tensors. Both tails receive the same Run4 metadata.

Replay first proves the selected payload and archived selected predictions.
Only then is the dense branch inferred. No quality or latency claim is made by
this core; use the retained complete-target evaluator for paired measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as wire
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as protocol
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.continuous_execution_v2 import ExecutionEnvelopeV3
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2 import Run4TailMetadataV2


SCHEMA = "splitfusion.paired_quality_replay.v1"
DENSE_DEFINITION = "FROZEN_FRONT_FP32_C2_NO_AE_NO_SPATIAL_DROP_NO_QUANTIZATION"


class ReplayParityError(ValueError):
    """The saved live frame cannot support a faithful paired replay."""


@dataclass(frozen=True)
class Prediction:
    labels: np.ndarray
    records: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class ReplayResult:
    selected: Prediction
    dense: Prediction
    metadata: Run4TailMetadataV2
    evidence: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayParityError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _c2(value: torch.Tensor, label: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{label} must be a tensor")
    _require(value.dtype == torch.float32, f"{label} must be FP32")
    if tuple(value.shape) == (1, 256, 112, 192):
        value = value[0]
    _require(tuple(value.shape) == (256, 112, 192), f"{label} C2 shape differs")
    _require(bool(torch.isfinite(value).all()), f"{label} contains non-finite C2")
    return value.detach().clone().contiguous()


def _prediction(value: Prediction, metadata: Run4TailMetadataV2, label: str) -> Prediction:
    _require(isinstance(value, Prediction), f"{label} must return Prediction")
    labels = value.labels
    _require(isinstance(labels, np.ndarray) and labels.dtype == np.uint8
             and labels.shape == (720, 1280), f"{label} native labels must be 720x1280 uint8")
    _require(bool(np.all(labels <= 2)), f"{label} labels must be 0/1/2")
    _require(isinstance(value.records, (list, tuple)) and len(value.records) <= 512,
             f"{label} requires bounded complete prediction records")
    try:
        records = json.loads(_canonical(list(value.records)))
    except (TypeError, ValueError) as exc:
        raise ReplayParityError(f"{label} records are not finite JSON") from exc
    context = metadata.frame_context
    for row in records:
        _require(isinstance(row, dict), f"{label} record must be an object")
        _require(type(row.get("frame_id")) is int and row["frame_id"] == context.frame_id
                 and row.get("stream_id") == context.stream_id
                 and type(row.get("capture_timestamp_ns")) is int
                 and row["capture_timestamp_ns"] == context.capture_timestamp_ns,
                 f"{label} prediction context differs")
    # Bytes-backed arrays cannot be made writable by a downstream branch.
    owned = np.frombuffer(np.ascontiguousarray(labels).tobytes(), dtype=np.uint8).reshape(labels.shape)
    return Prediction(owned, tuple(records))


def replay_pair(
    *,
    prepared_input: torch.Tensor,
    selected_wire: bytes,
    expected_identity: Mapping[str, Any],
    archived_selected: Prediction,
    front: Callable[[torch.Tensor], torch.Tensor],
    encode_selected: Callable[[torch.Tensor, ExecutionEnvelopeV3], bytes],
    decode_selected: Callable[[bytes, ExecutionEnvelopeV3], torch.Tensor],
    tail: Callable[[torch.Tensor, Run4TailMetadataV2], Prediction],
) -> ReplayResult:
    """Replay one identity-bound pair, refusing either selected parity failure.

    Exact byte parity is intentional: a numerically similar re-encoding does
    not establish that the archived input was the selected live model input.
    The caller must validate ``expected_identity`` against its input archive
    and the GT/AVO identities before calling this function.
    """
    _require(isinstance(selected_wire, bytes) and bool(selected_wire), "saved SFD4 bytes required")
    try:
        envelope, context = wire.unpack_sfd4(selected_wire)
        protocol.validate_run4_identity(expected_identity)
    except Exception as exc:
        raise ReplayParityError(f"saved wire or expected identity invalid: {exc}") from exc
    identity = protocol.run4_identity(envelope)
    _require(dict(expected_identity) == identity, "saved wire/action identity differs")
    metadata = Run4TailMetadataV2(
        frame_context=context, sequence_id=context.sequence_id,
        capture_timestamp_ns=context.capture_timestamp_ns,
        run4_identity=MappingProxyType(identity),
    )
    archived = _prediction(archived_selected, metadata, "archived selected")
    _require(isinstance(prepared_input, torch.Tensor), "prepared input must be a tensor")
    _require(prepared_input.device.type == "cpu" and prepared_input.dtype == torch.float32
             and tuple(prepared_input.shape) == (1, 7, 448, 768),
             "saved prepared input must be CPU FP32 [1,7,448,768]")
    _require(bool(torch.isfinite(prepared_input).all()), "saved input is non-finite")
    _require(bool(torch.count_nonzero(prepared_input[:, :, 432:, :]) == 0),
             "saved input differs from zero-after-normalization bottom padding")
    input_copy = prepared_input.detach().clone().contiguous()
    input_sha = _sha(input_copy.numpy().tobytes())
    with torch.inference_mode():
        dense_c2 = _c2(front(input_copy), "frozen front")
        regenerated = encode_selected(dense_c2.clone(), envelope)
        _require(isinstance(regenerated, bytes), "selected encoder must return inner bytes")
        _require(regenerated == envelope.inner_payload,
                 "selected inner payload byte parity failed; paired replay refused")
        reconstructed = _c2(decode_selected(envelope.inner_payload, envelope), "selected decoder")
        selected = _prediction(tail(reconstructed.clone(), metadata), metadata, "replayed selected")
        _require(np.array_equal(selected.labels, archived.labels),
                 "selected native-label parity failed; dense replay refused")
        selected_records = _canonical(list(selected.records))
        _require(selected_records == _canonical(list(archived.records)),
                 "selected complete-record parity failed; dense replay refused")
        dense = _prediction(tail(dense_c2.clone(), metadata), metadata, "dense")
    return ReplayResult(selected=selected, dense=dense, metadata=metadata, evidence={
        "schema": SCHEMA, "dense_definition": DENSE_DEFINITION,
        "action_identity": dict(identity), "prepared_input_array_sha256": input_sha,
        "selected_wire_sha256": _sha(selected_wire),
        "selected_inner_payload_sha256": _sha(envelope.inner_payload),
        "selected_inner_payload_byte_identical": True,
        "selected_native_labels_identical": True,
        "selected_complete_records_identical": True,
        "selected_labels_array_sha256": _sha(selected.labels.tobytes()),
        "selected_records_sha256": _sha(selected_records),
        "dense_labels_array_sha256": _sha(dense.labels.tobytes()),
        "dense_records_sha256": _sha(_canonical(list(dense.records))),
        "same_frame_context": True, "live_latency_measurement": False,
    })
