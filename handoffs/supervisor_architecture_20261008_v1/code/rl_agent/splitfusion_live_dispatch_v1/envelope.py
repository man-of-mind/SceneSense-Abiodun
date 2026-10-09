"""Minimal control envelope around one unchanged SplitFusion inner payload.

The existing UDP transport's ``!IHH`` header is a chunk/reassembly header, not
an action contract.  This envelope is intended to become that transport's
reassembled message body: it adds the minimum routing and capture identity
without changing or recompressing the scientific inner zstd frame.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .frame_context import (
    FRAME_CONTEXT_VERSION,
    FrameContextV1,
    Pose6D,
    validate_frame_context,
)


MAGIC = b"SFD1"
PROTOCOL_VERSION = 1  # Historical Phase-13B default.
CONTEXT_PROTOCOL_VERSION = 2
HEADER = struct.Struct("<4sHHIQQQ")
HEADER_BYTES = HEADER.size  # 36-byte common/v1 header.
CONTEXT_FIXED = struct.Struct("<HHIQQQ6d32s32s")
CONTEXT_FIXED_BYTES = CONTEXT_FIXED.size  # 144 bytes before stream identity.
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class EnvelopeError(ValueError):
    """The outer control envelope is malformed or unsupported."""


@dataclass(frozen=True)
class SplitEnvelope:
    protocol_version: int
    action_id: int
    sequence_id: int
    capture_timestamp_ns: int
    inner_payload_length: int
    inner_payload: bytes
    header_bytes: int
    frame_context: FrameContextV1 | None = None

    @property
    def control_overhead_bytes(self) -> int:
        return self.header_bytes

    @property
    def total_transmitted_bytes(self) -> int:
        return self.header_bytes + self.inner_payload_length


def _unsigned(value: int, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EnvelopeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise EnvelopeError(f"{name} is outside its unsigned wire range")
    return int(value)


def pack_envelope(
    inner_payload: bytes | bytearray | memoryview,
    *,
    action_id: int,
    sequence_id: int,
    capture_timestamp_ns: int,
    frame_context: FrameContextV1 | None = None,
) -> bytes:
    payload = bytes(inner_payload)
    if not payload:
        raise EnvelopeError("inner payload must be non-empty")
    action = _unsigned(action_id, UINT32_MAX, "action_id")
    sequence = _unsigned(sequence_id, UINT64_MAX, "sequence_id")
    captured = _unsigned(capture_timestamp_ns, UINT64_MAX, "capture_timestamp_ns")
    version = PROTOCOL_VERSION
    extension = b""
    if frame_context is not None:
        try:
            stream = validate_frame_context(frame_context)
        except Exception as exc:
            raise EnvelopeError(str(exc)) from exc
        if frame_context.sequence_id != sequence:
            raise EnvelopeError("frame-context sequence mismatch")
        if frame_context.capture_timestamp_ns != captured:
            raise EnvelopeError("frame-context timestamp mismatch")
        context_bytes = CONTEXT_FIXED_BYTES + len(stream)
        extension = CONTEXT_FIXED.pack(
            FRAME_CONTEXT_VERSION,
            context_bytes,
            len(stream),
            frame_context.frame_id,
            frame_context.sequence_id,
            frame_context.capture_timestamp_ns,
            *frame_context.ego_world.values(),
            bytes.fromhex(frame_context.camera_model_sha256),
            bytes.fromhex(frame_context.camera_mount_sha256),
        ) + stream
        version = CONTEXT_PROTOCOL_VERSION
    header_bytes = HEADER_BYTES + len(extension)
    return HEADER.pack(
        MAGIC,
        version,
        header_bytes,
        action,
        sequence,
        captured,
        len(payload),
    ) + extension + payload


def _unpack_context(
    extension: bytes, *, sequence_id: int, capture_timestamp_ns: int
) -> FrameContextV1:
    if len(extension) < CONTEXT_FIXED_BYTES:
        raise EnvelopeError("SFD1 v2 context is shorter than its fixed header")
    (
        context_version,
        context_bytes,
        stream_bytes,
        frame_id,
        context_sequence,
        context_timestamp,
        x,
        y,
        z,
        pitch,
        yaw,
        roll,
        camera_hash,
        mount_hash,
    ) = CONTEXT_FIXED.unpack_from(extension)
    if context_version != FRAME_CONTEXT_VERSION:
        raise EnvelopeError(f"unsupported frame-context version {context_version}")
    if context_bytes != len(extension):
        raise EnvelopeError("frame-context declared length mismatch")
    if stream_bytes != len(extension) - CONTEXT_FIXED_BYTES:
        raise EnvelopeError("frame-context stream length mismatch")
    stream_raw = extension[CONTEXT_FIXED_BYTES:]
    try:
        stream_id = stream_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EnvelopeError("frame-context stream identity is invalid UTF-8") from exc
    context = FrameContextV1(
        stream_id=stream_id,
        frame_id=int(frame_id),
        sequence_id=int(context_sequence),
        capture_timestamp_ns=int(context_timestamp),
        ego_world=Pose6D(x, y, z, pitch, yaw, roll),
        camera_model_sha256=camera_hash.hex(),
        camera_mount_sha256=mount_hash.hex(),
    )
    try:
        validate_frame_context(context)
    except Exception as exc:
        raise EnvelopeError(str(exc)) from exc
    if context.sequence_id != sequence_id:
        raise EnvelopeError("frame-context sequence mismatch")
    if context.capture_timestamp_ns != capture_timestamp_ns:
        raise EnvelopeError("frame-context timestamp mismatch")
    return context


def unpack_envelope(frame: bytes | bytearray | memoryview) -> SplitEnvelope:
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise EnvelopeError("envelope input must be bytes-like")
    data = bytes(frame)
    if len(data) < HEADER_BYTES:
        raise EnvelopeError("frame is shorter than the outer envelope")
    magic, version, header_bytes, action, sequence, captured, payload_bytes = (
        HEADER.unpack_from(data)
    )
    if magic != MAGIC:
        raise EnvelopeError("outer envelope magic mismatch")
    if version == PROTOCOL_VERSION:
        if header_bytes != HEADER_BYTES:
            raise EnvelopeError("SFD1 v1 header length mismatch")
        context = None
    elif version == CONTEXT_PROTOCOL_VERSION:
        if not HEADER_BYTES + CONTEXT_FIXED_BYTES <= header_bytes <= len(data):
            raise EnvelopeError("protocol version 2 header length mismatch")
        context = _unpack_context(
            data[HEADER_BYTES:header_bytes],
            sequence_id=int(sequence),
            capture_timestamp_ns=int(captured),
        )
    else:
        raise EnvelopeError(f"unsupported outer protocol version {version}")
    if payload_bytes == 0:
        raise EnvelopeError("outer envelope declares an empty inner payload")
    if len(data) != header_bytes + payload_bytes:
        raise EnvelopeError("outer envelope inner length mismatch")
    return SplitEnvelope(
        protocol_version=int(version),
        action_id=int(action),
        sequence_id=int(sequence),
        capture_timestamp_ns=int(captured),
        inner_payload_length=int(payload_bytes),
        inner_payload=data[header_bytes:],
        header_bytes=int(header_bytes),
        frame_context=context,
    )
