"""Minimal control envelope around one unchanged SplitFusion inner payload.

The existing UDP transport's ``!IHH`` header is a chunk/reassembly header, not
an action contract.  This envelope is intended to become that transport's
reassembled message body: it adds the minimum routing and capture identity
without changing or recompressing the scientific inner zstd frame.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


MAGIC = b"SFD1"
PROTOCOL_VERSION = 1
HEADER = struct.Struct("<4sHHIQQQ")
HEADER_BYTES = HEADER.size  # 36
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

    @property
    def control_overhead_bytes(self) -> int:
        return HEADER_BYTES

    @property
    def total_transmitted_bytes(self) -> int:
        return HEADER_BYTES + self.inner_payload_length


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
) -> bytes:
    payload = bytes(inner_payload)
    if not payload:
        raise EnvelopeError("inner payload must be non-empty")
    action = _unsigned(action_id, UINT32_MAX, "action_id")
    sequence = _unsigned(sequence_id, UINT64_MAX, "sequence_id")
    captured = _unsigned(capture_timestamp_ns, UINT64_MAX, "capture_timestamp_ns")
    return HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        HEADER_BYTES,
        action,
        sequence,
        captured,
        len(payload),
    ) + payload


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
    if version != PROTOCOL_VERSION:
        raise EnvelopeError(f"unsupported outer protocol version {version}")
    if header_bytes != HEADER_BYTES:
        raise EnvelopeError("outer envelope header length mismatch")
    if payload_bytes == 0:
        raise EnvelopeError("outer envelope declares an empty inner payload")
    if len(data) != HEADER_BYTES + payload_bytes:
        raise EnvelopeError("outer envelope inner length mismatch")
    return SplitEnvelope(
        protocol_version=int(version),
        action_id=int(action),
        sequence_id=int(sequence),
        capture_timestamp_ns=int(captured),
        inner_payload_length=int(payload_bytes),
        inner_payload=data[HEADER_BYTES:],
    )
