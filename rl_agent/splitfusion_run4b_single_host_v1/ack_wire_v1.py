"""GT-free operational ACK wire format (edge -> UE), single-host Run-4B.

The edge emits one ACK per reward-requested (policy-decision) tensor as soon
as usable tail output exists.  The ACK carries the exact SFD4 decision/frame
identity, the tail-output digest and edge timing.  It carries no quality, no
Q_perc, no reward and no ground truth; it never waits for map installation.

Datagram: ``MAGIC (4) | body length (uint32 BE) | canonical JSON body |
SHA-256(body) (32)``.  Importing this module performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

MAGIC = b"R4BA"
SCHEMA = "scenesense.splitfusion.run4b_single_host.operational_ack.v1"
SEMANTICS = "TAIL_OUTPUT_READY__GT_FREE__BEFORE_MAP_PUBLICATION"
_HEADER = struct.Struct("!4sI")
_FORBIDDEN = ("q_perc", "quality", "ground_truth", "gt_")


class AckWireError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AckWireError(message)


@dataclass(frozen=True, slots=True)
class OperationalAckV1:
    session_uuid: str
    controller_lineage_sha256: str
    decision_seq: int
    ticket_seq: int
    frame_id: int
    tensor_seq: int
    capture_timestamp_ns: int
    mode_id: int
    q_e4: int
    keep_count: int
    execution_bundle_sha256: str
    anchor_action_id: Optional[int]
    reward_requested: bool
    tail_output_sha256: str
    record_count: int
    reassembly_complete_wall_s: float
    compute_start_wall_s: float
    tail_ready_wall_ns: int
    ack_send_wall_ns: int

    def __post_init__(self) -> None:
        _require(self.reward_requested is True,
                 "only a reward-requested decision tensor is acknowledged")
        for name in ("decision_seq", "ticket_seq", "frame_id", "tensor_seq",
                     "capture_timestamp_ns", "mode_id", "q_e4", "keep_count",
                     "record_count", "tail_ready_wall_ns", "ack_send_wall_ns"):
            value = getattr(self, name)
            _require(type(value) is int and value >= 0, f"{name} is invalid")
        for name in ("controller_lineage_sha256", "execution_bundle_sha256",
                     "tail_output_sha256"):
            value = getattr(self, name)
            _require(type(value) is str and len(value) == 64
                     and all(ch in "0123456789abcdef" for ch in value),
                     f"{name} is not a lowercase SHA-256")
        _require(self.anchor_action_id is None
                 or type(self.anchor_action_id) is int, "anchor is invalid")
        _require(self.ack_send_wall_ns >= self.tail_ready_wall_ns,
                 "ACK was stamped before tail output was ready")

    def body(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "semantics": SEMANTICS,
                **{item.name: getattr(self, item.name) for item in fields(self)}}

    @property
    def request_key(self) -> tuple[str, str, int, int]:
        return (self.session_uuid, self.controller_lineage_sha256,
                self.decision_seq, self.ticket_seq)


# The only reward-named field is the identity flag ``reward_requested``.
_NAMES = tuple(item.name for item in fields(OperationalAckV1))
if any(token in name for name in _NAMES for token in _FORBIDDEN) or any(
        "reward" in name for name in _NAMES if name != "reward_requested"):
    raise RuntimeError("operational ACK schema gained a quality/reward/GT field")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def encode(ack: OperationalAckV1) -> bytes:
    _require(type(ack) is OperationalAckV1, "ACK has a foreign type")
    body = _canonical(ack.body())
    return _HEADER.pack(MAGIC, len(body)) + body + hashlib.sha256(body).digest()


def is_ack(datagram: bytes) -> bool:
    return datagram[:4] == MAGIC


def decode(datagram: bytes) -> OperationalAckV1:
    _require(type(datagram) is bytes and len(datagram) >= _HEADER.size + 32,
             "ACK datagram is truncated")
    magic, length = _HEADER.unpack_from(datagram)
    _require(magic == MAGIC, "not an operational ACK")
    body = datagram[_HEADER.size:_HEADER.size + length]
    digest = datagram[_HEADER.size + length:]
    _require(len(body) == length and len(digest) == 32,
             "ACK datagram length differs from its header")
    _require(hashlib.sha256(body).digest() == digest, "ACK digest mismatch")
    raw = json.loads(body.decode("ascii"))
    _require(_canonical(raw) == body, "ACK body is not canonical")
    _require(raw.pop("schema") == SCHEMA and raw.pop("semantics") == SEMANTICS,
             "ACK schema/semantics drift")
    names = {item.name for item in fields(OperationalAckV1)}
    _require(set(raw) == names, "ACK fields are incomplete or foreign")
    return OperationalAckV1(**raw)


def tail_output_sha256(records: Any, labels: Any) -> str:
    """Digest of the usable tail output: canonical records + mask bytes."""
    import numpy as np

    digest = hashlib.sha256()
    digest.update(json.dumps(list(records), sort_keys=True,
                             separators=(",", ":"), default=float).encode())
    mask = np.ascontiguousarray(labels)
    digest.update(str(mask.dtype).encode() + str(tuple(mask.shape)).encode())
    digest.update(mask.tobytes())
    return digest.hexdigest()


__all__ = ["MAGIC", "SCHEMA", "SEMANTICS", "AckWireError", "OperationalAckV1",
           "encode", "decode", "is_ack", "tail_output_sha256"]
