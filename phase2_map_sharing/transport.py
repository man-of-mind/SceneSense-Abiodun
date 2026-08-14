"""Production-header-compatible chunking for Phase-2 contribution JSON."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# Matches rl_agent.multiue_oai.endpoint and the existing production split path.
CHUNK_HEADER = struct.Struct("!IHH")


def chunk_payload(payload: bytes, *, message_id: int, chunk_bytes: int = 60_000) -> Tuple[bytes, ...]:
    capacity = int(chunk_bytes) - CHUNK_HEADER.size
    if not payload or capacity <= 0:
        raise ValueError("payload must be nonempty and chunk_bytes must exceed the 8-byte header")
    total = (len(payload) + capacity - 1) // capacity
    if not 0 <= int(message_id) <= 0xFFFFFFFF or total > 0xFFFF:
        raise ValueError("message ID or chunk count exceeds production header range")
    return tuple(
        CHUNK_HEADER.pack(int(message_id), index, total)
        + payload[index * capacity : (index + 1) * capacity]
        for index in range(total)
    )


@dataclass(frozen=True)
class ReassembledPayload:
    source_endpoint: str
    message_id: int
    first_chunk_at_s: float
    last_chunk_at_s: float
    chunk_count: int
    duplicate_chunks: int
    payload: bytes


class ChunkReassembler:
    def __init__(self, *, timeout_s: float = 1.0, max_chunks: int = 4096) -> None:
        if timeout_s <= 0 or max_chunks <= 0:
            raise ValueError("timeout_s and max_chunks must be positive")
        self.timeout_s = float(timeout_s)
        self.max_chunks = int(max_chunks)
        self.pending: Dict[tuple[str, int], dict] = {}
        self.expired_messages = 0

    def expire(self, now_s: float) -> None:
        stale = [
            key
            for key, item in self.pending.items()
            if float(now_s) - float(item["first_chunk_at_s"]) > self.timeout_s
        ]
        for key in stale:
            del self.pending[key]
            self.expired_messages += 1

    def ingest(
        self,
        source_endpoint: str,
        datagram: bytes,
        *,
        received_at_s: float,
    ) -> Optional[ReassembledPayload]:
        self.expire(received_at_s)
        if len(datagram) < CHUNK_HEADER.size:
            raise ValueError("datagram is shorter than the production chunk header")
        message_id, index, total = CHUNK_HEADER.unpack_from(datagram)
        if total <= 0 or total > self.max_chunks or index >= total:
            raise ValueError("invalid chunk index/count")
        key = (str(source_endpoint), message_id)
        item = self.pending.setdefault(
            key,
            {
                "total": total,
                "first_chunk_at_s": float(received_at_s),
                "chunks": {},
                "duplicates": 0,
            },
        )
        if item["total"] != total:
            del self.pending[key]
            raise ValueError("chunk count changed within a message")
        chunks = item["chunks"]
        if index in chunks:
            item["duplicates"] += 1
        else:
            chunks[index] = datagram[CHUNK_HEADER.size :]
        if len(chunks) != total:
            return None
        result = ReassembledPayload(
            source_endpoint=str(source_endpoint),
            message_id=message_id,
            first_chunk_at_s=float(item["first_chunk_at_s"]),
            last_chunk_at_s=float(received_at_s),
            chunk_count=total,
            duplicate_chunks=int(item["duplicates"]),
            payload=b"".join(chunks[position] for position in range(total)),
        )
        del self.pending[key]
        return result
