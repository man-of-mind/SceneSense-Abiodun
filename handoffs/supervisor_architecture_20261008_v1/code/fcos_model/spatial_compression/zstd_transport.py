"""Lossless zstd entropy coding of the frozen hybrid-q sparse wire payload.

Transport only. This wrapper compresses `SparsePayload.data` — the already
framed 44-byte-header + bitmask + retained-value byte string — and nothing
else. It never sees the dense C2 tensor, never zstd-compresses a
zero-scattered dense tensor (which would spend its whole gain re-encoding
dropped zeros that the sparse wire already removed), and changes no model
output. Compression is lossless, so decoded perception is bit-identical to the
uncompressed sparse path and no accuracy claim changes.

The compressor settings are frozen constants, not tunables: level 1, no worker
threads, no dictionary, frame checksum on, content size on, dictionary ID off.
There is no level search here, and adding one would make the reported sizes and
latencies incomparable across phases.

One camera frame produces exactly one independent zstd frame. Frames are never
concatenated and no batch-compression API is used, because a real UE emits each
frame's payload on its own and cannot condition frame N on frame N+1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import zstandard

from . import guards
from .codec import SparsePayload


# ---------------------------------------------------------------------------
# Frozen compressor configuration
# ---------------------------------------------------------------------------

ZSTD_LEVEL = 1
# threads=0 means "no worker threads": compression runs inline on the calling
# thread, so the measured latency is one core's cost and not a thread pool's.
ZSTD_THREADS = 0
ZSTD_DICT_DATA = None
ZSTD_WRITE_CHECKSUM = True
ZSTD_WRITE_CONTENT_SIZE = True
ZSTD_WRITE_DICT_ID = False

FROZEN_ZSTD_SETTINGS: dict[str, Any] = {
    "level": ZSTD_LEVEL,
    "threads": ZSTD_THREADS,
    "dict_data": None,
    "write_checksum": ZSTD_WRITE_CHECKSUM,
    "write_content_size": ZSTD_WRITE_CONTENT_SIZE,
    "write_dict_id": ZSTD_WRITE_DICT_ID,
    "one_frame_per_camera_frame": True,
    "level_search_performed": False,
}


class HybridQCompressionError(guards.HybridQPayloadError):
    """A zstd round trip did not return the exact input bytes."""


def implementation_report() -> dict[str, Any]:
    """Exact binding of the compressor actually used, for the record."""
    return {
        "implementation": "python-zstandard",
        "module": zstandard.__name__,
        "binding_version": str(zstandard.__version__),
        "zstd_library_version": ".".join(str(part) for part in zstandard.ZSTD_VERSION),
        "backend": str(zstandard.backend),
        "settings": dict(FROZEN_ZSTD_SETTINGS),
    }


@dataclass(frozen=True)
class CompressedFrame:
    """One zstd frame over one sparse payload, with its measured lengths."""

    data: bytes
    uncompressed_bytes: int

    @property
    def compressed_bytes(self) -> int:
        return len(self.data)

    @property
    def zstd_ratio(self) -> float:
        """compressed_zstd_bytes / sparse_payload_bytes."""
        return self.compressed_bytes / self.uncompressed_bytes


def _payload_bytes(payload: bytes | bytearray | memoryview | SparsePayload) -> bytes:
    """Extract exactly the framed sparse wire bytes to be compressed."""
    if isinstance(payload, SparsePayload):
        return payload.data
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    raise guards.HybridQPayloadError(
        "zstd transport accepts a SparsePayload or a bytes-like wire payload, "
        f"got {type(payload).__name__}"
    )


class ZstdWireCodec:
    """One reused compressor and one reused decompressor context.

    The contexts are constructed once and used sequentially, which is what a UE
    would do: paying context setup per frame would inflate the measured latency
    with work production never repeats.
    """

    def __init__(self) -> None:
        self._compressor = zstandard.ZstdCompressor(
            level=ZSTD_LEVEL,
            threads=ZSTD_THREADS,
            dict_data=ZSTD_DICT_DATA,
            write_checksum=ZSTD_WRITE_CHECKSUM,
            write_content_size=ZSTD_WRITE_CONTENT_SIZE,
            write_dict_id=ZSTD_WRITE_DICT_ID,
        )
        self._decompressor = zstandard.ZstdDecompressor()

    def compress(
        self, payload: bytes | bytearray | memoryview | SparsePayload
    ) -> CompressedFrame:
        """Compress one framed sparse payload into one independent zstd frame."""
        data = _payload_bytes(payload)
        if not data:
            raise guards.HybridQPayloadError("refusing to compress an empty payload")
        return CompressedFrame(
            data=self._compressor.compress(data), uncompressed_bytes=len(data)
        )

    def compress_bytes(self, data: bytes) -> bytes:
        """Timing-path entry point: bytes in, one zstd frame out, no wrapping."""
        return self._compressor.compress(data)

    def decompress(self, frame: bytes, *, expected_bytes: int | None = None) -> bytes:
        """Decompress one zstd frame back to the framed sparse payload.

        The frame carries its content size, so this is a single-shot decode with
        no size guessing. `expected_bytes` is an optional structural cross-check.
        """
        data = self._decompressor.decompress(frame)
        if expected_bytes is not None and len(data) != int(expected_bytes):
            raise HybridQCompressionError(
                f"decompressed {len(data)} bytes, expected {int(expected_bytes)}"
            )
        return data

    def decompress_bytes(self, frame: bytes) -> bytes:
        """Timing-path entry point: one zstd frame in, payload bytes out."""
        return self._decompressor.decompress(frame)

    def round_trip_is_exact(
        self, payload: bytes | bytearray | memoryview | SparsePayload, frame: bytes
    ) -> bool:
        """Byte-for-byte equality of the decompressed frame with the payload."""
        return self.decompress_bytes(frame) == _payload_bytes(payload)


def frame_content_size(frame: bytes) -> int:
    """Declared content size in the zstd frame header.

    Meaningful only because `write_content_size=True` is frozen on; it lets the
    edge allocate the exact payload buffer before decoding.
    """
    return int(zstandard.frame_content_size(frame))
