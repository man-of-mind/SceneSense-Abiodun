"""Recipient-specific Phase-2 cooperative spatial-map core."""

from .adapters import snapshot_stream_to_contribution
from .engine import RecipientMapEngine
from .evaluation import TruthTrajectory, WarningTruthMatch, match_warning_to_truth
from .selection import select_recipient_hazards
from .transport import ChunkReassembler, ReassembledPayload, chunk_payload
from .schemas import (
    EgoState,
    MapContribution,
    MapObjectObservation,
    WarningEvent,
    with_exact_payload_bytes,
)

__all__ = [
    "EgoState",
    "ChunkReassembler",
    "MapContribution",
    "MapObjectObservation",
    "RecipientMapEngine",
    "ReassembledPayload",
    "TruthTrajectory",
    "WarningEvent",
    "WarningTruthMatch",
    "match_warning_to_truth",
    "snapshot_stream_to_contribution",
    "select_recipient_hazards",
    "chunk_payload",
    "with_exact_payload_bytes",
]
