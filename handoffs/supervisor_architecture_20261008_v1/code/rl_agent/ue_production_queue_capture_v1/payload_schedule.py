#!/usr/bin/env python3
"""Deterministic per-frame byte schedule drawn from the registered payload authority.

The live sender never invents a byte load.  Every frame replays the exact
``total_transmitted_bytes`` of one registered quality-grid row, so the measured
ingress is natural and nonconstant and every byte is traceable to an authority
row digest.

FIT cells draw from the ``fit`` scene split and VALIDATION cells from
``held_scene``, so the queue-model partition and the scene partition are both
disjoint.

Selection is by SHA-256 rank over a frozen seed.  No global RNG is used.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import contract as C


SCHEDULE_SCHEMA = "scenesense.production_queue_capture_payload_schedule.v1"


@dataclass(frozen=True, slots=True)
class ScheduledFrame:
    frame_index: int
    block_index: int
    tier: str
    action_id: int
    mode_id: int
    q_e4: int
    total_transmitted_bytes: int
    datagram_count: int
    udp_application_bytes: int
    row_sha256: str
    episode_id: str
    sample_id: str
    source_frame_id: int
    grid_split: str
    q_perc: float | None
    camera_si: float | None
    radar_p40: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index, "block_index": self.block_index,
            "tier": self.tier, "action_id": self.action_id,
            "mode_id": self.mode_id, "q_e4": self.q_e4,
            "total_transmitted_bytes": self.total_transmitted_bytes,
            "datagram_count": self.datagram_count,
            "udp_application_bytes": self.udp_application_bytes,
            "row_sha256": self.row_sha256, "episode_id": self.episode_id,
            "sample_id": self.sample_id, "source_frame_id": self.source_frame_id,
            "grid_split": self.grid_split, "q_perc": self.q_perc,
            "camera_si": self.camera_si, "radar_p40": self.radar_p40,
        }


def _rank_key(seed: int, cell_id: str, block_index: int, row_sha256: str) -> str:
    material = f"{seed}|{cell_id}|{block_index}|{row_sha256}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _open_authority(repo_root: Path) -> sqlite3.Connection:
    path = repo_root / C.PAYLOAD_AUTHORITY_RELPATH
    C.require(path.is_file(), f"payload authority missing: {path}")
    manifest = repo_root / C.PAYLOAD_AUTHORITY_MANIFEST_RELPATH
    C.require(manifest.is_file(), "payload authority manifest missing")
    C.require(
        C.sha256_file(manifest) == C.PAYLOAD_AUTHORITY_MANIFEST_SHA256,
        "payload authority manifest drifted",
    )
    # Read-only.  The authority database is immutable evidence.
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _candidate_rows(
    connection: sqlite3.Connection, *, mode_id: int, q_e4: int, grid_split: str,
) -> list[dict[str, Any]]:
    cursor = connection.execute(
        "SELECT row_json FROM quality_rows "
        "WHERE mode_id = ? AND q_e4 = ? AND grid_split = ?",
        (mode_id, q_e4, grid_split),
    )
    rows = [json.loads(value[0]) for value in cursor]
    C.require(bool(rows), f"no authority rows for mode {mode_id} q {q_e4} {grid_split}")
    return rows


def build_cell_schedule(
    cell: C.Cell, *, repo_root: Path = C.ROOT,
    connection: sqlite3.Connection | None = None,
) -> tuple[ScheduledFrame, ...]:
    """Return the frozen 450-frame byte schedule for one cell."""
    owned = connection is None
    connection = connection or _open_authority(repo_root)
    try:
        frames: list[ScheduledFrame] = []
        frame_index = 0
        for block in cell.blocks:
            spec = C.tier_by_name(block.tier)
            candidates = _candidate_rows(
                connection, mode_id=spec.mode_id, q_e4=spec.q_e4,
                grid_split=cell.scene_split,
            )
            C.require(
                len(candidates) >= block.frames,
                f"authority has {len(candidates)} rows but the block needs "
                f"{block.frames}",
            )
            ordered = sorted(
                candidates,
                key=lambda row: _rank_key(
                    C.PAYLOAD_SCHEDULE_SEED, cell.cell_id, block.block_index,
                    row["row_sha256"],
                ),
            )
            for row in ordered[: block.frames]:
                total = int(row["total_transmitted_bytes"])
                C.require(total > 0, "authority payload must be positive")
                # The authority already carries the deployed accounting; recompute
                # it independently and refuse any disagreement.
                C.require(
                    C.datagram_count(total) == int(row["datagram_count"]),
                    "datagram accounting disagrees with the authority row",
                )
                C.require(
                    C.udp_application_bytes(total)
                    == int(row["udp_application_bytes"]),
                    "udp application accounting disagrees with the authority row",
                )
                C.require(
                    int(row["udp_chunk_bytes_including_header"])
                    == C.UDP_CHUNK_BYTES_INCLUDING_HEADER
                    and int(row["udp_chunk_header_bytes_per_datagram"])
                    == C.UDP_CHUNK_HEADER_BYTES,
                    "authority row is not in the production packetization domain",
                )
                frames.append(ScheduledFrame(
                    frame_index=frame_index, block_index=block.block_index,
                    tier=block.tier, action_id=block.action_id,
                    mode_id=block.mode_id, q_e4=block.q_e4,
                    total_transmitted_bytes=total,
                    datagram_count=int(row["datagram_count"]),
                    udp_application_bytes=int(row["udp_application_bytes"]),
                    row_sha256=str(row["row_sha256"]),
                    episode_id=str(row["episode_id"]),
                    sample_id=str(row["sample_id"]),
                    source_frame_id=int(row["frame_id"]),
                    grid_split=str(row["grid_split"]),
                    q_perc=row.get("q_perc"),
                    camera_si=row.get("camera_si"),
                    radar_p40=row.get("radar_p40"),
                ))
                frame_index += 1
        C.require(
            len(frames) == C.FRAMES_PER_CELL,
            f"cell schedule has {len(frames)} frames, expected {C.FRAMES_PER_CELL}",
        )
        return tuple(frames)
    finally:
        if owned:
            connection.close()


def schedule_document(
    cell: C.Cell, frames: Sequence[ScheduledFrame],
) -> dict[str, Any]:
    payloads = [frame.total_transmitted_bytes for frame in frames]
    return {
        "schema": SCHEDULE_SCHEMA,
        "cell_id": cell.cell_id, "partition": cell.partition,
        "profile_id": cell.profile_id, "scene_split": cell.scene_split,
        "permutation_index": cell.permutation_index,
        "payload_coordinate": C.PAYLOAD_COORDINATE,
        "payload_schedule_seed": C.PAYLOAD_SCHEDULE_SEED,
        "frames": [frame.to_json() for frame in frames],
        "summary": {
            "frames": len(frames),
            "distinct_payload_bytes": len(set(payloads)),
            "min_total_transmitted_bytes": min(payloads),
            "max_total_transmitted_bytes": max(payloads),
            "sum_total_transmitted_bytes": sum(payloads),
            "sum_udp_application_bytes":
                sum(frame.udp_application_bytes for frame in frames),
            "sum_datagrams": sum(frame.datagram_count for frame in frames),
        },
        "schedule_sha256": C.canonical_sha256(
            [frame.to_json() for frame in frames]
        ),
    }


def build_campaign_schedule(
    repo_root: Path = C.ROOT,
) -> dict[str, Any]:
    connection = _open_authority(repo_root)
    try:
        documents = []
        for cell in C.planned_cells():
            frames = build_cell_schedule(
                cell, repo_root=repo_root, connection=connection)
            documents.append(schedule_document(cell, frames))
    finally:
        connection.close()
    return {
        "schema": SCHEDULE_SCHEMA + ".campaign",
        "contract_sha256": C.CONTRACT_SHA256,
        "cells": documents,
        "campaign_sha256": C.canonical_sha256(documents),
    }
