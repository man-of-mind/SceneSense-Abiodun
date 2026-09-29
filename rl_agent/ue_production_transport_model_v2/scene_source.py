#!/usr/bin/env python3
"""FIT-split scene / quality / payload source for the modeled collector.

Reads the registered offline quality grid and exposes, per scene:

* ``camera_si`` and ``radar_p40`` - the two causal scene descriptors;
* for any ``(mode_id, q_e4)``, the ``q_perc`` and ``total_transmitted_bytes``.

Only the ``fit`` grid split is loaded, so no held scene can reach training.

Off-anchor continuous ``q`` is interpolated **monotonically** between the two
neighbouring registered q anchors of the same scene and mode.  That transfer
is explicitly exploratory and is labelled as such on every draw; it carries no
reward penalty and makes no accuracy claim.
"""

from __future__ import annotations

import bisect
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_production_queue_capture_v1 import contract as V1

from . import contract_v2 as C2


SCENE_SOURCE_SCHEMA = "scenesense.modeled_collector_scene_source.v1"

# `radar_p40` is null for 95.9% of grid rows (`InvalidRadarRangesError`): the
# offline selector computed the Route-B range mask but omitted applying it
# before the strict descriptor call.  The registered read-only sidecar repairs
# exactly that, for all 768 scenes, with `no_zero_imputation`.  39 originally
# valid values are exactly re-derived by the same code path, which is the
# sidecar's own self-check.  Using it is what keeps radar P40 a genuine
# varying observation instead of a fallback constant.
P40_SIDECAR_RELPATH = (
    "experiments/splitfusion_hybrid_sac_p40_sidecar_v1/"
    "20260921_exact_grid_corrected_p40_v1/corrected_p40_sidecar.json"
)
P40_SIDECAR_SHA256 = (
    "b8b7e4b337e3feebc751f2627f0671b07f6a1b34f75447b36518bd90f542e27f"
)
FIT_SPLIT = "fit"
Q_E4_GRID = (0, 1500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9400, 9800)
MODE_COUNT = 12

CONTINUOUS_Q_TRANSFER_STATUS = "CONTINUOUS_Q_INTERPOLATION_EXPLORATORY"
MODE_TRANSFER_STATUS = "MODE_TRANSFER_EXPLORATORY"


class SceneSourceError(RuntimeError):
    """A scene-source invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SceneSourceError(message)


@dataclass(frozen=True, slots=True)
class SceneDraw:
    scene_key: str
    camera_si: float
    radar_p40: float
    mode_id: int
    q_e4: int
    q_perc: float
    total_transmitted_bytes: int
    wire_bytes: int
    is_registered_anchor: bool
    transfer_status: str
    row_sha256_low: str
    row_sha256_high: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_key": self.scene_key, "camera_si": self.camera_si,
            "radar_p40": self.radar_p40, "mode_id": self.mode_id,
            "q_e4": self.q_e4, "q_perc": self.q_perc,
            "total_transmitted_bytes": self.total_transmitted_bytes,
            "wire_bytes": self.wire_bytes,
            "is_registered_anchor": self.is_registered_anchor,
            "transfer_status": self.transfer_status,
            "row_sha256_low": self.row_sha256_low,
            "row_sha256_high": self.row_sha256_high,
        }


class FitSceneCatalog:
    """Immutable in-memory view of the registered FIT quality grid."""

    def __init__(self, repo_root: Path = C2.ROOT) -> None:
        database = repo_root / V1.PAYLOAD_AUTHORITY_RELPATH
        require(database.is_file(), f"payload authority missing: {database}")
        manifest = repo_root / V1.PAYLOAD_AUTHORITY_MANIFEST_RELPATH
        require(C2.sha256_file(manifest)
                == V1.PAYLOAD_AUTHORITY_MANIFEST_SHA256,
                "payload authority manifest drifted")
        sidecar_path = repo_root / P40_SIDECAR_RELPATH
        require(sidecar_path.is_file(),
                f"corrected P40 sidecar missing: {sidecar_path}")
        require(C2.sha256_file(sidecar_path) == P40_SIDECAR_SHA256,
                "corrected P40 sidecar drifted")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        require(sidecar["status"] == "COMPLETE"
                and sidecar["no_zero_imputation"] is True,
                "corrected P40 sidecar is incomplete or imputes zeros")
        corrected: dict[str, float] = {}
        for record in sidecar["records"]:
            if record["grid_split"] != FIT_SPLIT:
                continue
            value = record["corrected_p40"]
            require(value is not None, "sidecar record lacks a corrected P40")
            corrected[f"{record['episode_id']}|{record['sample_id']}"
                      f"|{record['frame_id']}"] = float(value)
        require(bool(corrected), "sidecar has no FIT records")
        self._p40_sidecar_sha256 = C2.sha256_file(sidecar_path)

        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            cursor = connection.execute(
                "SELECT row_json FROM quality_rows WHERE grid_split = ?",
                (FIT_SPLIT,))
            scenes: dict[str, dict[str, Any]] = {}
            for (blob,) in cursor:
                row = json.loads(blob)
                key = (f"{row['episode_id']}|{row['sample_id']}"
                       f"|{row['frame_id']}")
                entry = scenes.setdefault(key, {
                    "camera_si": row.get("camera_si"),
                    "radar_p40": corrected.get(key),
                    "grid": {},
                })
                entry["grid"][(int(row["mode_id"]), int(row["q_e4"]))] = (
                    row.get("q_perc"),
                    int(row["total_transmitted_bytes"]),
                    int(row["udp_application_bytes"]),
                    str(row["row_sha256"]),
                )
        finally:
            connection.close()

        usable = {
            key: value for key, value in scenes.items()
            if value["camera_si"] is not None and value["radar_p40"] is not None
            and len(value["grid"]) == MODE_COUNT * len(Q_E4_GRID)
            and all(item[0] is not None for item in value["grid"].values())
        }
        require(bool(usable), "no complete FIT scenes were loaded")
        self._scenes = usable
        self._keys = tuple(sorted(usable))
        self._binding = C2.canonical_sha256({
            "schema": SCENE_SOURCE_SCHEMA, "split": FIT_SPLIT,
            "scene_count": len(self._keys),
            "keys_sha256": C2.canonical_sha256(list(self._keys)),
            "p40_sidecar_sha256": self._p40_sidecar_sha256,
        })

    @property
    def scene_count(self) -> int:
        return len(self._keys)

    @property
    def binding_sha256(self) -> str:
        return self._binding

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def scene_descriptors(self, scene_key: str) -> tuple[float, float]:
        entry = self._scenes[scene_key]
        return float(entry["camera_si"]), float(entry["radar_p40"])

    def draw(self, scene_key: str, *, mode_id: int, q_e4: int) -> SceneDraw:
        require(0 <= mode_id < MODE_COUNT, f"invalid mode_id {mode_id}")
        require(0 <= q_e4 <= 9800, f"q_e4 {q_e4} outside the registered range")
        entry = self._scenes[scene_key]
        grid = entry["grid"]
        if q_e4 in Q_E4_GRID:
            q_perc, total, wire, digest = grid[(mode_id, q_e4)]
            return SceneDraw(
                scene_key=scene_key, camera_si=float(entry["camera_si"]),
                radar_p40=float(entry["radar_p40"]), mode_id=mode_id,
                q_e4=q_e4, q_perc=float(q_perc),
                total_transmitted_bytes=total, wire_bytes=wire,
                is_registered_anchor=True,
                transfer_status="REGISTERED_Q_ANCHOR",
                row_sha256_low=digest, row_sha256_high=digest)
        index = bisect.bisect_left(Q_E4_GRID, q_e4)
        low_q, high_q = Q_E4_GRID[index - 1], Q_E4_GRID[index]
        low = grid[(mode_id, low_q)]
        high = grid[(mode_id, high_q)]
        weight = (q_e4 - low_q) / (high_q - low_q)
        q_perc = float(low[0]) + weight * (float(high[0]) - float(low[0]))
        total = int(round(low[1] + weight * (high[1] - low[1])))
        wire = V1.udp_application_bytes(total)
        return SceneDraw(
            scene_key=scene_key, camera_si=float(entry["camera_si"]),
            radar_p40=float(entry["radar_p40"]), mode_id=mode_id, q_e4=q_e4,
            q_perc=q_perc, total_transmitted_bytes=total, wire_bytes=wire,
            is_registered_anchor=False,
            transfer_status=CONTINUOUS_Q_TRANSFER_STATUS,
            row_sha256_low=str(low[3]), row_sha256_high=str(high[3]))
