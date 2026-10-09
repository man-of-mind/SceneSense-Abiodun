"""FIT quality-grid view that keeps undefined-quality cells, so timely undefined Q can occur as it does live.

Same registered sources and pins as ``ue_production_transport_model_v2.scene_source.FitSceneCatalog`` (payload
authority manifest, corrected-P40 sidecar). Quality values, anchors and the linear off-anchor rule are unchanged;
the only difference is that a scene with complete descriptors and a complete 12x11 grid is no longer dropped when
some cells have ``q_perc = None``. A draw whose bracketing anchors include an undefined cell is undefined.
Scenes lacking a camera-SI or radar-P40 descriptor stay excluded: live, such a state is an external fallback.
"""
from __future__ import annotations

import bisect
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

SCHEMA = "splitfusion.corrected_latency_env.fit_scene_catalog_with_undefined.v1"


class FitSceneCatalogWithUndefinedV1:
    def __init__(self, repo_root: Path = C2.ROOT) -> None:
        database = repo_root / V1.PAYLOAD_AUTHORITY_RELPATH
        SS.require(database.is_file(), f"payload authority missing: {database}")
        SS.require(C2.sha256_file(repo_root / V1.PAYLOAD_AUTHORITY_MANIFEST_RELPATH)
                   == V1.PAYLOAD_AUTHORITY_MANIFEST_SHA256, "payload authority manifest drifted")
        sidecar_path = repo_root / SS.P40_SIDECAR_RELPATH
        SS.require(C2.sha256_file(sidecar_path) == SS.P40_SIDECAR_SHA256, "corrected P40 sidecar drifted")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        SS.require(sidecar["status"] == "COMPLETE" and sidecar["no_zero_imputation"] is True,
                   "corrected P40 sidecar is incomplete or imputes zeros")
        corrected = {f"{r['episode_id']}|{r['sample_id']}|{r['frame_id']}": float(r["corrected_p40"])
                     for r in sidecar["records"] if r["grid_split"] == SS.FIT_SPLIT}
        connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
        try:
            scenes: dict[str, dict[str, Any]] = {}
            for (blob,) in connection.execute("SELECT row_json FROM quality_rows WHERE grid_split = ?",
                                              (SS.FIT_SPLIT,)):
                row = json.loads(blob)
                key = f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}"
                entry = scenes.setdefault(key, {"camera_si": row.get("camera_si"), "radar_p40": corrected.get(key),
                                                "grid": {}})
                entry["grid"][(int(row["mode_id"]), int(row["q_e4"]))] = (
                    row.get("q_perc"), int(row["total_transmitted_bytes"]), int(row["udp_application_bytes"]),
                    str(row["row_sha256"]))
        finally:
            connection.close()
        complete = {k: v for k, v in scenes.items()
                    if v["camera_si"] is not None and v["radar_p40"] is not None
                    and len(v["grid"]) == SS.MODE_COUNT * len(SS.Q_E4_GRID)}
        SS.require(bool(complete), "no FIT scenes were loaded")
        self._scenes = complete
        self._keys = tuple(sorted(complete))
        self.defined_keys = tuple(k for k in self._keys
                                  if all(c[0] is not None for c in complete[k]["grid"].values()))
        self.partially_undefined_keys = tuple(k for k in self._keys if k not in set(self.defined_keys))
        self.descriptor_excluded = len(scenes) - len(complete)
        self._binding = C2.canonical_sha256({
            "schema": SCHEMA, "split": SS.FIT_SPLIT, "scene_count": len(self._keys),
            "keys_sha256": C2.canonical_sha256(list(self._keys)),
            "defined_keys_sha256": C2.canonical_sha256(list(self.defined_keys)),
            "p40_sidecar_sha256": SS.P40_SIDECAR_SHA256})

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

    def undefined_cell_count(self, scene_key: str) -> int:
        return sum(1 for cell in self._scenes[scene_key]["grid"].values() if cell[0] is None)

    def draw(self, scene_key: str, *, mode_id: int, q_e4: int) -> tuple[Optional[float], int, int, bool, str, str]:
        """(q_perc or None, total_transmitted_bytes, wire_bytes, is_anchor, row_sha_low, row_sha_high).

        Payload never depends on quality definedness, so bytes are always available.
        """
        SS.require(0 <= mode_id < SS.MODE_COUNT, f"invalid mode_id {mode_id}")
        SS.require(0 <= q_e4 <= 9800, f"q_e4 {q_e4} outside the registered range")
        grid = self._scenes[scene_key]["grid"]
        if q_e4 in SS.Q_E4_GRID:
            q, total, wire, digest = grid[(mode_id, q_e4)]
            return (None if q is None else float(q)), total, wire, True, digest, digest
        index = bisect.bisect_left(SS.Q_E4_GRID, q_e4)
        low_q, high_q = SS.Q_E4_GRID[index - 1], SS.Q_E4_GRID[index]
        low, high = grid[(mode_id, low_q)], grid[(mode_id, high_q)]
        weight = (q_e4 - low_q) / (high_q - low_q)
        q = None if low[0] is None or high[0] is None else float(low[0]) + weight * (float(high[0]) - float(low[0]))
        total = int(round(low[1] + weight * (high[1] - low[1])))
        return q, total, V1.udp_application_bytes(total), False, str(low[3]), str(high[3])
