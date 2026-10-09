"""Exact quality-grid scene catalogue for one registered split, carrying the three arms' reward values.

``split="fit"`` is required to equal ``FitSceneCatalogWithUndefinedV1`` exactly (keys, binding, every draw); it
serves training. ``split="held_scene"`` applies the identical construction to the registered held-out scenes (same
pinned grid, same corrected-P40 sidecar, same descriptor exclusion, undefined cells kept) and serves evaluation only.
Per grid cell it also stores the evaluation components (eligible, within-tolerance hits, gate TP/FP, segmentation
intersection/union, matched errors) so common metrics can be computed for any policy; cell (mode 0, q 0) =
noAE/UINT8 with no spatial drop is the per-scene uncompressed-equivalent reference (shown identical to the
full-local branch on all 99 retained live pairs, ``splitfusion_run5_quality_reference_v1``).

``reward_value`` serves the catalogue's own ``arm`` (the training reward); ``value_for`` any arm (evaluation).
"""
from __future__ import annotations

import bisect
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from rl_agent.splitfusion_corrected_latency_env_v1.scene_catalog import SCHEMA as FIT_SCHEMA
from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

from . import rewards as R

SPLITS = ("fit", "held_scene")
REFERENCE_CELL = (0, 0)


def components(row: dict) -> dict:
    out = {}
    for cls, tol in R.TOL.items():
        errors = [float(e) for e in row[f"loc_{cls}_matched_xy_errors_m"]]
        out[cls] = dict(eligible=int(row[f"loc_{cls}_eligible_gt"]), hits=sum(1 for e in errors if e <= tol),
                        tp=int(row[f"loc_{cls}_tp"]), fp=int(row[f"loc_{cls}_fp"]),
                        inter=int(row[f"seg_{cls}_intersection_pixels"]), union=int(row[f"seg_{cls}_union_pixels"]),
                        errors=errors)
    return out


class TrialSceneCatalogV1:
    def __init__(self, split: str, arm: str, repo_root: Path = C2.ROOT) -> None:
        SS.require(split in SPLITS, f"split must be one of {SPLITS}")
        SS.require(arm in R.ARMS, f"arm must be one of {R.ARMS}")
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
                     for r in sidecar["records"] if r["grid_split"] == split}
        connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
        try:
            scenes: dict[str, dict[str, Any]] = {}
            for (blob,) in connection.execute("SELECT row_json FROM quality_rows WHERE grid_split = ?", (split,)):
                row = json.loads(blob)
                key = f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}"
                entry = scenes.setdefault(key, {"camera_si": row.get("camera_si"), "radar_p40": corrected.get(key),
                                                "grid": {}, "values": {a: {} for a in R.ARMS}, "components": {}})
                cell = (int(row["mode_id"]), int(row["q_e4"]))
                entry["grid"][cell] = (row.get("q_perc"), int(row["total_transmitted_bytes"]),
                                       int(row["udp_application_bytes"]), str(row["row_sha256"]))
                for a in R.ARMS:
                    entry["values"][a][cell] = R.value_from_row(row, a)
                entry["components"][cell] = components(row)
        finally:
            connection.close()
        complete = {k: v for k, v in scenes.items()
                    if v["camera_si"] is not None and v["radar_p40"] is not None
                    and len(v["grid"]) == SS.MODE_COUNT * len(SS.Q_E4_GRID)}
        SS.require(bool(complete), f"no {split} scenes were loaded")
        for key, entry in complete.items():
            for cell, (q, *_rest) in entry["grid"].items():
                for a in R.ARMS:
                    SS.require((entry["values"][a][cell] is None) == (q is None),
                               f"{key} {cell}: {a} and Q_perc definedness differ")
        self.split = split
        self.arm = arm
        self._scenes = complete
        self._keys = tuple(sorted(complete))
        self.defined_keys = tuple(k for k in self._keys
                                  if all(c[0] is not None for c in complete[k]["grid"].values()))
        self.partially_undefined_keys = tuple(k for k in self._keys if k not in set(self.defined_keys))
        self.descriptor_excluded = len(scenes) - len(complete)
        # Same binding formula as FitSceneCatalogWithUndefinedV1, with the split named (identical for "fit").
        self._binding = C2.canonical_sha256({
            "schema": FIT_SCHEMA, "split": split, "scene_count": len(self._keys),
            "keys_sha256": C2.canonical_sha256(list(self._keys)),
            "defined_keys_sha256": C2.canonical_sha256(list(self.defined_keys)),
            "p40_sidecar_sha256": SS.P40_SIDECAR_SHA256})
        self.reward_sha256 = R.reward_sha256(arm)
        self.reward_binding_sha256 = C2.canonical_sha256({"catalog": self._binding, "reward": self.reward_sha256})

    # --------------------------------------------------- FitSceneCatalogWithUndefinedV1 interface (unchanged rules)
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

    @staticmethod
    def _bracket(q_e4: int) -> tuple[tuple[int, float], ...]:
        SS.require(0 <= q_e4 <= 9800, f"q_e4 {q_e4} outside the registered range")
        if q_e4 in SS.Q_E4_GRID:
            return ((q_e4, 1.0),)
        index = bisect.bisect_left(SS.Q_E4_GRID, q_e4)
        low_q, high_q = SS.Q_E4_GRID[index - 1], SS.Q_E4_GRID[index]
        weight = (q_e4 - low_q) / (high_q - low_q)
        return ((low_q, 1.0 - weight), (high_q, weight))

    def draw(self, scene_key: str, *, mode_id: int, q_e4: int) -> tuple[Optional[float], int, int, bool, str, str]:
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

    # --------------------------------------------------- reward values (scalar linear rule between grid anchors)
    def value_for(self, arm: str, scene_key: str, *, mode_id: int, q_e4: int) -> Optional[float]:
        grid = self._scenes[scene_key]["values"][arm]
        if q_e4 in SS.Q_E4_GRID:
            v = grid[(mode_id, q_e4)]
            return None if v is None else float(v)
        index = bisect.bisect_left(SS.Q_E4_GRID, q_e4)
        lo, hi = SS.Q_E4_GRID[index - 1], SS.Q_E4_GRID[index]
        a, b = grid[(mode_id, lo)], grid[(mode_id, hi)]
        if a is None or b is None:
            return None
        w = (q_e4 - lo) / (hi - lo)
        return float(a) + w * (float(b) - float(a))

    def reward_value(self, scene_key: str, *, mode_id: int, q_e4: int) -> Optional[float]:
        return self.value_for(self.arm, scene_key, mode_id=mode_id, q_e4=q_e4)

    # --------------------------------------------------- evaluation only
    def components(self, scene_key: str, *, mode_id: int, q_e4: int) -> list[tuple[float, dict]]:
        """Grid-row components with their linear weights (expected counts off-grid)."""
        cells = self._scenes[scene_key]["components"]
        return [(w, cells[(mode_id, q)]) for q, w in self._bracket(q_e4) if w > 0]

    def reference_components(self, scene_key: str) -> dict:
        return self._scenes[scene_key]["components"][REFERENCE_CELL]
