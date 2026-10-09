"""Pilot scene catalog: exact grid rows for the supported scenes, Q_perc (state/control) and the candidate value.

Rows come from the pinned exact quality grid (``quality_rows.sqlite3``, same file and corrected P40 sidecar as the
corrected candidate's FIT catalog). Split ``fit`` serves training, ``held_scene`` serves evaluation. A scene is
supported if its camera SI and corrected radar P40 lie inside the measured SI/P40 range of every measured latency
stratum of the matching chain split (TRAIN for fit, EVAL for held) and its grid is complete for modes 5, 8 and 11.

Candidate value per grid row (definition authorised for this pilot only):
    Y      = class-weighted within-tolerance recall over classes with eligible GT
             (person 0.6 / vehicle 0.4 = the registered Q_perc class weights; vehicle 1.0 m, person 1.2 m)
    value  = Y * (0.7 + 0.3 * Q_seg)        Q_seg = the row's registered normalized segmentation term
Y (hence the value) is undefined exactly when Q_perc is undefined (no eligible GT in either class). Off-grid q uses
the registered linear rule between the bracketing grid rows, undefined if either row is undefined -- identical to
how Q_perc is interpolated (``FitSceneCatalogWithUndefinedV1.draw``).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Optional

from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

from . import support as S

ROOT = Path(__file__).resolve().parents[2]
TOL = {"vehicle": 1.0, "person": 1.2}
WEIGHT = {"person": 0.6, "vehicle": 0.4}
SEG_BETA = 0.3
GRID = SS.Q_E4_GRID


class QualityError(ValueError):
    pass


def candidate_from_row(row: dict) -> Optional[float]:
    num = den = 0.0
    for cls in TOL:
        eligible = int(row[f"loc_{cls}_eligible_gt"])
        if eligible:
            hits = sum(1 for e in row[f"loc_{cls}_matched_xy_errors_m"] if float(e) <= TOL[cls])
            num += WEIGHT[cls] * hits / eligible
            den += WEIGHT[cls]
    if den == 0:
        if row["q_perc"] is not None:
            raise QualityError("Y undefined but Q_perc defined")
        return None
    if row["q_perc"] is None:
        raise QualityError("Y defined but Q_perc undefined")
    q_seg = row["q_seg"]
    modulation = 1.0 if q_seg is None else (1.0 - SEG_BETA) + SEG_BETA * float(q_seg)
    return (num / den) * modulation


def _components(row: dict) -> dict:
    out = {}
    for cls in TOL:
        errors = [float(e) for e in row[f"loc_{cls}_matched_xy_errors_m"]]
        out[cls] = dict(eligible=int(row[f"loc_{cls}_eligible_gt"]), tp=int(row[f"loc_{cls}_tp"]), fp=int(row[f"loc_{cls}_fp"]),
                        hits=sum(1 for e in errors if e <= TOL[cls]), errors=errors,
                        inter=int(row[f"seg_{cls}_intersection_pixels"]), union=int(row[f"seg_{cls}_union_pixels"]))
    return out


class PilotSceneCatalogV1:
    def __init__(self, split: str, supported: Callable[[float, float], bool]) -> None:
        if split not in ("fit", "held_scene"):
            raise QualityError("split must be fit or held_scene")
        database = ROOT / V1.PAYLOAD_AUTHORITY_RELPATH
        sidecar = json.loads((ROOT / SS.P40_SIDECAR_RELPATH).read_text())
        if hashlib.sha256((ROOT / SS.P40_SIDECAR_RELPATH).read_bytes()).hexdigest() != SS.P40_SIDECAR_SHA256:
            raise QualityError("P40 sidecar drifted")
        p40 = {f"{r['episode_id']}|{r['sample_id']}|{r['frame_id']}": float(r["corrected_p40"])
               for r in sidecar["records"] if r["grid_split"] == split}
        con = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
        scenes: dict[str, dict[str, Any]] = {}
        try:
            for (blob,) in con.execute("SELECT row_json FROM quality_rows WHERE grid_split = ? AND mode_id IN (5, 8, 11)",
                                       (split,)):
                row = json.loads(blob)
                key = f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}"
                entry = scenes.setdefault(key, {"camera_si": row.get("camera_si"), "radar_p40": p40.get(key), "grid": {}})
                entry["grid"][(int(row["mode_id"]), int(row["q_e4"]))] = dict(
                    q_perc=row["q_perc"], total=int(row["total_transmitted_bytes"]), sha=str(row["row_sha256"]),
                    candidate=candidate_from_row(row), components=_components(row))
        finally:
            con.close()
        complete = {k: v for k, v in scenes.items() if v["camera_si"] is not None and v["radar_p40"] is not None
                    and len(v["grid"]) == 3 * len(GRID)}
        self._scenes = {k: v for k, v in complete.items() if supported(float(v["camera_si"]), float(v["radar_p40"]))}
        if not self._scenes:
            raise QualityError("no supported scenes")
        self._keys = tuple(sorted(self._scenes))
        self.split = split
        self.complete_count = len(complete)
        self.defined_keys = tuple(k for k in self._keys if self._scenes[k]["grid"][(11, 3000)]["q_perc"] is not None)
        self.binding_sha256 = hashlib.sha256(json.dumps({"split": split, "keys": list(self._keys),
                                                         "support": S.SUPPORT_SHA256}).encode()).hexdigest()

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    @property
    def scene_count(self) -> int:
        return len(self._keys)

    def scene_descriptors(self, key: str) -> tuple[float, float]:
        e = self._scenes[key]
        return float(e["camera_si"]), float(e["radar_p40"])

    def _bracket(self, mode_id: int, q_e4: int):
        if not 0 <= q_e4 <= 9800:
            raise QualityError(f"q_e4 {q_e4} outside the registered range")
        if q_e4 in GRID:
            return ((q_e4, 1.0),)
        i = bisect.bisect_left(GRID, q_e4)
        lo, hi = GRID[i - 1], GRID[i]
        w = (q_e4 - lo) / (hi - lo)
        return ((lo, 1.0 - w), (hi, w))

    def draw(self, key: str, *, mode_id: int, q_e4: int):
        """Same contract as ``FitSceneCatalogWithUndefinedV1.draw``: (q_perc|None, total, wire, is_anchor, sha_lo, sha_hi)."""
        grid = self._scenes[key]["grid"]
        parts = self._bracket(mode_id, q_e4)
        if len(parts) == 1:
            cell = grid[(mode_id, q_e4)]
            q = cell["q_perc"]
            return (None if q is None else float(q)), cell["total"], V1.udp_application_bytes(cell["total"]), True, cell["sha"], cell["sha"]
        (lo, _), (hi, w) = parts
        a, b = grid[(mode_id, lo)], grid[(mode_id, hi)]
        q = None if a["q_perc"] is None or b["q_perc"] is None else float(a["q_perc"]) + w * (float(b["q_perc"]) - float(a["q_perc"]))
        total = int(round(a["total"] + w * (b["total"] - a["total"])))
        return q, total, V1.udp_application_bytes(total), False, a["sha"], b["sha"]

    def candidate(self, key: str, *, mode_id: int, q_e4: int) -> Optional[float]:
        grid = self._scenes[key]["grid"]
        parts = self._bracket(mode_id, q_e4)
        values = [grid[(mode_id, q)]["candidate"] for q, _ in parts]
        if any(v is None for v in values):
            return None
        if len(parts) == 1:
            return float(values[0])
        (lo, _), (hi, w) = parts
        return float(values[0]) + w * (float(values[1]) - float(values[0]))

    def components(self, key: str, *, mode_id: int, q_e4: int) -> list[tuple[float, dict]]:
        """Grid-row components with their linear weights (for evaluation metrics only)."""
        grid = self._scenes[key]["grid"]
        return [(weight, grid[(mode_id, q)]["components"]) for q, weight in self._bracket(mode_id, q_e4) if weight > 0]
