"""Full quality catalogue: the existing exact quality grid (all 12 modes x 11 q points, FIT split), with the agreed
localisation-aware reward value per grid cell.

Scenes, payloads, Q_perc, undefined cells and the linear off-anchor rule are exactly
``FitSceneCatalogWithUndefinedV1`` (corrected latency candidate; 512 FIT scenes, 36 with undefined cells kept so
timely-undefined faults occur as live). Added per grid cell, from the same row:

    Y      = class-weighted within-tolerance recall over classes with eligible GT
             (person 0.6 @ 1.2 m, vehicle 0.4 @ 1.0 m: the registered Q_perc class weights)
    value  = Y * (0.7 + 0.3 * Q_seg)        Q_seg = the row's registered normalised segmentation term

(the restricted pilot's CANDIDATE definition, ``splitfusion_run5_restricted_pilot_v1.quality.candidate_from_row``,
imported unchanged). The value is undefined exactly when Q_perc is undefined; off-grid q is linear between the
bracketing grid cells and undefined if either is.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import sqlite3
from typing import Optional

from rl_agent.splitfusion_corrected_latency_env_v1.scene_catalog import FitSceneCatalogWithUndefinedV1
from rl_agent.splitfusion_run5_restricted_pilot_v1.quality import SEG_BETA, TOL, WEIGHT, candidate_from_row
from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

REWARD_DEFINITION = dict(
    schema="splitfusion.run5_full.reward.v1",
    success="Y * (0.7 + 0.3 * Q_seg); Y = class-weighted within-tolerance recall (person 0.6 @ 1.2 m, "
            "vehicle 0.4 @ 1.0 m) over classes with eligible GT",
    timeout=0.0, timely_undefined="EXCLUDED_EVALUATOR_FAULT_SESSION_RESET", late_undefined="TIMEOUT",
    deadline_ms_inclusive=170, tolerances_m=dict(TOL), class_weights=dict(WEIGHT), seg_beta=SEG_BETA,
    state_previous_quality="Q_perc (unchanged Run-5 feature)")
REWARD_SHA256 = hashlib.sha256(json.dumps(REWARD_DEFINITION, sort_keys=True).encode()).hexdigest()


class FullSceneCatalogV1(FitSceneCatalogWithUndefinedV1):
    def __init__(self, repo_root=C2.ROOT) -> None:
        super().__init__(repo_root)
        keys = set(self.keys)
        values: dict[str, dict[tuple[int, int], Optional[float]]] = {k: {} for k in keys}
        connection = sqlite3.connect(f"file:{repo_root / V1.PAYLOAD_AUTHORITY_RELPATH}?mode=ro&immutable=1", uri=True)
        try:
            for (blob,) in connection.execute("SELECT row_json FROM quality_rows WHERE grid_split = ?", (SS.FIT_SPLIT,)):
                row = json.loads(blob)
                key = f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}"
                if key in keys:
                    values[key][(int(row["mode_id"]), int(row["q_e4"]))] = candidate_from_row(row)
        finally:
            connection.close()
        for key in keys:
            SS.require(len(values[key]) == SS.MODE_COUNT * len(SS.Q_E4_GRID), f"{key}: reward grid incomplete")
            for cell, value in values[key].items():
                SS.require((value is None) == (self._scenes[key]["grid"][cell][0] is None),
                           f"{key} {cell}: reward and Q_perc definedness differ")
        self._value = values
        self.reward_binding_sha256 = hashlib.sha256(json.dumps(
            {"catalog": self.binding_sha256, "reward": REWARD_SHA256}).encode()).hexdigest()

    def reward_value(self, scene_key: str, *, mode_id: int, q_e4: int) -> Optional[float]:
        grid = self._value[scene_key]
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
