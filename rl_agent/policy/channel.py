"""Measured channel anchors and reproducible Markov channel process."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import REPO_ROOT
from .types import ChannelSnapshot


RUNG_BY_MCS = {28: "clear", 24: "mild", 19: "mid", 9: "strong"}


@dataclass(frozen=True)
class MeasuredRung:
    name: str
    mcs: int
    representative_snr_db: float
    nominal_capacity_mbps: float
    capture_to_map_p50_90_ms: float
    front_to_edge_p95_90_ms: float


class ChannelSurface:
    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = config
        self.combined_path = REPO_ROOT / "channel_condition_sweep" / "combined_surface.csv"
        self.raw_90_path = (
            REPO_ROOT / "uplink_only_spatial_map_pipeline" / "results" / "chsweep_full_p3ae_.csv"
        )
        combined = pd.read_csv(self.combined_path)
        raw_90 = pd.read_csv(self.raw_90_path)
        combined["rung"] = combined["mcs"].map(RUNG_BY_MCS)
        if combined["rung"].isna().any() or len(combined) != 12:
            raise ValueError("combined channel surface must contain the four known MCS rungs x three payloads")
        if combined.groupby("rung")["payload"].nunique().to_dict() != {
            "clear": 3,
            "mid": 3,
            "mild": 3,
            "strong": 3,
        }:
            raise ValueError("combined channel surface is not a complete 3x4 surface")
        raw_90["rung"] = raw_90["mcs_p50"].map(RUNG_BY_MCS)
        row_90 = combined[combined["payload"] == "90 KB"].set_index("rung")
        p95_90 = raw_90.set_index("rung")["front_to_edge_p95_ms"]
        self.rungs: Dict[str, MeasuredRung] = {}
        for name, values in config["channel"]["rungs"].items():
            mcs = int(values["mcs"])
            measured = row_90.loc[name]
            if int(measured["mcs"]) != mcs:
                raise ValueError(f"MCS mismatch for rung {name}")
            self.rungs[name] = MeasuredRung(
                name=name,
                mcs=mcs,
                representative_snr_db=float(values["representative_snr_db"]),
                nominal_capacity_mbps=float(values["capacity_mbps"]),
                capture_to_map_p50_90_ms=float(measured["cap2map_p50_ms"]),
                front_to_edge_p95_90_ms=float(p95_90.loc[name]),
            )
        self.source_hashes = {
            str(self.combined_path.relative_to(REPO_ROOT)): hashlib.sha256(self.combined_path.read_bytes()).hexdigest(),
            str(self.raw_90_path.relative_to(REPO_ROOT)): hashlib.sha256(self.raw_90_path.read_bytes()).hexdigest(),
        }


class ChannelProcess:
    """Four-rung Markov channel with episode-stable capacity uncertainty per rung."""

    def __init__(
        self,
        config: Mapping[str, object],
        surface: ChannelSurface,
        seed: int,
        fixed_rungs: Optional[Sequence[str]] = None,
        fixed_capacity_multiplier: Optional[float] = None,
    ) -> None:
        self.config = config
        self.surface = surface
        self.rng = np.random.default_rng(seed)
        self.fixed_rungs = list(fixed_rungs) if fixed_rungs is not None else None
        self.fixed_capacity_multiplier = fixed_capacity_multiplier
        self.order = list(config["channel"]["rungs"])
        self.lag_steps = int(config["channel"]["telemetry_lag_steps"])
        self.noise_fraction = float(config["channel"]["estimate_noise_fraction"])
        uncertainty = float(config["channel"]["capacity_uncertainty_fraction"])
        if fixed_capacity_multiplier is None:
            self.multipliers = {
                name: float(self.rng.uniform(1.0 - uncertainty, 1.0 + uncertainty)) for name in self.order
            }
        else:
            self.multipliers = {name: float(fixed_capacity_multiplier) for name in self.order}
        self.current_rung = str(config["channel"]["initial_rung"])
        if self.fixed_rungs:
            self.current_rung = str(self.fixed_rungs[0])
        self.capacity_history: List[float] = []
        self.rung_history: List[str] = []
        self.step_index = 0

    def _choose_next_rung(self) -> str:
        if self.fixed_rungs is not None:
            return self.fixed_rungs[min(self.step_index, len(self.fixed_rungs) - 1)]
        row = self.config["channel"]["transition_matrix"][self.current_rung]
        probabilities = [float(row[name]) for name in self.order]
        return str(self.rng.choice(self.order, p=probabilities))

    def snapshot(self) -> ChannelSnapshot:
        rung = self.surface.rungs[self.current_rung]
        true_capacity = rung.nominal_capacity_mbps * self.multipliers[self.current_rung]
        if self.lag_steps == 0:
            lagged = true_capacity
            observed_rung = self.current_rung
        elif len(self.capacity_history) >= self.lag_steps:
            # At control step t, history[-lag_steps] is the completed sample
            # from t-lag_steps. Using -1-lag_steps adds an unintended extra tick.
            lagged = self.capacity_history[-self.lag_steps]
            observed_rung = self.rung_history[-self.lag_steps]
        elif self.capacity_history:
            lagged = self.capacity_history[0]
            observed_rung = self.rung_history[0]
        else:
            lagged = rung.nominal_capacity_mbps
            observed_rung = self.current_rung
        noise = float(self.rng.normal(0.0, self.noise_fraction * max(lagged, 1e-6)))
        estimate = max(0.1, lagged + noise)
        return ChannelSnapshot(
            rung=self.current_rung,
            mcs=rung.mcs,
            observed_rung=observed_rung,
            observed_mcs=self.surface.rungs[observed_rung].mcs,
            true_capacity_mbps=true_capacity,
            estimated_capacity_mbps=estimate,
            estimate_sigma_mbps=self.noise_fraction * lagged,
            representative_snr_db=rung.representative_snr_db,
        )

    def advance(self) -> None:
        current = self.surface.rungs[self.current_rung]
        self.capacity_history.append(current.nominal_capacity_mbps * self.multipliers[self.current_rung])
        self.rung_history.append(self.current_rung)
        self.step_index += 1
        self.current_rung = self._choose_next_rung()
