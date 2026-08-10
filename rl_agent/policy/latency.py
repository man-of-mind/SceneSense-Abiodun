"""Declared p50/p95 latency projection rooted in the measured 90 KiB channel cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .catalog import Action
from .channel import ChannelSurface


@dataclass(frozen=True)
class LatencyEstimate:
    p50_ms: float
    p95_ms: float
    payload_anchor: str
    rate_provenance: str


class LatencyProjector:
    def __init__(self, config: Mapping[str, object], surface: ChannelSurface) -> None:
        self.config = config
        self.surface = surface
        spec = config["latency_projection"]
        self.anchor_payload = float(spec["anchor_payload_kib"])
        self.non_uplink_p95 = float(spec["fast_full_pipeline_p95_ms"]) - float(
            spec["fast_uplink_p95_ms"]
        )
        self.reference_compute = float(spec["reference_front_plus_back_ms"])
        self.p50_factors = {key: float(value) for key, value in spec["p50_serialization_factors"].items()}
        self.p95_factors = {key: float(value) for key, value in spec["p95_serialization_factors"].items()}

    def estimate(self, action: Action, rung_name: str) -> LatencyEstimate:
        if action.mode != "SPLIT":
            return LatencyEstimate(0.0, 0.0, "none", "none")
        rung = self.surface.rungs[rung_name]
        serialization = 8.192 / rung.nominal_capacity_mbps
        payload_delta = action.payload_kib - self.anchor_payload
        compute_delta = action.front_ms + action.back_ms - self.reference_compute
        p50 = (
            rung.capture_to_map_p50_90_ms
            + compute_delta
            + payload_delta * serialization * self.p50_factors[rung_name]
        )
        p95 = (
            self.non_uplink_p95
            + rung.front_to_edge_p95_90_ms
            + compute_delta
            + payload_delta * serialization * self.p95_factors[rung_name]
        )
        compute_floor = action.front_ms + action.back_ms
        p50 = max(compute_floor, p50)
        p95 = max(p50, compute_floor, p95)
        payload_anchor = "measured_90k_anchor" if abs(action.payload_kib - 90.0) <= 0.5 else "payload_projection"
        rate_provenance = "measured_rate_band" if 5.9 <= action.target_fps <= 8.0 else "fps_projection"
        return LatencyEstimate(p50, p95, payload_anchor, rate_provenance)
