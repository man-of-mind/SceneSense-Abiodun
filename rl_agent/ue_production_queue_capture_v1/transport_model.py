#!/usr/bin/env python3
"""Consumer for the exported production-transport model.

This is the artifact that **replaces** the old 288 `application_feature_uplink`
latency component.  It is never added to it.

Composition used by downstream modeled training, per source row:

    total_new = retained_source_total
              - retained_uplink            (first send -> edge enqueue)
              + measured_send_span         (first -> last socket handoff)
              + predicted_transport        (last handoff -> complete reassembly)

The removed interval `[first send, reassembly]` is tiled exactly by the two
added intervals, which is what the Phase-0 retained-evidence identity
`application_feature_uplink_ms == ue_send_loop_ms + post_send_to_reassembly_ms`
proves.  Nothing is double counted and no gap is introduced.

Out-of-support requests are refused, never extrapolated.
"""

from __future__ import annotations

import ast
import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import contract as C


MODEL_SCHEMA = "scenesense.production_transport_model.v1"


class TransportModelError(RuntimeError):
    """The model artifact is foreign, or a request is outside measured support."""


class OutOfSupport(TransportModelError):
    """Refused rather than extrapolated."""


def _require(condition: bool, message: str, error=TransportModelError) -> None:
    if not condition:
        raise error(message)


@dataclass(frozen=True, slots=True)
class TransportPrediction:
    latency_ns: int
    rate_bps: float
    source: str
    within_deadline: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ns": self.latency_ns, "rate_bps": self.rate_bps,
            "source": self.source, "within_deadline": self.within_deadline,
        }


class ProductionTransportModelV1:
    """Loaded, digest-bound, refuse-outside-support transport model."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        _require(document.get("schema") == MODEL_SCHEMA,
                 "transport model schema drifted")
        _require(document.get("contract_sha256") == C.CONTRACT_SHA256,
                 "transport model is not bound to the frozen capture contract")
        _require(bool(document.get("all_gates_passed")),
                 "transport model did not pass its registered gates")
        _require(document.get("replaces_288_component")
                 == C.REPLACED_288_COMPONENT,
                 "transport model does not declare the 288 replacement")
        _require(bool(document.get("adding_both_components_is_forbidden")),
                 "transport model must forbid additive composition")
        self._document = dict(document)
        self._bins = {
            self._parse_key(key): value
            for key, value in document["bins"].items()
        }
        self._global = float(document["global_median_rate_bps"])
        support = document["measured_support"]
        self._min_bytes = int(support["min_udp_application_bytes"])
        self._max_bytes = int(support["max_udp_application_bytes"])
        self._min_backlog = float(support["min_pre_action_backlog_bytes"])
        self._max_backlog = float(support["max_pre_action_backlog_bytes"])
        self._min_support = int(document["min_bin_support"])
        binning = document["binning"]
        self._backlog_edges = [
            math.inf if value is None else float(value)
            for value in binning["backlog_edges"]
        ]
        self._mcs_edges = [
            math.inf if value is None else float(value)
            for value in binning["mcs_edges"]
        ]

    @staticmethod
    def _parse_key(key: str) -> tuple:
        value = ast.literal_eval(key)
        return value if isinstance(value, tuple) else (value,)

    @classmethod
    def load(cls, path: Path) -> "ProductionTransportModelV1":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def document_sha256(self) -> str:
        return C.canonical_sha256(self._document)

    @property
    def measured_support(self) -> dict[str, Any]:
        return dict(self._document["measured_support"])

    def _backlog_bin(self, value: float) -> int:
        return max(0, bisect.bisect_right(self._backlog_edges, value) - 1)

    def _mcs_bin(self, value: float) -> int:
        return max(0, bisect.bisect_right(self._mcs_edges, value) - 1)

    def rate(self, *, backlog_bytes: float, ul_mcs: int | None
             ) -> tuple[float, str]:
        candidates: list[tuple] = []
        if ul_mcs is not None:
            candidates.append((self._backlog_bin(backlog_bytes),
                               self._mcs_bin(ul_mcs)))
        candidates.append((self._backlog_bin(backlog_bytes),))
        if ul_mcs is not None:
            candidates.append((self._mcs_bin(ul_mcs),))
        candidates.append(())
        for key in candidates:
            entry = self._bins.get(key)
            if entry is not None and int(entry["count"]) >= self._min_support:
                return float(entry["median_rate_bps"]), f"BIN{key}"
        return self._global, "GLOBAL"

    def predict(
        self, *, backlog_bytes: float, udp_application_bytes: int,
        ul_mcs: int | None, allow_out_of_support: bool = False,
    ) -> TransportPrediction:
        if not allow_out_of_support:
            if not self._min_bytes <= udp_application_bytes <= self._max_bytes:
                raise OutOfSupport(
                    f"{udp_application_bytes} bytes is outside the measured "
                    f"support [{self._min_bytes}, {self._max_bytes}]")
            if not self._min_backlog <= backlog_bytes <= self._max_backlog:
                raise OutOfSupport(
                    f"backlog {backlog_bytes} is outside the measured support "
                    f"[{self._min_backlog}, {self._max_backlog}]")
        rate, source = self.rate(backlog_bytes=backlog_bytes, ul_mcs=ul_mcs)
        _require(rate > 0, "fitted rate must be positive")
        latency_ns = int(round((backlog_bytes + udp_application_bytes)
                               / rate * 1e9))
        _require(latency_ns > 0, "predicted transport latency must be positive")
        return TransportPrediction(
            latency_ns=latency_ns, rate_bps=rate, source=source,
            within_deadline=latency_ns <= C.REWARD_DEADLINE_NS)


@dataclass(frozen=True, slots=True)
class RetainedSourceRow:
    """One retained action-50 probe row, decomposed at the replacement seam."""

    row_id: str
    profile_label: str
    total_ns: int
    uplink_ns: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        _require(self.total_ns > 0, "retained total must be positive")
        _require(self.uplink_ns > 0, "retained uplink must be positive")
        _require(self.uplink_ns < self.total_ns,
                 "retained uplink must be strictly inside the retained total")

    @property
    def residual_ns(self) -> int:
        """Everything the new transport model does NOT replace."""
        return self.total_ns - self.uplink_ns


@dataclass(frozen=True, slots=True)
class ComposedLatency:
    """The replaced, non-additive total handed to the modeled composite."""

    source_total_ns: int
    retained_residual_ns: int
    measured_send_span_ns: int
    predicted_transport_ns: int
    actor_reserve_ns: int
    total_ns: int
    replacement: str
    prediction: TransportPrediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_total_ns": self.source_total_ns,
            "retained_residual_ns": self.retained_residual_ns,
            "measured_send_span_ns": self.measured_send_span_ns,
            "predicted_transport_ns": self.predicted_transport_ns,
            "actor_reserve_ns": self.actor_reserve_ns,
            "total_ns": self.total_ns,
            "replacement": self.replacement,
            "prediction": self.prediction.to_dict(),
        }


def compose_latency(
    *, retained: RetainedSourceRow, prediction: TransportPrediction,
    measured_send_span_ns: int, actor_reserve_ns: int,
) -> ComposedLatency:
    """Replace the retained uplink segment; never add the two."""
    _require(measured_send_span_ns >= 0, "send span cannot be negative")
    _require(actor_reserve_ns > 0,
             "a positive actor reserve is mandatory; zero actor cost is refused")
    source_total = (retained.residual_ns + measured_send_span_ns
                    + prediction.latency_ns)
    _require(source_total > 0, "composed source total must be positive")
    return ComposedLatency(
        source_total_ns=source_total,
        retained_residual_ns=retained.residual_ns,
        measured_send_span_ns=measured_send_span_ns,
        predicted_transport_ns=prediction.latency_ns,
        actor_reserve_ns=actor_reserve_ns,
        total_ns=source_total + actor_reserve_ns,
        replacement=(
            "REPLACED_288_FEATURE_UPLINK_WITH_MEASURED_SEND_SPAN_PLUS_"
            "MODELED_PRODUCTION_TRANSPORT__NOT_ADDED"),
        prediction=prediction,
    )
