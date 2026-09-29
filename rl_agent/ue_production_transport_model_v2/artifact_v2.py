#!/usr/bin/env python3
"""Phase E: versioned v2 artifact whose exporter and consumer share one equation.

The v1 exporter and its runtime consumer had drifted into different field sets
and a different latency equation.  That cannot recur here: the consumer does
not reimplement anything.  It rebuilds the exact
``model_v2.DeadlineHead`` / ``model_v2.LatencyHead`` dataclasses from the
artifact and calls the *same* methods the fitter used, so export -> load ->
predict is bit-identical by construction and is asserted by a round-trip test.

The artifact is unusable unless its gates actually passed.  Flipping
``all_gates_passed`` alone is not sufficient: the loader recomputes the gate
verdict from the recorded metrics and refuses any disagreement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rl_agent.ue_production_queue_capture_v1 import contract as V1

from . import contract_v2 as C2
from . import model_v2 as M


ARTIFACT_SCHEMA = "scenesense.production_transport_model.v2"
ARTIFACT_VERSION = 2


class ArtifactError(RuntimeError):
    """The artifact is foreign, ungated, internally inconsistent, or unusable."""


class OutOfSupport(ArtifactError):
    """Refused rather than extrapolated, and never silently clipped."""


def _require(condition: bool, message: str, error=ArtifactError) -> None:
    if not condition:
        raise error(message)


def gate_verdicts(metrics: Mapping[str, Any]) -> dict[str, bool]:
    """Recomputed from the recorded metrics; never read from a stored flag."""
    return {
        "DEADLINE_HEAD_CALIBRATION": (
            float(metrics["brier"]) <= 0.15
            and float(metrics["false_success_rate"]) <= 0.05),
        "CONDITIONAL_ON_TIME_LATENCY_ERROR": (
            float(metrics["latency_abs_error_p50_ms"]) <= 17.0
            and float(metrics["latency_abs_error_p95_ms"]) <= 34.0),
        "REWARD_ERROR": (
            float(metrics["reward_error_p50"]) <= 0.025
            and float(metrics["reward_error_p95"]) <= 0.05),
        "MONOTONICITY": int(metrics["monotonicity_violations"]) == 0,
        "CAUSAL_JOIN_INTEGRITY": float(metrics["causal_coverage"]) == 1.0,
        "FEATURE_PROVENANCE": bool(metrics["feature_provenance_passed"]),
    }


def export_artifact(
    *, model: M.TwoPartModel, cv_metrics: Mapping[str, Any],
    causal_coverage: float, feature_provenance_passed: bool,
    monotonicity_violations: int, action_ranking: Mapping[str, Any],
    raw_backlog_support: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = {
        "brier": cv_metrics["brier"],
        "false_success_rate": cv_metrics["false_success_rate"],
        "latency_abs_error_p50_ms": cv_metrics["latency_abs_error_p50_ms"],
        "latency_abs_error_p95_ms": cv_metrics["latency_abs_error_p95_ms"],
        "reward_error_p50": cv_metrics["reward_error_p50"],
        "reward_error_p95": cv_metrics["reward_error_p95"],
        "monotonicity_violations": monotonicity_violations,
        "causal_coverage": causal_coverage,
        "feature_provenance_passed": feature_provenance_passed,
        "confusion": dict(cv_metrics.get("confusion", {})),
        "n": cv_metrics["n"], "n_on_time": cv_metrics["n_on_time"],
    }
    verdicts = gate_verdicts(metrics)
    document = {
        "schema": ARTIFACT_SCHEMA, "version": ARTIFACT_VERSION,
        "contract_v2_sha256": C2.CONTRACT_V2_SHA256,
        "v1_contract_sha256": V1.CONTRACT_SHA256,
        "evidence_class": C2.EVIDENCE_CLASS,
        "confirmatory": C2.CONFIRMATORY,
        "validation_population_status": C2.VALIDATION_POPULATION_STATUS,
        "v1_gate5_status": C2.V1_GATE5_STATUS,
        "model_family": C2.MODEL_FAMILY, "model_seed": C2.MODEL_SEED,
        "selection_protocol": C2.SELECTION_PROTOCOL,
        "equations": {
            "deadline_head":
                "p = sigmoid(w0 - w_backlog*backlog_scaled "
                "- w_bytes*(wire_bytes/1e6) + w_mcs*(mcs/28))",
            "latency_head":
                "latency_ms = 170 * sigmoid(z0 + a_backlog*backlog_scaled "
                "+ b_bytes*(wire_bytes/1e6) - c_mcs*(mcs/28))",
            "queue_head":
                "B_next = max(0, B + deterministic_ingress - service(MCS))",
            "backlog_scaled": C2.BACKLOG_FEATURE_FORM,
        },
        "deadline_head": model.deadline.to_dict(),
        "latency_head": model.latency.to_dict(),
        "queue_head": model.queue.to_dict(),
        "backlog_normalization": {
            "reference_bytes": C2.RLC_AM_TX_ADMISSION_CEILING_BYTES,
            "reference_rule": C2.BACKLOG_REFERENCE_RULE,
            "semantics": C2.BACKLOG_FEATURE_SEMANTICS,
            "oai_source_pins": dict(C2.OAI_CEILING_SOURCE_PINS),
            "clip_is_not_a_support_check":
                C2.NORMALIZATION_CLIP_IS_NOT_A_SUPPORT_CHECK,
        },
        "fit_support": dict(model.support),
        "raw_backlog_support": dict(raw_backlog_support),
        "metrics": metrics,
        "gate_verdicts": verdicts,
        "all_gates_passed": all(verdicts.values()),
        "action_ranking_sensitivity": dict(action_ranking),
        "replaces_288_component": V1.REPLACED_288_COMPONENT,
        "boundary": V1.PRODUCTION_TRANSPORT_BOUNDARY,
        "shared_endpoint": V1.SHARED_ENDPOINT,
        "adding_both_components_is_forbidden":
            V1.ADDING_BOTH_COMPONENTS_IS_FORBIDDEN,
        "deadline_ns": V1.REWARD_DEADLINE_NS,
        "out_of_support_policy": "REFUSE_DO_NOT_EXTRAPOLATE",
        "disclosures": [
            C2.EVIDENCE_CLASS,
            "NOT_CONFIRMATORY_EVIDENCE",
            "V1_GATE5_REMAINS_FAILED",
            "SINGLE_UE_RADIO_CONFIGURATION_ONLY",
            "PROFILE_TRANSFER_UNVALIDATED",
            "MODE_TRANSFER_UNVALIDATED",
            "PAYLOAD_INTERPOLATION_UNVALIDATED",
            "NOT_PERCEPTION_ENDORSEMENT",
            "NOT_P_ADMIT__INTERNAL_TRAINING_ENVIRONMENT_MODEL_ONLY",
        ],
    }
    return document


@dataclass(frozen=True, slots=True)
class TransportPredictionV2:
    on_time_probability: float
    conditional_latency_ms: float
    within_deadline: bool

    def to_dict(self) -> dict[str, Any]:
        return {"on_time_probability": self.on_time_probability,
                "conditional_latency_ms": self.conditional_latency_ms,
                "within_deadline": self.within_deadline}


class ProductionTransportModelV2:
    """Consumer that reuses the fitter's own equations verbatim."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        _require(document.get("schema") == ARTIFACT_SCHEMA,
                 "artifact schema drifted")
        _require(document.get("version") == ARTIFACT_VERSION,
                 "artifact version drifted")
        _require(document.get("contract_v2_sha256") == C2.CONTRACT_V2_SHA256,
                 "artifact is not bound to the frozen v2 contract")
        _require(document.get("v1_contract_sha256") == V1.CONTRACT_SHA256,
                 "artifact is not bound to the frozen v1 capture contract")
        _require(bool(document.get("adding_both_components_is_forbidden")),
                 "artifact must forbid additive 288 composition")
        _require(document.get("replaces_288_component")
                 == V1.REPLACED_288_COMPONENT,
                 "artifact does not declare the 288 replacement")

        # Recompute the verdict; a flipped flag alone cannot enable the model.
        recomputed = gate_verdicts(document["metrics"])
        _require(recomputed == document.get("gate_verdicts"),
                 "recorded gate verdicts disagree with the recorded metrics")
        _require(document.get("all_gates_passed") is all(recomputed.values()),
                 "all_gates_passed disagrees with the recomputed verdicts")
        _require(all(recomputed.values()),
                 "artifact did not pass its registered gates")

        normalization = document["backlog_normalization"]
        _require(normalization["reference_bytes"]
                 == C2.RLC_AM_TX_ADMISSION_CEILING_BYTES,
                 "artifact backlog reference is not the bound OAI ceiling")

        self._document = dict(document)
        self._deadline = M.DeadlineHead(**document["deadline_head"])
        self._latency = M.LatencyHead(**document["latency_head"])
        queue = document["queue_head"]
        self._queue = M.QueueTransitionHead(
            service_by_mcs_bin={int(k): float(v) for k, v
                                in queue["service_by_mcs_bin"].items()},
            global_service_bytes=float(queue["global_service_bytes"]),
            support=dict(queue["support"]))
        support = document["fit_support"]
        self._min_wire = float(support["min_wire_bytes"])
        self._max_wire = float(support["max_wire_bytes"])
        self._min_mcs = float(support["min_prior_ul_mcs"])
        self._max_mcs = float(support["max_prior_ul_mcs"])
        raw = document["raw_backlog_support"]
        self._min_backlog = float(raw["fit_min_bytes"])
        self._max_backlog = float(raw["fit_max_bytes"])

    @classmethod
    def load(cls, path: Path) -> "ProductionTransportModelV2":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def document_sha256(self) -> str:
        return C2.canonical_sha256(self._document)

    @property
    def fit_support(self) -> dict[str, Any]:
        return dict(self._document["fit_support"])

    def predict(
        self, *, pre_enqueue_backlog_bytes: float, wire_bytes: int,
        prior_ul_mcs: int, allow_out_of_support: bool = False,
    ) -> TransportPredictionV2:
        if not allow_out_of_support:
            # The RAW backlog is checked, never the clipped feature.
            if not self._min_backlog <= pre_enqueue_backlog_bytes <= self._max_backlog:
                raise OutOfSupport(
                    f"raw backlog {pre_enqueue_backlog_bytes} outside fitted "
                    f"support [{self._min_backlog}, {self._max_backlog}]; "
                    "invoke the registered external fallback")
            if not self._min_wire <= wire_bytes <= self._max_wire:
                raise OutOfSupport(
                    f"wire bytes {wire_bytes} outside fitted support "
                    f"[{self._min_wire}, {self._max_wire}]")
            if not self._min_mcs <= prior_ul_mcs <= self._max_mcs:
                raise OutOfSupport(
                    f"prior UL MCS {prior_ul_mcs} outside fitted support "
                    f"[{self._min_mcs}, {self._max_mcs}]")
        features = M.Features(
            backlog_scaled=np.array(
                [C2.backlog_scaled(float(pre_enqueue_backlog_bytes))]),
            backlog_bytes=np.array([float(pre_enqueue_backlog_bytes)]),
            bytes_mb=np.array([float(wire_bytes) / M.BYTES_SCALE]),
            wire_bytes=np.array([float(wire_bytes)]),
            mcs_norm=np.array([float(prior_ul_mcs) / M.MCS_SCALE]))
        probability = float(self._deadline.probability(features)[0])
        latency = float(self._latency.latency_ms(features)[0])
        return TransportPredictionV2(
            on_time_probability=probability,
            conditional_latency_ms=latency,
            within_deadline=latency <= V1.REWARD_DEADLINE_MS)

    def predict_next_backlog_bytes(
        self, *, pre_enqueue_backlog_bytes: float,
        deterministic_action_ingress_bytes: int, prior_ul_mcs: int,
        allow_out_of_support: bool = False,
    ) -> float:
        """B_next = max(0, B + deterministic ingress - service(MCS)).

        ``service`` comes from the fitted causal queue-transition head, which
        was estimated from FIT transitions only.  No realized ingress, no
        measured service and no future observation is consulted, and the raw
        backlog is checked against the fitted support first.
        """
        if not allow_out_of_support:
            if not (self._min_backlog <= pre_enqueue_backlog_bytes
                    <= self._max_backlog):
                raise OutOfSupport(
                    f"raw backlog {pre_enqueue_backlog_bytes} outside fitted "
                    f"support [{self._min_backlog}, {self._max_backlog}]; "
                    "invoke the registered external fallback")
            if not self._min_mcs <= prior_ul_mcs <= self._max_mcs:
                raise OutOfSupport(
                    f"prior UL MCS {prior_ul_mcs} outside fitted support")
        value, _source = self._queue.next_backlog_bytes(
            backlog_bytes=float(pre_enqueue_backlog_bytes),
            ingress_bytes=float(deterministic_action_ingress_bytes),
            mcs=float(prior_ul_mcs))
        return value

    def service_bytes_per_cycle(self, prior_ul_mcs: int) -> float:
        return self._queue.service_bytes(float(prior_ul_mcs))[0]
