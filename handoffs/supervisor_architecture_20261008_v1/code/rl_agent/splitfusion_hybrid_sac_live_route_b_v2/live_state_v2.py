"""Exact Run-4 21-feature live state from typed, already-causal observations.

No telemetry parsing or clock estimation happens here (that is
``ue_telemetry_provider_v2``).  This module only joins typed observations and
applies, in order:

1. ``run4_contract.guard_state_for_action`` with the exact training
   freshness policy (100 ms, digest ``6c694ebe…``);
2. the registered raw-support refusal of the v2 transport model
   (``out_of_support_policy = REFUSE_DO_NOT_EXTRAPOLATE``): prior UL MCS in
   ``[8, 28]`` and raw pre-enqueue backlog in ``[0, 36_002_536]`` bytes.  The
   raw value is checked *before* the feature, so the ``min(1, ·)`` clip can
   never hide an out-of-support backlog;
3. ``run4_contract.build_policy_features`` with the exact training scaling
   (digest ``cb1a3e4d…``).

Any refusal raises ``ExternalFallbackRequired`` and the actor must not run.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract

from .ue_telemetry_provider_v2 import TRAINING_FRESHNESS

__all__ = [
    "TRAINING_SCALING",
    "TRAINING_SCALING_SHA256",
    "RADIO_SUPPORT",
    "build_live_state",
    "scene_observations",
]

TRAINING_SCALING = contract.EmpiricalScalingV2(
    scaling_id="run4-production-transport-v2-scaling",
    scaling_version=1,
    evidence_sha256=(
        "23a73949d54278dc6e04dd043944806c98241f08e83d0416488cd7d39be4b79b"),
    camera_si_center=117.08553307797729,
    camera_si_scale=13.336440703846854,
    backlog_log1p_scale=math.log1p(50_000_000),
)
TRAINING_SCALING_SHA256 = (
    "cb1a3e4d9c9478a9106287fc3cf8b1bac5f92cfe5294fbf32543da0679155671")
if TRAINING_SCALING.canonical_sha256() != TRAINING_SCALING_SHA256:
    raise RuntimeError("training empirical scaling reconstruction drifted")

# rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/
# transport_model_v2.json: fit_support.{min,max}_prior_ul_mcs and
# raw_backlog_support.fit_{min,max}_bytes.
RADIO_SUPPORT = {"prior_ul_mcs": (8, 28), "backlog_bytes": (0, 36_002_536)}


def scene_observations(
    identity: contract.DecisionIdentityV1, *, sample_seq: int, camera_si: float,
    radar_p40: float, source: str, source_timestamp_ns: int,
    available_timestamp_ns: int, clock_domain: str,
) -> Tuple[contract.ScalarObservationV1, contract.ScalarObservationV1]:
    """Camera SI and radar P40 of one exact scene sample (shared identity)."""
    def observation(kind, value):
        return contract.ScalarObservationV1(
            value=float(value),
            metadata=contract.MeasurementMetadataV1(
                identity=contract.SampleIdentityV1(
                    identity.session_uuid, identity.ue_id, sample_seq),
                kind=kind, observer=contract.Observer.SCENE_PIPELINE,
                link_direction=contract.LinkDirection.NOT_APPLICABLE,
                source=source, source_timestamp_ns=source_timestamp_ns,
                available_timestamp_ns=available_timestamp_ns,
                clock_domain=clock_domain, valid=True),
            missing_reason=None)
    return (observation(contract.MeasurementKind.CAMERA_SI, camera_si),
            observation(contract.MeasurementKind.RADAR_P40, radar_p40))


def build_live_state(
    *, identity: contract.DecisionIdentityV1, boundary: contract.DecisionBoundaryV1,
    camera_si: contract.ScalarObservationV1, radar_p40: contract.ScalarObservationV1,
    prior_ul_mcs: contract.PriorUlGrantObservationV1,
    pre_action_rlc_backlog: contract.ScalarObservationV1,
    previous: Optional[contract.PreviousOutcomeV1],
) -> Tuple[contract.GuardedPolicyStateV2, contract.PolicyFeatureVectorV2]:
    state = contract.PolicyStateV2(
        identity=identity, camera_si=camera_si, radar_p40=radar_p40,
        prior_ul_mcs=prior_ul_mcs, pre_action_rlc_backlog=pre_action_rlc_backlog,
        previous=previous)
    guarded = contract.guard_state_for_action(state, boundary, TRAINING_FRESHNESS)
    mcs = prior_ul_mcs.observation.value
    backlog = pre_action_rlc_backlog.value
    low, high = RADIO_SUPPORT["prior_ul_mcs"]
    if not low <= mcs <= high:
        raise contract.ExternalFallbackRequired(
            f"prior UL MCS {mcs} outside registered training support [{low}, {high}]")
    low, high = RADIO_SUPPORT["backlog_bytes"]
    if not low <= backlog <= high:
        raise contract.ExternalFallbackRequired(
            f"raw backlog {backlog} B outside registered training support "
            f"[{low}, {high}]; clipping must not hide it")
    features = contract.build_policy_features(guarded, TRAINING_SCALING)
    registered = min(1.0, math.log1p(backlog) / math.log1p(50_000_000))
    if features.as_dict()["pre_action_rlc_backlog_log1p_scaled"] != registered:
        raise contract.ScalingError("backlog feature differs from the registered form")
    return guarded, features
