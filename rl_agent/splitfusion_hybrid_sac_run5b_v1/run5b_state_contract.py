"""Run-5B 21-feature state: the exact Run-4B 20-D state plus the causal UL-SNR proxy.

* Positions 0-19 are produced by the unchanged Run-4B builder
  ``splitfusion_hybrid_sac_run4b_v1.contract.build_features`` from the current
  observation and the Run-4B ``OperationalPriorV1`` (action, mode, q and
  operational ACK latency; no quality field).  Q_perc therefore cannot enter
  the state; it remains in the training reward only.
* Position 20 is ``effective_external_ul_snr_proxy_scaled`` under the frozen
  Run-5 v2 provider semantics: the modeled controller ACKs the active target
  and renews its lease before state commit on a virtual
  ``CLOCK_MONOTONIC_RAW`` timeline, the unchanged live ``observe`` rule selects
  it, and :func:`admit_snr` applies the lease/support rule.  Scaling is
  ``(snr_db - 5.5) / 19.0`` on ``[5.5, 24.5]``; anything else raises
  ``ExternalFallbackRequired`` and nothing is zero-filled or clipped.
* The SNR value is the joint channel's current causal sample (the decision's
  observed tick); future samples are generated only after the outcome.

Importing this module reads no files and starts nothing.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as R5V1

ExternalFallbackRequired = R4.ExternalFallbackRequired
SNR_FEATURE_NAME = R5V1.SNR_FEATURE_NAME
SNR_FEATURE_INDEX = 20
FEATURE_ORDER: Tuple[str, ...] = (*C4.FEATURE_ORDER, SNR_FEATURE_NAME)
FEATURE_COUNT = 21
MODE_COUNT = C4.MODE_COUNT
canonical_sha256 = C4.canonical_sha256
# Unchanged Run-4B contract objects the derived runner uses.
GAMMA, DISCOUNT, DURATION = C4.GAMMA, C4.DISCOUNT, C4.DURATION
REWARD_SCHEMA_SHA256 = C4.REWARD_SCHEMA_SHA256
ObservationV1, OperationalPriorV1, ScalingV1 = (C4.ObservationV1, C4.OperationalPriorV1,
                                                C4.ScalingV1)
CRITIC_INPUT_WIDTH = FEATURE_COUNT + C4.MODE_COUNT + 1          # 34
PRIOR_SLICE = slice(4, 20)
REMOVED_RUN4_FEATURE = "prev_quality_qperc"

# Run-4B's deny-list minus the deliberately added "snr", plus Run-5's.
FORBIDDEN_FEATURE_TOKENS: Tuple[str, ...] = tuple(
    t for t in (*C4.FORBIDDEN_FEATURE_TOKENS, *R5V1.RUN5_EXTRA_FORBIDDEN_TERMS) if t != "snr")

# Virtual RAW timeline of the modeled controller (frozen Run-5 offsets).
SESSION_PREFIX = "00000000-0000-4000-8000-"
UE_ID = "ue-modeled-run5b"
TIMELINE_ORIGIN_NS = 2_000_000_000
DECISION_PERIOD_NS = 200_000_000          # two 100-ms ticks per decision
COMMIT_TO_OPEN_NS = 10_000_000
ACK_LEAD_NS = 20_000_000                  # run5_collector.SNR_EFFECTIVE_LEAD_NS
HEARTBEAT_LEAD_NS = 5_000_000             # run5_collector.SNR_HEARTBEAT_LEAD_NS
LEASE_MAX_HEARTBEAT_AGE_NS = 200_000_000  # run5_collector.LEASE_MAX_HEARTBEAT_AGE_NS


class Run5BContractError(ValueError):
    pass


def assert_feature_schema() -> None:
    if FEATURE_ORDER[:C4.FEATURE_COUNT] != tuple(C4.FEATURE_ORDER):
        raise Run5BContractError("positions 0-19 are not the Run-4B order")
    if len(FEATURE_ORDER) != FEATURE_COUNT or len(set(FEATURE_ORDER)) != FEATURE_COUNT:
        raise Run5BContractError("Run-5B feature count/uniqueness drift")
    if FEATURE_ORDER[SNR_FEATURE_INDEX] != SNR_FEATURE_NAME:
        raise Run5BContractError("SNR is not feature 21 (index 20)")
    for name in FEATURE_ORDER:
        for token in FORBIDDEN_FEATURE_TOKENS:
            if token in name.lower():
                raise Run5BContractError(f"forbidden token {token!r} in {name!r}")


assert_feature_schema()

FEATURE_SCHEMA = {
    "schema_id": "splitfusion_run5b_policy_features_v2",
    "schema_version": 2,
    "feature_count": FEATURE_COUNT,
    "feature_order": list(FEATURE_ORDER),
    "positions_0_19": {"builder": "splitfusion_hybrid_sac_run4b_v1.contract.build_features",
                       "run4b_feature_schema_sha256": C4.FEATURE_SCHEMA_SHA256},
    "position_20": {"name": SNR_FEATURE_NAME, "label": R5V1.SNR_PROXY_LABEL,
                    "provider_rule": SNR.LEASE_SELECTION_RULE_ID,
                    "run5_v2_feature_schema_sha256": SNR.FEATURE_SCHEMA_SHA256,
                    "scaling": "(snr_db - 5.5) / 19.0 on [5.5, 24.5]; else external fallback",
                    "timeline": {"origin_ns": TIMELINE_ORIGIN_NS,
                                 "decision_period_ns": DECISION_PERIOD_NS,
                                 "commit_to_open_ns": COMMIT_TO_OPEN_NS,
                                 "ack_lead_ns": ACK_LEAD_NS,
                                 "heartbeat_lead_ns": HEARTBEAT_LEAD_NS,
                                 "lease_max_heartbeat_age_ns": LEASE_MAX_HEARTBEAT_AGE_NS}},
    "q_perc": "training reward only; the Run-4B operational prior has no quality field",
    "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
}
FEATURE_SCHEMA_SHA256 = C4.canonical_sha256(FEATURE_SCHEMA)
FEATURE_ORDER_SHA256 = C4.canonical_sha256(list(FEATURE_ORDER))


def lease_policy() -> SNR.SnrLeasePolicyV1:
    return SNR.SnrLeasePolicyV1(
        policy_id="run5-modeled-controller-lease", policy_version=1,
        evidence_sha256=SNR.NETWORK_PROFILE_DESIGN_SHA256,
        max_heartbeat_age_ns=LEASE_MAX_HEARTBEAT_AGE_NS)


def admit_snr(observation: Optional[SNR.UlSnrLeaseObservationV1],
              boundary: R4.DecisionBoundaryV1, lease: SNR.SnrLeasePolicyV1) -> float:
    """Frozen Run-5 v2 lease rule; returns the admitted raw dB or falls back."""
    if not isinstance(observation, SNR.UlSnrLeaseObservationV1):
        raise ExternalFallbackRequired("SNR observation is absent; use external fallback")
    if observation.kind is not R5V1.SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB:
        raise ExternalFallbackRequired("SNR observation has a foreign kind")
    if not observation.valid or observation.value_db is None:
        raise ExternalFallbackRequired(f"SNR is missing/invalid ({observation.missing_reason})")
    if (observation.identity.session_uuid != boundary.identity.session_uuid
            or observation.identity.ue_id != boundary.identity.ue_id
            or observation.controller_session_uuid != boundary.identity.session_uuid):
        raise ExternalFallbackRequired("SNR identity/controller session differs from decision")
    if observation.clock_domain != boundary.clock_domain:
        raise ExternalFallbackRequired("SNR and decision boundary use different clocks")
    if observation.heartbeat_command_id != observation.active_command_id:
        raise ExternalFallbackRequired("controller lease does not name the active command")
    commit = boundary.state_commit_timestamp_ns
    if observation.effective_since_ns > commit or observation.heartbeat_ns > commit:
        raise ExternalFallbackRequired("SNR command or lease was not available at commit")
    age = boundary.action_open_timestamp_ns - observation.heartbeat_ns
    if age <= 0 or age > lease.max_heartbeat_age_ns:
        raise ExternalFallbackRequired(f"controller lease is stale ({age} ns)")
    SNR.scale_snr_db(float(observation.value_db))   # support check; raises, never clips
    return float(observation.value_db)


class ModeledSnrObserverV1:
    """Modeled twin of the live controller for one seed's session.

    Per decision: ACK the command carrying the current causal target, renew
    its lease, then call the unchanged live ``observe`` at the commit cutoff.
    Idempotent per decision; consumes no environment RNG.
    """

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise Run5BContractError("seed must be an int >= 0")
        self.session_uuid = f"{SESSION_PREFIX}{seed % 10**12:012d}"
        self._adapter = SNR.ModeledLeaseSnrAdapterV1(
            provider_id="splitfusion.run5b.modeled_snr_observer.v1",
            session_uuid=self.session_uuid, ue_id=UE_ID)
        self._lease = lease_policy()
        self._cache: dict[int, Tuple[float, float]] = {}

    def boundary(self, decision_seq: int) -> R4.DecisionBoundaryV1:
        commit = TIMELINE_ORIGIN_NS + decision_seq * DECISION_PERIOD_NS
        return R4.DecisionBoundaryV1(
            identity=R4.DecisionIdentityV1(self.session_uuid, UE_ID, decision_seq),
            state_commit_timestamp_ns=commit,
            action_open_timestamp_ns=commit + COMMIT_TO_OPEN_NS,
            clock_domain=SNR.LIVE_CLOCK_DOMAIN)

    def observe(self, decision_seq: int, causal_snr_db: float) -> float:
        """Admitted, scaled feature 21 for this decision."""
        cached = self._cache.get(decision_seq)
        if cached is not None:
            if cached[0] != causal_snr_db:
                raise Run5BContractError("causal SNR changed within one decision")
            return cached[1]
        boundary = self.boundary(decision_seq)
        commit = boundary.state_commit_timestamp_ns
        command = f"run5b-modeled-effective-command-{decision_seq}"
        self._adapter.record_command_ack_at(at_ns=commit - ACK_LEAD_NS, command_id=command,
                                            status="ACK", clamped=False,
                                            target_snr_db=float(causal_snr_db))
        self._adapter.record_heartbeat_at(at_ns=commit - HEARTBEAT_LEAD_NS,
                                          active_command_id=command)
        admitted = admit_snr(self._adapter.observe(boundary), boundary, self._lease)
        if admitted != causal_snr_db:
            raise Run5BContractError("admitted SNR is not the decision's causal sample")
        value = float(SNR.scale_snr_db(admitted))
        self._cache = {decision_seq: (causal_snr_db, value)}
        return value


def build_features(observation: C4.ObservationV1, prior: C4.OperationalPriorV1,
                   scaling: C4.ScalingV1, snr_feature: float) -> Tuple[float, ...]:
    """Exact Run-4B 20-D vector, then the admitted scaled SNR."""
    prefix = C4.build_features(observation, prior, scaling)
    if type(snr_feature) is not float or not 0.0 <= snr_feature <= 1.0:
        raise ExternalFallbackRequired("SNR feature must be an admitted value in [0, 1]")
    return (*prefix, snr_feature)


def features_for(env: Any, observer: ModeledSnrObserverV1) -> Tuple[float, ...]:
    """Current Run-5B state of a joint-channel environment."""
    return build_features(env.observation(), env.prior, env.scaling,
                          observer.observe(env.decision_seq, env.current_snr_db()))
