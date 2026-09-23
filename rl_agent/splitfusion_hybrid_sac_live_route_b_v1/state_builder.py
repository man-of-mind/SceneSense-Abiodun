"""Genesis-matched 31-D live policy state, built only from admitted evidence.

What this module does
---------------------

It assembles a :class:`CausalStateV1` from a timestamped scene observation and
an *already attested* radio observation, vectorizes it with the **registered
Run-3 preprocessing** -- ``build_registered_normalization_spec`` and
``build_registered_freshness_policy``, imported unchanged from the training
environment -- and audits every resulting feature against the Run-3 training
support.

What it refuses to do
---------------------

*It never manufactures a measurement.*  There is no default SNR, no default
MCS, no default BSR and no default age.  ``RadioObservationV1`` must arrive
already constructed and attested by whoever actually measured it; this module
only checks that the record is attested, in the right clock domain, in the
right units for the registered normalization spec, and not in the future.  A
missing, stale, cross-domain or unattested observation raises.

*It never chains a previous outcome.*  Run-3 was trained exclusively on
independent one-step genesis observations, so the frozen actor has literally
never seen a non-zero previous-outcome feature.  Inventing one for the first
live pilot would feed the policy an input outside anything it was fitted on and
then attribute the result to the policy.  :class:`LivePolicyStateBuilderV1`
therefore requires an explicit episode-start proof on every frame and refuses a
``PreviousOutcomeV1`` outright.  Ticket feedback is still recorded -- it gates
decisions and it is the reward -- it simply does not enter the observation.
That is the whole content of
``GENESIS_MATCHED_FROZEN_POLICY_LIVE_PILOT_NOT_SEQUENTIAL_ONLINE_ADAPTATION``.

The honest part
---------------

Because Run-3's genesis states had exactly zero measurement ages and an exactly
empty uplink buffer, 27 of the 31 features were constant zero in training.  A
live frame cannot reproduce that and should not pretend to: a real age is not
zero and a real buffer is not always empty.  So the builder splits the 31
features in two.

* The 22 previous-outcome features are *policy controlled*.  The pilot decides
  them, they must be exactly zero, and a non-zero value is a defect --
  :meth:`LivePolicyStateBuilderV1.build` raises.
* The 9 measured features carry live values.  Out-of-support values are
  recorded per feature in :class:`TrainingSupportAuditV1` and surfaced in the
  evidence, never clamped, zeroed or otherwise repaired.

A run whose freshness features are persistently far from zero has not failed to
be built; it has produced a measured statement about how far the live operating
point sits from the training support, which is exactly what a deployment
validation is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_environment import (
    build_registered_freshness_policy,
    build_registered_normalization_spec,
)
from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
    CausalStateError,
    CausalStateV1,
    ClockDomain,
    EpisodeStartProofV1,
    POLICY_FEATURE_ORDER,
    PolicyFeatureVectorV1,
    PreviousOutcomeV1,
    RadioObservationV1,
    SceneObservationV1,
    StateFreshnessPolicyV1,
    StateNormalizationSpecV1,
    build_policy_features,
)

from . import pilot_contract as contract

__all__ = [
    "LivePolicyStateBuilderV1",
    "LivePolicyStateV1",
    "SequentialObservationRefusedError",
    "StateBuilderError",
    "TrainingSupportAuditV1",
]


class StateBuilderError(RuntimeError):
    """A live policy state could not be built from admissible evidence."""


class SequentialObservationRefusedError(StateBuilderError):
    """A chained previous-outcome observation was offered to a genesis pilot."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StateBuilderError(message)


# --------------------------------------------------------------------------- #
# Training-support audit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TrainingSupportAuditV1:
    """Per-feature comparison of one live observation against Run-3's support."""

    out_of_support: Tuple[Tuple[str, float, float, float], ...]
    measured_feature_count: int
    policy_controlled_feature_count: int
    training_support_limitation: str

    @property
    def fully_in_support(self) -> bool:
        return not self.out_of_support

    @property
    def out_of_support_features(self) -> Tuple[str, ...]:
        return tuple(name for name, _value, _low, _high in self.out_of_support)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "fully_in_support": self.fully_in_support,
            "measured_feature_count": self.measured_feature_count,
            "out_of_support": [
                {"feature": name, "high": high, "low": low, "value": value}
                for name, value, low, high in self.out_of_support
            ],
            "out_of_support_count": len(self.out_of_support),
            "policy_controlled_feature_count": self.policy_controlled_feature_count,
            "record": "splitfusion.live_route_b_pilot_support_audit.v1",
            "training_support_limitation": self.training_support_limitation,
        }


def _audit_support(values: Mapping[str, float]) -> TrainingSupportAuditV1:
    findings = []
    for name in contract.MEASURED_LIVE_FEATURES:
        descriptor = contract.TRAINING_SUPPORT[name]
        value = float(values[name])
        if not descriptor.contains(value):
            findings.append((name, value, descriptor.low, descriptor.high))
    return TrainingSupportAuditV1(
        out_of_support=tuple(findings),
        measured_feature_count=len(contract.MEASURED_LIVE_FEATURES),
        policy_controlled_feature_count=len(
            contract.POLICY_CONTROLLED_ZERO_FEATURES
        ),
        training_support_limitation=contract.TRAINING_SUPPORT_LIMITATION,
    )


# --------------------------------------------------------------------------- #
# Built state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LivePolicyStateV1:
    """One verified live observation, ready for the frozen actor."""

    state: CausalStateV1
    features: PolicyFeatureVectorV1
    support_audit: TrainingSupportAuditV1
    measurement_ages_ns: Mapping[str, int]
    normalization_spec_sha256: str
    freshness_policy_sha256: str
    state_sha256: str
    features_sha256: str
    genesis_matched: bool
    pilot_label: str
    pilot_contract_sha256: str

    @property
    def values(self) -> Tuple[float, ...]:
        """The 31 floats in the registered order, ready for ``decide``."""
        return self.features.as_tuple()

    def as_mapping(self) -> Dict[str, float]:
        return self.features.as_mapping()

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "carla_frame_id": self.state.carla_frame_id,
            "features": [float(value) for value in self.values],
            "features_sha256": self.features_sha256,
            "feature_order": list(POLICY_FEATURE_ORDER),
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "genesis_matched": self.genesis_matched,
            "measurement_ages_ns": dict(self.measurement_ages_ns),
            "normalization_spec_sha256": self.normalization_spec_sha256,
            "observed_ns": self.state.observed_ns,
            "pilot_contract_sha256": self.pilot_contract_sha256,
            "pilot_label": self.pilot_label,
            "raw": {
                "bsr_bytes": self.state.radio.bsr_bytes,
                "camera_si": self.state.camera_si,
                "mcs_index": self.state.radio.mcs_index,
                "radar_p40": self.state.radar_p40,
                "snr_db": self.state.radio.achieved_snr_db,
            },
            "record": "splitfusion.live_route_b_pilot_state.v1",
            "state_sha256": self.state_sha256,
            "support_audit": self.support_audit.to_canonical_dict(),
            "tensor_seq": self.state.tensor_seq,
        }

    def canonical_sha256(self) -> str:
        return contract.canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


class LivePolicyStateBuilderV1:
    """Builds genesis-matched live observations with the registered Run-3 scaling.

    The normalization spec and freshness policy are the *training* ones by
    default and cannot be swapped silently: overriding either requires passing
    it explicitly, and the resulting digests travel in every record, so a
    changed preprocessing constant is visible in the evidence.
    """

    def __init__(
        self,
        *,
        normalization: Optional[StateNormalizationSpecV1] = None,
        freshness: Optional[StateFreshnessPolicyV1] = None,
    ) -> None:
        self._normalization = (
            build_registered_normalization_spec()
            if normalization is None
            else normalization
        )
        self._freshness = (
            build_registered_freshness_policy() if freshness is None else freshness
        )
        _require(
            isinstance(self._normalization, StateNormalizationSpecV1),
            "normalization must be a StateNormalizationSpecV1",
        )
        _require(
            isinstance(self._freshness, StateFreshnessPolicyV1),
            "freshness must be a StateFreshnessPolicyV1",
        )
        self._uses_registered_preprocessing = (
            normalization is None and freshness is None
        )

    @property
    def normalization(self) -> StateNormalizationSpecV1:
        return self._normalization

    @property
    def freshness(self) -> StateFreshnessPolicyV1:
        return self._freshness

    @property
    def uses_registered_run3_preprocessing(self) -> bool:
        return self._uses_registered_preprocessing

    @property
    def normalization_spec_sha256(self) -> str:
        return self._normalization.canonical_sha256()

    @property
    def freshness_policy_sha256(self) -> str:
        return self._freshness.canonical_sha256()

    def build(
        self,
        *,
        scene: SceneObservationV1,
        radio: RadioObservationV1,
        episode_start: EpisodeStartProofV1,
        session_uuid: str,
        observed_ns: int,
        tensor_seq: int,
        carla_frame_id: int,
        clock_domain: ClockDomain = ClockDomain.UE_LOCAL_MONOTONIC,
        previous: Optional[PreviousOutcomeV1] = None,
    ) -> LivePolicyStateV1:
        """Assemble, validate and vectorize one live decision-time observation.

        Raises:
            SequentialObservationRefusedError: if ``previous`` is supplied.
            StateBuilderError: if the radio record is unattested or was not
                produced for this observation instant, or if a policy-controlled
                feature is non-zero.
            StaleTelemetryError: if any measurement exceeds its registered
                freshness bound.  Nothing is substituted for a stale value.
        """
        if previous is not None:
            raise SequentialObservationRefusedError(
                "the first live pilot is genesis matched: Run-3 was trained "
                "only on independent one-step genesis observations, so a "
                "chained previous-outcome feature is outside everything the "
                "frozen actor was fitted on.  Ticket feedback is recorded and "
                "gates decisions, but it is not a policy input in this phase "
                f"({contract.PILOT_LABEL})"
            )
        if not isinstance(scene, SceneObservationV1):
            raise StateBuilderError(
                f"scene must be a SceneObservationV1, got {type(scene).__name__}"
            )
        if not isinstance(radio, RadioObservationV1):
            raise StateBuilderError(
                f"radio must be a RadioObservationV1 produced and attested by "
                f"the measuring collector, got {type(radio).__name__}.  This "
                f"builder never constructs radio evidence, because it has not "
                f"measured anything"
            )
        radio.require_attested()
        if not isinstance(episode_start, EpisodeStartProofV1):
            raise StateBuilderError(
                "a genesis-matched state requires an explicit "
                "EpisodeStartProofV1 issued by the reward-ticket controller"
            )

        try:
            state = CausalStateV1(
                scene=scene,
                radio=radio,
                session_uuid=session_uuid,
                observed_ns=observed_ns,
                tensor_seq=tensor_seq,
                carla_frame_id=carla_frame_id,
                clock_domain=clock_domain,
                previous=None,
                episode_start=episode_start,
            )
        except CausalStateError as exc:
            raise StateBuilderError(
                f"the live observation is not a valid causal state: {exc}"
            ) from exc

        # build_policy_features checks freshness first and unconditionally, and
        # checks that the radio semantics are the quantity this spec scales.
        # StaleTelemetryError is allowed to propagate: the caller's guard must
        # see a stale observation as stale, not as a builder failure.
        features = build_policy_features(state, self._normalization, self._freshness)
        named = features.as_mapping()

        for name in contract.POLICY_CONTROLLED_ZERO_FEATURES:
            value = named[name]
            _require(
                value == 0.0,
                f"policy-controlled feature {name!r} is {value!r}, not 0.0.  "
                f"Run-3 never observed a non-zero previous-outcome feature, so "
                f"this state would take the frozen actor outside its training "
                f"support in a way the pilot itself caused",
            )

        return LivePolicyStateV1(
            state=state,
            features=features,
            support_audit=_audit_support(named),
            measurement_ages_ns=dict(state.measurement_ages_ns),
            normalization_spec_sha256=self.normalization_spec_sha256,
            freshness_policy_sha256=self.freshness_policy_sha256,
            state_sha256=state.canonical_sha256(),
            features_sha256=features.canonical_sha256(),
            genesis_matched=True,
            pilot_label=contract.PILOT_LABEL,
            pilot_contract_sha256=contract.PILOT_CONTRACT_SHA256,
        )
