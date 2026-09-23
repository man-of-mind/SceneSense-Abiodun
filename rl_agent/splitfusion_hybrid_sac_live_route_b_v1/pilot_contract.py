"""Frozen labels, artifact pins and training-support descriptor for the pilot.

This module is the single source of truth for *what* the live Route-B pilot is
bound to.  It contains no logic beyond validation of its own literals, reads no
file, imports no ``torch``, and initializes nothing.

Three separable things are pinned here.

**The pre-registered actor.**  Seed 17, update 10,000, of the completed Run-3
three-seed campaign.  Seed 17 is the *first registered seed* and was chosen
before any live result existed.  Substituting a seed on the basis of live
performance would turn a pre-registered pilot into a selected one, so the seed
and update are module constants rather than parameters.

**The artifact chain.**  ``campaign_complete.json`` names each seed report's
canonical digest; the seed report names every artifact file's SHA-256; the
model-only snapshot carries its own recomputable content digest and the
runner-binding digest.  :mod:`checkpoint_loader` walks that chain, so the pins
below are the chain's two endpoints, not a replacement for it.

**The Run-3 training support.**  This is the part that matters scientifically
and is easy to get wrong, so it is stated explicitly.

The Run-3 collector drew *independent one-step genesis observations* from the
partitioned empirical environment.  Reading the registered environment's own
hard invariants, rather than guessing:

* every genesis state carries ``previous=None`` and an explicit episode-start
  proof, so all 22 previous-outcome features are exactly ``0.0``
  (12 mode one-hots, 4 terminal one-hots, ``prev_q``, ``prev_quality``,
  ``prev_latency`` and the 3 masks);
* ``empirical_contextual_environment`` raises if any measurement age is
  non-zero, so all 4 normalized freshness features are exactly ``0.0``;
* ``empirical_radio_context`` writes ``bsr_bytes=0`` on every sampled row with
  the explicit ``GENESIS_BSR_JUSTIFICATION``, so ``radio_bsr_log1p_scaled`` is
  exactly ``0.0``;
* the radio store refuses to load unless the achieved-SNR range is exactly
  ``[6.0, 23.5]`` dB and the MCS-median range is exactly ``[9.0, 28.0]``;
* the registered normalization spec clips camera SI to
  ``[84.53893280029297, 147.98822021484375]``.

**Therefore 27 of the 31 policy features were identically zero throughout
Run-3 training, and only 4 varied**: scaled camera SI, radar P40, scaled
achieved SNR and scaled MCS.  A live decision-time observation will not
reproduce that: real measurement ages are not zero and a real uplink buffer is
not always empty.  Feeding the actor a *causal* live value for those fields is
correct and required; it is also, unavoidably, out of the training support.
The pilot therefore records a per-feature support audit on every frame instead
of pretending the question does not exist.  See
:data:`TRAINING_SUPPORT_LIMITATION`.

Zeroing a live age or a live BSR to "stay in distribution" would be fabricated
telemetry and is explicitly refused by :mod:`state_builder`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

__all__ = [
    "ANALYSIS_SCHEMA_ID",
    "CAMPAIGN_COMPLETE_RELATIVE_PATH",
    "CAMPAIGN_RELATIVE_PATH",
    "EVIDENCE_FRAME_SCHEMA_ID",
    "EVIDENCE_SESSION_SCHEMA_ID",
    "FULL_CHECKPOINT_FILE_SHA256",
    "FULL_CHECKPOINT_RELATIVE_PATH",
    "MEASURED_LIVE_FEATURES",
    "PILOT_CONTRACT_SHA256",
    "PILOT_LABEL",
    "PILOT_PHASE",
    "PILOT_SCOPE",
    "PREREGISTERED_SEED",
    "PREREGISTERED_UPDATE",
    "POLICY_CONTROLLED_ZERO_FEATURES",
    "REGISTERED_RUNNER_BINDING_SHA256",
    "REGISTERED_SEED17_SESSION_UUID",
    "REPORT_FILE_SHA256",
    "REPORT_RELATIVE_PATH",
    "SEED_DIRECTORY_RELATIVE_PATH",
    "SNAPSHOT_FILE_SHA256",
    "SNAPSHOT_RELATIVE_PATH",
    "SUPPORT_PROVENANCE",
    "TERMINAL_MARKER_RELATIVE_PATH",
    "TRAINING_SUPPORT",
    "TRAINING_SUPPORT_LIMITATION",
    "FeatureSupport",
    "PilotContractError",
    "canonical_json_bytes",
    "canonical_sha256",
    "contract_descriptor",
]


class PilotContractError(ValueError):
    """A pilot binding literal is malformed or internally contradictory."""


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

#: The mandatory scientific label for every artifact this pilot produces.
PILOT_LABEL = (
    "GENESIS_MATCHED_FROZEN_POLICY_LIVE_PILOT_NOT_SEQUENTIAL_ONLINE_ADAPTATION"
)

#: The phase this module's consumers are currently permitted to execute.
PILOT_PHASE = "PHASE1_AUDIT_AND_FROZEN_POLICY_RUNTIME_NO_LIVE_LAUNCH"

#: What the pilot is and, just as importantly, what it is not.
PILOT_SCOPE = (
    "DEPLOYMENT_VALIDATION_OF_A_FROZEN_ACTOR;"
    "NO_OPTIMIZER_NO_REPLAY_NO_PARAMETER_UPDATE_IN_THE_LIVE_PROCESS;"
    "REWARD_FEEDBACK_IS_RECORDED_AND_GATES_DECISIONS_BUT_CHANGES_NO_WEIGHT"
)

#: Stated once, carried into every evidence file.
TRAINING_SUPPORT_LIMITATION = (
    "RUN3_TRAINED_ON_INDEPENDENT_ONE_STEP_GENESIS_OBSERVATIONS;"
    "27_OF_31_FEATURES_WERE_IDENTICALLY_ZERO_IN_TRAINING;"
    "ONLY_CAMERA_SI_RADAR_P40_SNR_AND_MCS_VARIED;"
    "LIVE_MEASUREMENT_AGES_AND_UPLINK_BUFFER_ARE_CAUSALLY_NONZERO_AND_THEREFORE_"
    "OUT_OF_THE_TRAINING_SUPPORT;"
    "THIS_IS_RECORDED_PER_FRAME_AND_NEVER_REPAIRED_BY_ZEROING_A_MEASUREMENT"
)


# --------------------------------------------------------------------------- #
# Pre-registered actor and its artifact chain
# --------------------------------------------------------------------------- #

#: First registered seed of the three-seed campaign; chosen before live results.
PREREGISTERED_SEED = 17

#: The configured fixed endpoint; no peak-checkpoint selection was performed.
PREREGISTERED_UPDATE = 10_000

CAMPAIGN_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_run3_training_v1/"
    "20260922_three_seed_10000_updates_optimized_v1"
)
CAMPAIGN_COMPLETE_RELATIVE_PATH = f"{CAMPAIGN_RELATIVE_PATH}/campaign_complete.json"
SEED_DIRECTORY_RELATIVE_PATH = f"{CAMPAIGN_RELATIVE_PATH}/seed_{PREREGISTERED_SEED}"
REPORT_RELATIVE_PATH = f"{SEED_DIRECTORY_RELATIVE_PATH}/report.json"
TERMINAL_MARKER_RELATIVE_PATH = (
    f"{SEED_DIRECTORY_RELATIVE_PATH}/RUN3_TRAINING_COMPLETE.json"
)
SNAPSHOT_RELATIVE_PATH = (
    f"{SEED_DIRECTORY_RELATIVE_PATH}/model_snapshots/"
    f"model_{PREREGISTERED_UPDATE:06d}.pt"
)
FULL_CHECKPOINT_RELATIVE_PATH = (
    f"{SEED_DIRECTORY_RELATIVE_PATH}/checkpoints/"
    f"full_{PREREGISTERED_UPDATE:06d}.pt"
)

#: Endpoints of the verification chain.  The intermediate links (report
#: ``artifact_hashes``, the snapshot's own content digest, the campaign's
#: per-seed report digest) are recomputed by the loader, not pinned here.
SNAPSHOT_FILE_SHA256 = (
    "7ea8e2ca2e77995a98d7e544ebd67c62b0f3402e59ba457adf1fbef842adcf7c"
)
FULL_CHECKPOINT_FILE_SHA256 = (
    "47e0bcccecb378fd587b61a66b332f3f17213962662b341b29f7ae94d65fe0b1"
)
REPORT_FILE_SHA256 = (
    "0ce534f41c231d960958ae4954fd9b686694d50a6ab1e262e6dc5a27c1ce8078"
)

#: The seed-17 runner binding carried inside the model-only snapshot.  The
#: runner module asserts this lineage itself; duplicating it here lets the
#: loader fail before any tensor is copied into a live actor.
REGISTERED_RUNNER_BINDING_SHA256 = (
    "6e82a29fbe56a3836318b88536ee46cb9d86172dd65b2dd2a2d5790fff4d5775"
)
REGISTERED_SEED17_SESSION_UUID = "6dc674b2-4fd5-5720-84b4-e59ffe1330a1"


# --------------------------------------------------------------------------- #
# Evidence and analysis schema identifiers
# --------------------------------------------------------------------------- #

EVIDENCE_SESSION_SCHEMA_ID = "splitfusion.live_route_b_pilot_session.v1"
EVIDENCE_FRAME_SCHEMA_ID = "splitfusion.live_route_b_pilot_frame.v1"
ANALYSIS_SCHEMA_ID = "splitfusion.live_route_b_pilot_analysis.v1"


# --------------------------------------------------------------------------- #
# Canonical serialization (same definition the rest of the repository uses)
# --------------------------------------------------------------------------- #


def canonical_json_bytes(payload: Any) -> bytes:
    """Sorted-key, separator-fixed, NaN-refusing UTF-8 JSON."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    """SHA-256 of :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


# --------------------------------------------------------------------------- #
# Run-3 training support
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeatureSupport:
    """Closed interval a feature occupied during Run-3 training, plus why.

    ``provenance`` names the registered mechanism that makes the interval a
    fact about the training run rather than an estimate.  ``exact_constant`` is
    True when the interval is a single point because a registered invariant
    forced it, which is the case for 27 of the 31 features.
    """

    low: float
    high: float
    provenance: str
    exact_constant: bool

    def __post_init__(self) -> None:
        if type(self.low) is not float or type(self.high) is not float:
            raise PilotContractError("support bounds must be exact floats")
        if not self.low <= self.high:
            raise PilotContractError(
                f"support interval is inverted: [{self.low}, {self.high}]"
            )
        if not self.provenance:
            raise PilotContractError("every support interval needs provenance")
        if self.exact_constant and self.low != self.high:
            raise PilotContractError(
                "a constant support must be a single point"
            )

    def contains(self, value: float) -> bool:
        return self.low <= float(value) <= self.high

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "exact_constant": self.exact_constant,
            "high": self.high,
            "low": self.low,
            "provenance": self.provenance,
        }


SUPPORT_PROVENANCE: Mapping[str, str] = MappingProxyType(
    {
        "GENESIS_PREVIOUS_BLOCK": (
            "state_reward_transition_contract.build_policy_features writes the "
            "whole previous-outcome block as zeros behind explicit masks when "
            "state.previous is None, and every Run-3 reset built a genesis "
            "state with previous=None"
        ),
        "GENESIS_ZERO_AGES": (
            "empirical_contextual_environment.reset raises "
            "'D1 genesis state has non-zero age' unless all four measurement "
            "ages are exactly zero, and the registered preflight records "
            "all_genesis_ages_zero=True"
        ),
        "GENESIS_EMPTY_BSR": (
            "empirical_radio_context samples every row with bsr_bytes=0 under "
            "GENESIS_BSR_JUSTIFICATION (empty queue observed before the "
            "selected action is enqueued); the registered normalization spec "
            "records bsr_scale_status=INERT_FOR_EXACT_GENESIS_ZERO_ONLY"
        ),
        "RADIO_STORE_ASSERTED_RANGE": (
            "empirical_radio_context refuses to load unless the achieved-SNR "
            "range is exactly (6.0, 23.5) dB and the MCS-median range is "
            "exactly (9.0, 28.0)"
        ),
        "REGISTERED_NORMALIZATION_CLIP": (
            "build_registered_normalization_spec clips camera SI to "
            "[84.53893280029297, 147.98822021484375] fitted over the 512 fit "
            "scenes, so the scaled feature is confined to [0, 1]"
        ),
        "CONTRACT_BOUND_ONLY": (
            "no registered Run-3 artifact states the empirical range of this "
            "feature; only the contract's structural bound is known, so the "
            "interval below is a bound and not a measured training range"
        ),
    }
)


def _constant_zero(provenance: str) -> FeatureSupport:
    return FeatureSupport(
        low=0.0, high=0.0, provenance=provenance, exact_constant=True
    )


def _build_training_support() -> Mapping[str, FeatureSupport]:
    support: Dict[str, FeatureSupport] = {
        # -- the four features that actually varied ------------------------ #
        "scene_camera_si_scaled": FeatureSupport(
            low=0.0,
            high=1.0,
            provenance="REGISTERED_NORMALIZATION_CLIP",
            exact_constant=False,
        ),
        "scene_radar_p40": FeatureSupport(
            low=0.0,
            high=1.0,
            provenance="CONTRACT_BOUND_ONLY",
            exact_constant=False,
        ),
        "radio_achieved_snr_db_scaled": FeatureSupport(
            low=0.0,
            high=1.0,
            provenance="RADIO_STORE_ASSERTED_RANGE",
            exact_constant=False,
        ),
        "radio_mcs_index_scaled": FeatureSupport(
            low=9.0 / 28.0,
            high=1.0,
            provenance="RADIO_STORE_ASSERTED_RANGE",
            exact_constant=False,
        ),
        # -- identically zero: empty genesis queue ------------------------- #
        "radio_bsr_log1p_scaled": _constant_zero("GENESIS_EMPTY_BSR"),
        # -- identically zero: every genesis age is exactly zero ----------- #
        "freshness_scene_normalized": _constant_zero("GENESIS_ZERO_AGES"),
        "freshness_snr_normalized": _constant_zero("GENESIS_ZERO_AGES"),
        "freshness_bsr_normalized": _constant_zero("GENESIS_ZERO_AGES"),
        "freshness_mcs_normalized": _constant_zero("GENESIS_ZERO_AGES"),
    }
    # -- identically zero: no predecessor decision exists at genesis ------- #
    for index in range(12):
        support[f"prev_joint_mode_onehot_{index:02d}"] = _constant_zero(
            "GENESIS_PREVIOUS_BLOCK"
        )
    for code in (
        "exact",
        "action_path_failure",
        "feedback_timeout",
        "infra_fault_excluded",
    ):
        support[f"prev_terminal_onehot_{code}"] = _constant_zero(
            "GENESIS_PREVIOUS_BLOCK"
        )
    for name in (
        "prev_q_normalized",
        "prev_quality_normalized",
        "prev_latency_normalized",
        "prev_present_mask",
        "prev_quality_valid_mask",
        "prev_latency_valid_mask",
    ):
        support[name] = _constant_zero("GENESIS_PREVIOUS_BLOCK")
    for descriptor in support.values():
        if descriptor.provenance not in SUPPORT_PROVENANCE:
            raise PilotContractError(
                f"support provenance {descriptor.provenance!r} is not registered"
            )
    return MappingProxyType(support)


#: ``feature name -> FeatureSupport`` for all 31 registered policy features.
TRAINING_SUPPORT: Mapping[str, FeatureSupport] = _build_training_support()

#: Features the pilot itself controls and must hold at exactly zero.  A
#: non-zero value here is an implementation defect -- the pilot decided to
#: chain a previous outcome the frozen actor never saw -- so it fails closed.
POLICY_CONTROLLED_ZERO_FEATURES: Tuple[str, ...] = tuple(
    sorted(
        name
        for name, descriptor in TRAINING_SUPPORT.items()
        if descriptor.provenance == "GENESIS_PREVIOUS_BLOCK"
    )
)

#: Features carrying a live measurement.  Out-of-support values here are a
#: scientific fact about the live environment, recorded per frame, never
#: repaired.
MEASURED_LIVE_FEATURES: Tuple[str, ...] = tuple(
    sorted(set(TRAINING_SUPPORT) - set(POLICY_CONTROLLED_ZERO_FEATURES))
)


def contract_descriptor() -> Dict[str, Any]:
    """The complete, serializable statement of what this pilot is bound to."""
    return {
        "analysis_schema_id": ANALYSIS_SCHEMA_ID,
        "artifacts": {
            "campaign": CAMPAIGN_RELATIVE_PATH,
            "campaign_complete": CAMPAIGN_COMPLETE_RELATIVE_PATH,
            "full_checkpoint": FULL_CHECKPOINT_RELATIVE_PATH,
            "full_checkpoint_file_sha256": FULL_CHECKPOINT_FILE_SHA256,
            "report": REPORT_RELATIVE_PATH,
            "report_file_sha256": REPORT_FILE_SHA256,
            "snapshot": SNAPSHOT_RELATIVE_PATH,
            "snapshot_file_sha256": SNAPSHOT_FILE_SHA256,
            "terminal_marker": TERMINAL_MARKER_RELATIVE_PATH,
        },
        "evidence_frame_schema_id": EVIDENCE_FRAME_SCHEMA_ID,
        "evidence_session_schema_id": EVIDENCE_SESSION_SCHEMA_ID,
        "label": PILOT_LABEL,
        "measured_live_features": list(MEASURED_LIVE_FEATURES),
        "phase": PILOT_PHASE,
        "policy_controlled_zero_features": list(POLICY_CONTROLLED_ZERO_FEATURES),
        "preregistered_seed": PREREGISTERED_SEED,
        "preregistered_update": PREREGISTERED_UPDATE,
        "record": "splitfusion.live_route_b_pilot_contract.v1",
        "registered_runner_binding_sha256": REGISTERED_RUNNER_BINDING_SHA256,
        "registered_seed17_session_uuid": REGISTERED_SEED17_SESSION_UUID,
        "scope": PILOT_SCOPE,
        "support_provenance": dict(SUPPORT_PROVENANCE),
        "training_support": {
            name: descriptor.to_canonical_dict()
            for name, descriptor in sorted(TRAINING_SUPPORT.items())
        },
        "training_support_limitation": TRAINING_SUPPORT_LIMITATION,
    }


#: Digest of this module's complete binding statement.  Carried by every
#: evidence record so a silent pin change is detectable after the fact.
PILOT_CONTRACT_SHA256: str = canonical_sha256(contract_descriptor())
