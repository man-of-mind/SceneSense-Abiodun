"""Runtime evidence contract for the exact-16 Phase-2 factor smoke.

This module deliberately has no CARLA or OAI imports.  It provides two hard
boundaries needed by the factor-smoke collector:

* an exact, causal policy-feature loader (silently projecting a wider runtime
  dictionary is forbidden); and
* recipient-consumer availability/provenance records from which the primary C2
  endpoint and installed-track integrity diagnostics can be recomputed.

Ground-truth association is accepted only by the post-capture endpoint builder.
It is never stored in, or returned by, the runtime policy loader.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .causal_contract import DecisionRecord, FORBIDDEN_RUNTIME_SOURCES
from .schemas_v2 import _reject_forbidden_runtime_keys


POLICY_AUDIT_SCHEMA = "scenesense.phase2_causal_policy_runtime_audit.v1"
AVAILABILITY_SCHEMA = "scenesense.phase2_recipient_availability_provenance.v1"
GUARDRAIL_SCHEMA = "scenesense.phase2_installed_track_guardrails.v1"
MAP_TARGET_MATCH_SCHEMA = "scenesense.phase2_recipient_map_target_match.v1"
ENDPOINT_NAME = "recipient_available_confirmed_track_margin_s"
LOCAL_LOOPBACK = "local_loopback"
OAI_TRANSPORT = "oai"
TRANSPORT_MODES = frozenset({LOCAL_LOOPBACK, OAI_TRANSPORT})

# Flattened policy fields remain bound to the causal source categories already
# used by causal_contract.py; arbitrary caller-invented source labels are not
# sufficient evidence of causality.
FEATURE_SOURCE_STAGE = {
    "lagged_capacity_estimate_mbps": "prior_network_estimator",
    "capacity_estimate_sigma_mbps": "prior_network_estimator",
    "previous_action_code": "prior_completed_event",
    "previous_delivery_success": "prior_completed_event",
    "previous_latency_s": "prior_completed_event",
    "previous_loss_fraction": "prior_completed_event",
    "scheduler_credit_normalized": "scheduler_state",
    "in_flight_bytes": "scheduler_state",
    "in_flight_oldest_age_s": "scheduler_state",
    "installed_map_object_count": "recipient_map_feedback_transport",
    "installed_map_max_aoi_s": "recipient_map_feedback_transport",
    "installed_map_max_position_sigma_m": "recipient_map_feedback_transport",
    "installed_map_min_estimated_ttc_s": "recipient_map_feedback_transport",
    "prior_source_track_count": "causal_tracker",
    "helper_recipient_relative_x_m": "derived_relative_kinematics",
    "helper_recipient_relative_y_m": "derived_relative_kinematics",
    "helper_recipient_relative_vx_mps": "derived_relative_kinematics",
    "helper_recipient_relative_vy_mps": "derived_relative_kinematics",
    "recipient_speed_mps": "recipient_state_transport",
    "recipient_acceleration_mps2": "recipient_state_transport",
    "local_compute_headroom": "local_compute_monitor",
    "current_causal_track_count": "causal_tracker",
    "current_min_track_confidence": "causal_tracker",
    "current_min_estimated_ttc_s": "causal_tracker",
    "current_min_estimated_clearance_m": "causal_tracker",
    "current_max_position_sigma_m": "causal_tracker",
    "recipient_state_age_s": "recipient_state_transport",
}


def _required(name: str, value: object) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _finite(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(name: str, value: object) -> str:
    result = _required(name, value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return result


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_consumer_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _reject_token_keys(value: object, forbidden_tokens: Sequence[str], path: str) -> None:
    """Reject forbidden substrings recursively, including inside nested summaries."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            hits = [token for token in forbidden_tokens if token in normalized]
            if hits:
                raise ValueError(
                    f"{path}.{key} contains forbidden policy-key token(s): {', '.join(hits)}"
                )
            _reject_token_keys(item, forbidden_tokens, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_token_keys(item, forbidden_tokens, f"{path}[{index}]")


@dataclass(frozen=True)
class FeatureComponent:
    source_stage: str
    observed_at_s: float
    available_at_s: float


@dataclass(frozen=True)
class FeatureSample:
    """A value plus the causal metadata checked before policy consumption."""

    value: object
    source_stage: str
    observed_at_s: float
    available_at_s: float
    component_provenance: tuple[FeatureComponent, ...] = ()
    evidence_kind: str = "observed"


class CausalPolicyRuntimeAuditor:
    """The exact runtime policy loader and its create-in-memory audit trail."""

    def __init__(
        self,
        *,
        trajectory_id: str,
        arm_id: str,
        clock_id: str,
        placement_features: Sequence[str],
        publication_features: Sequence[str],
        forbidden_feature_tokens: Sequence[str],
        required_stages: Sequence[str] = ("placement", "publication"),
        minimum_decisions_per_stage: int = 1,
        canary_required: bool = True,
        decision_locus: str = "helper",
        projection_exercise: Mapping[str, Any] | None = None,
    ) -> None:
        self.trajectory_id = _required("trajectory_id", trajectory_id)
        self.arm_id = _required("arm_id", arm_id)
        self.clock_id = _required("clock_id", clock_id)
        self.features = {
            "placement": tuple(_required("placement feature", item) for item in placement_features),
            "publication": tuple(_required("publication feature", item) for item in publication_features),
        }
        for stage, names in self.features.items():
            if not names or len(names) != len(set(names)):
                raise ValueError(f"{stage} feature contract must be nonempty and unique")
        self.forbidden_tokens = tuple(
            _required("forbidden feature token", item).lower()
            for item in forbidden_feature_tokens
        )
        self.required_stages = tuple(_required("required stage", item) for item in required_stages)
        if set(self.required_stages) != {"placement", "publication"}:
            raise ValueError("factor smoke requires placement and publication audit stages")
        self.minimum_decisions_per_stage = int(minimum_decisions_per_stage)
        if self.minimum_decisions_per_stage < 1:
            raise ValueError("minimum_decisions_per_stage must be positive")
        self.canary_required = bool(canary_required)
        self.decision_locus = _required("decision_locus", decision_locus)
        if self.decision_locus not in {"helper", "edge", "recipient"}:
            raise ValueError("decision_locus must be helper, edge, or recipient")
        self._decisions: list[dict[str, Any]] = []
        self._canary_attempts = 0
        self._canary_rejections = 0
        self._canary_acceptances = 0
        self._state_exposures: list[dict[str, Any]] = []
        self.projection_exercise = dict(projection_exercise or {})

        for names in self.features.values():
            for name in names:
                hits = [token for token in self.forbidden_tokens if token in name.lower()]
                if hits:
                    raise ValueError(f"allowlisted feature {name!r} contains forbidden tokens {hits}")
                if name not in FEATURE_SOURCE_STAGE:
                    raise ValueError(f"allowlisted feature {name!r} has no frozen causal source")

    def record_policy_state_exposure(
        self,
        *,
        sample_at_s: float,
        source_track_count: int,
        installed_map_track_count: int,
    ) -> None:
        """Record track cardinality without pretending zero-track semantics exist.

        The exact-16 exercise audits the causal loader on a non-empty state.  A
        future live controller still needs an independently frozen missingness
        representation for zero-track states; recording their exposure here
        prevents this plumbing PASS from being read as environment readiness.
        """

        source_count = int(source_track_count)
        map_count = int(installed_map_track_count)
        if source_count < 0 or map_count < 0:
            raise ValueError("policy-state track counts cannot be negative")
        item = {
            "sample_at_s": _finite("sample_at_s", sample_at_s),
            "source_track_count": source_count,
            "installed_map_track_count": map_count,
        }
        item["record_sha256"] = canonical_sha256(item)
        self._state_exposures.append(item)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        trajectory_id: str,
        arm_id: str,
        clock_id: str,
        decision_locus: str = "helper",
    ) -> "CausalPolicyRuntimeAuditor":
        feature = config["policy_feature_contract"]
        audit = config["causal_policy_audit_contract"]
        canary = audit["forbidden_field_canary"]
        return cls(
            trajectory_id=trajectory_id,
            arm_id=arm_id,
            clock_id=clock_id,
            placement_features=feature["placement_features"],
            publication_features=feature["publication_features"],
            forbidden_feature_tokens=feature["forbidden_feature_tokens"],
            required_stages=audit["required_stages"],
            minimum_decisions_per_stage=int(
                audit["minimum_decisions_per_stage_per_trajectory"]
            ),
            canary_required=bool(canary["required_once_per_trajectory"]),
            decision_locus=decision_locus,
            projection_exercise=config.get("policy_projection_exercise", {}),
        )

    def _load(
        self,
        *,
        stage: str,
        decision_id: str,
        decision_at_s: float,
        action: str,
        samples: Mapping[str, FeatureSample],
        persist: bool,
    ) -> dict[str, object]:
        stage = _required("stage", stage)
        if stage not in self.features:
            raise ValueError("stage must be placement or publication")
        normalized_action = _required("action", action)
        fixed_actions = self.projection_exercise.get("fixed_actions", {})
        if isinstance(fixed_actions, Mapping) and stage in fixed_actions:
            expected_action = _required(
                f"fixed {stage} action", fixed_actions[stage]
            )
            if normalized_action != expected_action:
                raise ValueError(
                    f"{stage} projection exercise requires fixed action "
                    f"{expected_action!r}, not {normalized_action!r}"
                )
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a mapping")
        expected = self.features[stage]
        actual = tuple(str(name) for name in samples)
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
            raise ValueError(
                f"exact {stage} projection mismatch: missing={missing}, extra={extra}"
            )
        if len(actual) != len(set(actual)):
            raise ValueError("policy input contains duplicate feature names")

        decision = DecisionRecord(
            trajectory_id=self.trajectory_id,
            arm_id=self.arm_id,
            decision_id=_required("decision_id", decision_id),
            decision_stage=stage,
            decision_at_s=_finite("decision_at_s", decision_at_s),
            clock_id=self.clock_id,
            action=normalized_action,
        )
        decision.validate()
        fields: list[dict[str, Any]] = []
        projected: dict[str, object] = {}
        for name in expected:
            sample = samples[name]
            if not isinstance(sample, FeatureSample):
                raise TypeError(f"feature {name} must be a FeatureSample")
            evidence_kind = _required(f"{name}.evidence_kind", sample.evidence_kind)
            if evidence_kind not in {
                "observed",
                "preregistered_fixture",
                "local_loopback_transport_abstraction",
            }:
                raise ValueError(f"feature {name} evidence_kind is invalid")
            fixture_fields = set(self.projection_exercise.get("fixture_backed_fields", {}))
            if (name in fixture_fields) != (evidence_kind == "preregistered_fixture"):
                raise ValueError(
                    f"feature {name} fixture status differs from the preregistered exercise"
                )
            abstracted_contract = self.projection_exercise.get(
                "local_loopback_transport_abstracted_fields", {}
            )
            abstracted_fields = set(
                abstracted_contract.get("fields", {})
                if isinstance(abstracted_contract, Mapping)
                else abstracted_contract
            )
            if (name in abstracted_fields) != (
                evidence_kind == "local_loopback_transport_abstraction"
            ):
                raise ValueError(
                    f"feature {name} transport-abstraction status differs from the preregistered exercise"
                )
            source = _required(f"{name}.source_stage", sample.source_stage)
            if source in FORBIDDEN_RUNTIME_SOURCES or any(
                token in source.lower()
                for token in ("evaluation", "ground_truth", "future", "shadow")
            ):
                raise ValueError(f"feature {name} has forbidden runtime source {source!r}")
            if source != FEATURE_SOURCE_STAGE[name]:
                raise ValueError(
                    f"feature {name} must come from {FEATURE_SOURCE_STAGE[name]!r}, not {source!r}"
                )
            observed = _finite(f"{name}.observed_at_s", sample.observed_at_s)
            available = _finite(f"{name}.available_at_s", sample.available_at_s)
            components: list[dict[str, float | str]] = []
            if source == "derived_relative_kinematics":
                expected_recipient_source = (
                    "recipient_localization"
                    if self.decision_locus == "recipient"
                    else "recipient_state_transport"
                )
                by_source = {
                    _required("component source_stage", item.source_stage): item
                    for item in sample.component_provenance
                }
                expected_sources = {"helper_localization", expected_recipient_source}
                if set(by_source) != expected_sources:
                    raise ValueError(
                        f"relative feature {name} requires component provenance {sorted(expected_sources)}"
                    )
                component_observed = []
                component_available = []
                for component_source in sorted(by_source):
                    component = by_source[component_source]
                    component_observed_at = _finite(
                        f"{name}.{component_source}.observed_at_s",
                        component.observed_at_s,
                    )
                    component_available_at = _finite(
                        f"{name}.{component_source}.available_at_s",
                        component.available_at_s,
                    )
                    if component_available_at + 1e-12 < component_observed_at:
                        raise ValueError("relative feature component is available before observation")
                    component_observed.append(component_observed_at)
                    component_available.append(component_available_at)
                    components.append(
                        {
                            "source_stage": component_source,
                            "observed_at_s": component_observed_at,
                            "available_at_s": component_available_at,
                        }
                    )
                if abs(observed - max(component_observed)) > 1e-12:
                    raise ValueError(
                        f"relative feature {name} observed_at must equal its latest component observation"
                    )
                if abs(available - max(component_available)) > 1e-12:
                    raise ValueError(
                        f"relative feature {name} available_at must equal its latest component availability"
                    )
            elif sample.component_provenance:
                raise ValueError(f"non-derived feature {name} cannot claim component provenance")
            if available + 1e-12 < observed:
                raise ValueError(f"feature {name} is available before observation")
            if available > decision.decision_at_s + 1e-12:
                raise ValueError(f"feature {name} became available after the consuming decision")
            _reject_forbidden_runtime_keys({name: sample.value}, "policy_input")
            _reject_token_keys({name: sample.value}, self.forbidden_tokens, "policy_input")
            try:
                value_sha = canonical_sha256(sample.value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"feature {name} must be finite JSON") from exc
            projected[name] = sample.value
            fields.append(
                {
                    "feature_name": name,
                    "source_stage": source,
                    "observed_at_s": observed,
                    "available_at_s": available,
                    "value_sha256": value_sha,
                    "component_provenance": components,
                    "evidence_kind": evidence_kind,
                }
            )

        if persist:
            record = {
                "decision_id": decision.decision_id,
                "decision_stage": stage,
                "decision_at_s": decision.decision_at_s,
                "clock_id": decision.clock_id,
                "arm_id": decision.arm_id,
                "action": decision.action,
                "fields": fields,
            }
            record["record_sha256"] = canonical_sha256(record)
            self._decisions.append(record)
        return projected

    def consume(
        self,
        *,
        stage: str,
        decision_id: str,
        decision_at_s: float,
        action: str,
        samples: Mapping[str, FeatureSample],
    ) -> dict[str, object]:
        """Validate and return the only dictionary a policy may consume."""

        return self._load(
            stage=stage,
            decision_id=decision_id,
            decision_at_s=decision_at_s,
            action=action,
            samples=samples,
            persist=True,
        )

    def exercise_forbidden_canary(
        self,
        *,
        stage: str,
        decision_id: str,
        decision_at_s: float,
        action: str,
        valid_samples: Mapping[str, FeatureSample],
    ) -> None:
        """Prove on this trajectory that a wider, evaluation-bearing dict fails."""

        self._canary_attempts += 1
        widened: MutableMapping[str, FeatureSample] = dict(valid_samples)
        widened["ground_truth_id"] = FeatureSample(
            value="isolated-canary-never-policy-visible",
            source_stage="evaluation_truth",
            observed_at_s=float(decision_at_s),
            available_at_s=float(decision_at_s),
        )
        try:
            self._load(
                stage=stage,
                decision_id=decision_id,
                decision_at_s=decision_at_s,
                action=action,
                samples=widened,
                persist=False,
            )
        except (TypeError, ValueError) as exc:
            message = str(exc).lower()
            if "extra=" not in message and "forbidden" not in message and "evaluation" not in message:
                raise RuntimeError("canary failed for an unrelated reason") from exc
            self._canary_rejections += 1
            return
        self._canary_acceptances += 1
        raise RuntimeError("forbidden policy-field canary was accepted")

    def to_record(self) -> dict[str, Any]:
        counts = {
            stage: sum(item["decision_stage"] == stage for item in self._decisions)
            for stage in self.required_stages
        }
        for stage, count in counts.items():
            if count < self.minimum_decisions_per_stage:
                raise ValueError(f"no sufficient realized {stage} decisions were audited")
        if self.canary_required and (
            self._canary_attempts != 1
            or self._canary_rejections != 1
            or self._canary_acceptances != 0
        ):
            raise ValueError("exactly one rejected forbidden-field canary is required")
        result = {
            "schema": POLICY_AUDIT_SCHEMA,
            "trajectory_id": self.trajectory_id,
            "arm_id": self.arm_id,
            "decision_locus": self.decision_locus,
            "clock_id": self.clock_id,
            "consumer_enforces_exact_projection": True,
            "consumer_code_sha256": runtime_consumer_code_sha256(),
            "placement_features": list(self.features["placement"]),
            "publication_features": list(self.features["publication"]),
            "decision_counts": counts,
            "decisions": list(self._decisions),
            "forbidden_field_canary": {
                "canary_field_name": "ground_truth_id",
                "attempt_count": self._canary_attempts,
                "rejection_count": self._canary_rejections,
                "acceptance_count": self._canary_acceptances,
            },
            "projection_exercise": {
                "role": self.projection_exercise.get("role"),
                "policy_action_selected_from_features": self.projection_exercise.get(
                    "policy_action_selected_from_features"
                ),
                "policy_performance_evaluated": self.projection_exercise.get(
                    "policy_performance_evaluated"
                ),
                "observed_policy_state_complete": self.projection_exercise.get(
                    "observed_policy_state_complete"
                ),
                "fixed_actions": dict(
                    self.projection_exercise.get("fixed_actions", {})
                ),
                "fixture_backed_fields": sorted(
                    self.projection_exercise.get("fixture_backed_fields", {})
                ),
                "local_loopback_transport_abstracted_fields": sorted(
                    (
                        self.projection_exercise.get(
                            "local_loopback_transport_abstracted_fields", {}
                        ).get("fields", {})
                        if isinstance(
                            self.projection_exercise.get(
                                "local_loopback_transport_abstracted_fields", {}
                            ),
                            Mapping,
                        )
                        else self.projection_exercise.get(
                            "local_loopback_transport_abstracted_fields", []
                        )
                    )
                ),
            },
            "policy_state_exposure": {
                "sample_count": len(self._state_exposures),
                "zero_source_track_sample_count": sum(
                    item["source_track_count"] == 0 for item in self._state_exposures
                ),
                "zero_installed_map_track_sample_count": sum(
                    item["installed_map_track_count"] == 0
                    for item in self._state_exposures
                ),
                "zero_object_state_seen": any(
                    item["source_track_count"] == 0
                    or item["installed_map_track_count"] == 0
                    for item in self._state_exposures
                ),
                "samples": list(self._state_exposures),
                "zero_track_policy_state_handling_status": (
                    "future_controller_blocker_missingness_contract_not_frozen"
                ),
                "environment_readiness_claimed": False,
            },
        }
        result["audit_sha256"] = canonical_sha256(result)
        return result


class RecipientAvailabilityRecorder:
    """Record runtime provenance without accepting evaluation identities."""

    def __init__(
        self,
        *,
        trajectory_id: str,
        clock_id: str,
        transport_mode: str = LOCAL_LOOPBACK,
        clock_semantics: str = "monotonic_simulation_timestamp_s",
    ) -> None:
        self.trajectory_id = _required("trajectory_id", trajectory_id)
        self.clock_id = _required("clock_id", clock_id)
        self.transport_mode = _required("transport_mode", transport_mode)
        if self.transport_mode not in TRANSPORT_MODES:
            raise ValueError(f"transport_mode must be one of {sorted(TRANSPORT_MODES)}")
        self.clock_semantics = _required("clock_semantics", clock_semantics)
        self._source_observations: list[dict[str, Any]] = []
        self._confirmations: list[dict[str, Any]] = []
        self._install_attempts: list[dict[str, Any]] = []
        self._recipient_local_installs: list[dict[str, Any]] = []
        self._map_tracks: list[dict[str, Any]] = []

    def register_source_observation(
        self,
        *,
        source_role: str,
        source_track_id: str,
        observation_sha256: str,
        observed_at_s: float,
    ) -> None:
        role = _required("source_role", source_role)
        if role not in {"helper", "recipient"}:
            raise ValueError("source_role must be helper or recipient")
        record = {
            "source_role": role,
            "source_track_id": _required("source_track_id", source_track_id),
            "observation_sha256": _sha256("observation_sha256", observation_sha256),
            "observed_at_s": _finite("observed_at_s", observed_at_s),
        }
        if any(
            (item["source_role"], item["source_track_id"], item["observation_sha256"])
            == (record["source_role"], record["source_track_id"], record["observation_sha256"])
            for item in self._source_observations
        ):
            raise ValueError("source observation provenance is duplicated")
        self._source_observations.append(record)

    def record_source_confirmation(
        self, *, source_role: str, source_track_id: str, confirmed_at_s: float
    ) -> None:
        role = _required("source_role", source_role)
        track = _required("source_track_id", source_track_id)
        candidates = [
            item
            for item in self._source_observations
            if item["source_role"] == role and item["source_track_id"] == track
        ]
        if not candidates:
            raise ValueError("confirmation lacks registered source-observation provenance")
        at_s = _finite("confirmed_at_s", confirmed_at_s)
        if at_s + 1e-12 < min(item["observed_at_s"] for item in candidates):
            raise ValueError("confirmation cannot precede its source observation")
        self._confirmations.append(
            {"source_role": role, "source_track_id": track, "confirmed_at_s": at_s}
        )

    def record_install_attempt(
        self,
        *,
        attempt_id: str,
        contribution_id: str,
        source_role: str,
        source_track_id: str,
        source_observation_sha256: str,
        published_at_s: float,
        attempted_at_s: float,
        install_status: str,
        recipient_map_track_id: str | None = None,
        installed_at_s: float | None = None,
        available_at_s: float | None = None,
    ) -> None:
        attempt = _required("attempt_id", attempt_id)
        if any(item["attempt_id"] == attempt for item in self._install_attempts):
            raise ValueError("install attempt_id is duplicated")
        status = _required("install_status", install_status)
        if status not in {"accepted", "rejected"}:
            raise ValueError("install_status must be accepted or rejected")
        published = _finite("published_at_s", published_at_s)
        attempted = _finite("attempted_at_s", attempted_at_s)
        if published > attempted + 1e-12:
            raise ValueError("install attempt cannot precede publication")
        role = _required("source_role", source_role)
        if role not in {"helper", "recipient"}:
            raise ValueError("source_role must be helper or recipient")
        record: dict[str, Any] = {
            "attempt_id": attempt,
            "contribution_id": _required("contribution_id", contribution_id),
            "source_track_id": _required("source_track_id", source_track_id),
            "source_role": role,
            "source_observation_sha256": _sha256(
                "source_observation_sha256", source_observation_sha256
            ),
            "published_at_s": published,
            "attempted_at_s": attempted,
            "install_status": status,
            "clock_id": self.clock_id,
            "transport_mode": self.transport_mode,
        }
        if status == "accepted":
            map_id = _required("recipient_map_track_id", recipient_map_track_id)
            installed = _finite("installed_at_s", installed_at_s)
            available = _finite("available_at_s", available_at_s)
            if not published <= installed <= available:
                raise ValueError("publish/install/available timestamps are not monotone")
            record.update(
                {
                    "recipient_map_track_id": map_id,
                    "installed_at_s": installed,
                    "available_at_s": available,
                }
            )
            self._register_map_provenance(
                recipient_map_track_id=map_id,
                provenance_kind="installed_source_track",
                provenance_ref=record["attempt_id"],
            )
        elif any(value is not None for value in (recipient_map_track_id, installed_at_s, available_at_s)):
            raise ValueError("rejected install cannot claim a recipient map track or install time")
        self._install_attempts.append(record)

    def record_recipient_local_install(
        self,
        *,
        local_install_id: str,
        source_track_id: str,
        source_observation_sha256: str,
        recipient_map_track_id: str,
        confirmed_at_s: float,
        installed_at_s: float,
        available_at_s: float,
    ) -> None:
        """Record when a recipient-self track reaches the same consumer boundary."""

        install_id = _required("local_install_id", local_install_id)
        if any(
            item["local_install_id"] == install_id
            for item in self._recipient_local_installs
        ):
            raise ValueError("local_install_id is duplicated")
        track_id = _required("source_track_id", source_track_id)
        observation_sha = _sha256(
            "source_observation_sha256", source_observation_sha256
        )
        if not any(
            item["source_role"] == "recipient"
            and item["source_track_id"] == track_id
            and item["observation_sha256"] == observation_sha
            for item in self._source_observations
        ):
            raise ValueError("recipient local install lacks source-observation provenance")
        confirmed = _finite("confirmed_at_s", confirmed_at_s)
        installed = _finite("installed_at_s", installed_at_s)
        available = _finite("available_at_s", available_at_s)
        if not confirmed <= installed <= available:
            raise ValueError("recipient confirmation/install/availability timestamps are not monotone")
        map_id = _required("recipient_map_track_id", recipient_map_track_id)
        self._recipient_local_installs.append(
            {
                "local_install_id": install_id,
                "source_role": "recipient",
                "source_track_id": track_id,
                "source_observation_sha256": observation_sha,
                "recipient_map_track_id": map_id,
                "confirmed_at_s": confirmed,
                "installed_at_s": installed,
                "available_at_s": available,
                "clock_id": self.clock_id,
                "consumer_boundary": "recipient_map_policy_consumer",
            }
        )
        self._register_map_provenance(
            recipient_map_track_id=map_id,
            provenance_kind="recipient_local_install",
            provenance_ref=install_id,
        )

    def _register_map_provenance(
        self,
        *,
        recipient_map_track_id: str,
        provenance_kind: str,
        provenance_ref: str,
    ) -> None:
        existing = next(
            (
                item
                for item in self._map_tracks
                if item["recipient_map_track_id"] == recipient_map_track_id
            ),
            None,
        )
        event = {
            "provenance_kind": provenance_kind,
            "provenance_ref": provenance_ref,
        }
        if existing is None:
            self._map_tracks.append(
                {
                    "recipient_map_track_id": recipient_map_track_id,
                    "provenance_events": [event],
                }
            )
        elif event not in existing["provenance_events"]:
            existing["provenance_events"].append(event)

    def to_record(self) -> dict[str, Any]:
        result = {
            "schema": AVAILABILITY_SCHEMA,
            "trajectory_id": self.trajectory_id,
            "clock_id": self.clock_id,
            "clock_semantics": self.clock_semantics,
            "transport_mode": self.transport_mode,
            "oai_executed": self.transport_mode == OAI_TRANSPORT,
            "source_observations": list(self._source_observations),
            "source_confirmations": list(self._confirmations),
            "install_attempts": list(self._install_attempts),
            "recipient_local_installs": list(self._recipient_local_installs),
            "recipient_map_tracks": list(self._map_tracks),
        }
        result["provenance_sha256"] = canonical_sha256(result)
        return result


def validate_availability_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != AVAILABILITY_SCHEMA:
        raise ValueError("unsupported recipient availability schema")
    _required("trajectory_id", record.get("trajectory_id"))
    clock_id = _required("clock_id", record.get("clock_id"))
    _required("clock_semantics", record.get("clock_semantics"))
    mode = _required("transport_mode", record.get("transport_mode"))
    if mode not in TRANSPORT_MODES:
        raise ValueError("invalid recipient availability transport mode")
    if bool(record.get("oai_executed")) != (mode == OAI_TRANSPORT):
        raise ValueError("transport mode and oai_executed disagree")
    expected_sha = record.get("provenance_sha256")
    body = {key: value for key, value in record.items() if key != "provenance_sha256"}
    if _sha256("provenance_sha256", expected_sha) != canonical_sha256(body):
        raise ValueError("recipient availability provenance hash is inconsistent")
    for attempt in record.get("install_attempts", []):
        if attempt.get("clock_id") != clock_id:
            raise ValueError("install attempt crosses clock domains")
        if attempt.get("transport_mode") != mode:
            raise ValueError("install attempt transport mode drifted")
    for install in record.get("recipient_local_installs", []):
        if install.get("clock_id") != clock_id:
            raise ValueError("recipient local install crosses clock domains")


def _evaluation_class(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"car", "truck", "bus", "vehicle"}:
        return "vehicle"
    if normalized in {"pedestrian", "walker", "person"}:
        return "pedestrian"
    return normalized


def build_recipient_map_target_match(
    *,
    trajectory_id: str,
    install_kind: str,
    install_ref_id: str,
    source_role: str,
    source_track_id: str,
    recipient_map_track_id: str,
    available_at_s: float,
    canonical_map_state: Mapping[str, Any],
    target_truth_state: Mapping[str, Any],
    center_gate_m: float,
) -> dict[str, Any]:
    """Create evaluation-only proof that an installed map track is usable.

    Source-track truth association is insufficient: map association may merge
    that source into the wrong canonical object.  This evidence therefore
    checks the canonical state exposed at the recipient consumer boundary.
    It is never a policy input.
    """

    kind = _required("install_kind", install_kind)
    if kind not in {"helper_install_attempt", "recipient_local_install"}:
        raise ValueError("install_kind is invalid")
    role = _required("source_role", source_role)
    if (kind == "helper_install_attempt" and role != "helper") or (
        kind == "recipient_local_install" and role != "recipient"
    ):
        raise ValueError("install kind and source role disagree")
    available = _finite("available_at_s", available_at_s)
    gate = _finite("center_gate_m", center_gate_m)
    if gate <= 0.0:
        raise ValueError("center_gate_m must be positive")
    map_at = _finite(
        "canonical_map_state.snapshot_at_s", canonical_map_state.get("snapshot_at_s")
    )
    truth_at = _finite(
        "target_truth_state.observed_at_s", target_truth_state.get("observed_at_s")
    )
    if abs(map_at - available) > 1e-9 or abs(truth_at - available) > 1e-9:
        raise ValueError("map/target evidence is not aligned at recipient availability")
    map_class = _evaluation_class(canonical_map_state.get("class_name"))
    truth_class = _evaluation_class(target_truth_state.get("class_name"))
    map_x = _finite("canonical_map_state.x_m", canonical_map_state.get("x_m"))
    map_y = _finite("canonical_map_state.y_m", canonical_map_state.get("y_m"))
    truth_x = _finite("target_truth_state.x_m", target_truth_state.get("x_m"))
    truth_y = _finite("target_truth_state.y_m", target_truth_state.get("y_m"))
    distance = math.hypot(map_x - truth_x, map_y - truth_y)
    class_match = map_class == truth_class
    usable = class_match and distance <= gate + 1e-12
    result = {
        "schema": MAP_TARGET_MATCH_SCHEMA,
        "trajectory_id": _required("trajectory_id", trajectory_id),
        "scope": "evaluation_only_not_policy_state",
        "install_kind": kind,
        "install_ref_id": _required("install_ref_id", install_ref_id),
        "source_role": role,
        "source_track_id": _required("source_track_id", source_track_id),
        "recipient_map_track_id": _required(
            "recipient_map_track_id", recipient_map_track_id
        ),
        "available_at_s": available,
        "canonical_map_state": {
            "class_name": map_class,
            "x_m": map_x,
            "y_m": map_y,
            "snapshot_at_s": map_at,
        },
        "target_truth_state": {
            "class_name": truth_class,
            "x_m": truth_x,
            "y_m": truth_y,
            "observed_at_s": truth_at,
        },
        "center_gate_m": gate,
        "class_match": class_match,
        "center_distance_m": distance,
        "usable_target_match": usable,
    }
    result["match_sha256"] = canonical_sha256(result)
    return result


def validate_recipient_map_target_match(record: Mapping[str, Any]) -> None:
    if record.get("schema") != MAP_TARGET_MATCH_SCHEMA:
        raise ValueError("unsupported recipient-map target-match schema")
    expected = build_recipient_map_target_match(
        trajectory_id=record.get("trajectory_id"),
        install_kind=record.get("install_kind"),
        install_ref_id=record.get("install_ref_id"),
        source_role=record.get("source_role"),
        source_track_id=record.get("source_track_id"),
        recipient_map_track_id=record.get("recipient_map_track_id"),
        available_at_s=record.get("available_at_s"),
        canonical_map_state=record.get("canonical_map_state", {}),
        target_truth_state=record.get("target_truth_state", {}),
        center_gate_m=record.get("center_gate_m"),
    )
    if dict(record) != expected:
        raise ValueError("recipient-map target match is not recomputable")


def build_recipient_available_endpoint(
    provenance: Mapping[str, Any],
    *,
    helper_source_track_id: str | None,
    recipient_source_track_id: str | None,
    recipient_map_track_id: str | None,
    evaluation_horizon_s: float,
    evaluation_recipient_map_target_matches: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the target endpoint post-capture using evaluation-only association."""

    validate_availability_record(provenance)
    horizon = _finite("evaluation_horizon_s", evaluation_horizon_s)
    if horizon <= 0:
        raise ValueError("evaluation_horizon_s must be positive")
    helper_id = None if helper_source_track_id is None else _required(
        "helper_source_track_id", helper_source_track_id
    )
    recipient_id = None if recipient_source_track_id is None else _required(
        "recipient_source_track_id", recipient_source_track_id
    )
    map_id = None if recipient_map_track_id is None else _required(
        "recipient_map_track_id", recipient_map_track_id
    )

    helper_times = sorted(
        float(item["confirmed_at_s"])
        for item in provenance["source_confirmations"]
        if item["source_role"] == "helper" and item["source_track_id"] == helper_id
    )
    match_records = [dict(item) for item in evaluation_recipient_map_target_matches]
    for match in match_records:
        validate_recipient_map_target_match(match)
        if match["trajectory_id"] != provenance["trajectory_id"]:
            raise ValueError("recipient-map target match trajectory drifted")
    usable_refs = {
        (item["install_kind"], item["install_ref_id"])
        for item in match_records
        if item["usable_target_match"]
    }
    all_recipient_installs = [
        item
        for item in provenance["recipient_local_installs"]
        if item["source_track_id"] == recipient_id
    ]
    recipient_installs = sorted(
        (
            item
            for item in all_recipient_installs
            if ("recipient_local_install", item["local_install_id"]) in usable_refs
        ),
        key=lambda item: float(item["available_at_s"]),
    )
    installs = sorted(
        (
            item
            for item in provenance["install_attempts"]
            if item["install_status"] == "accepted"
            and item["source_track_id"] == helper_id
            and item["source_role"] == "helper"
            and item["recipient_map_track_id"] == map_id
            and ("helper_install_attempt", item["attempt_id"]) in usable_refs
        ),
        key=lambda item: float(item["available_at_s"]),
    )
    helper_at = helper_times[0] if helper_times else None
    recipient_confirmation_times = sorted(
        float(item["confirmed_at_s"])
        for item in provenance["source_confirmations"]
        if item["source_role"] == "recipient" and item["source_track_id"] == recipient_id
    )
    recipient_confirmation_at = (
        recipient_confirmation_times[0] if recipient_confirmation_times else None
    )
    recipient_install = recipient_installs[0] if recipient_installs else None
    recipient_at = (
        None
        if recipient_install is None
        else float(recipient_install["available_at_s"])
    )
    install = installs[0] if installs else None
    available_at = None if install is None else float(install["available_at_s"])
    helper_attempts = [
        item
        for item in provenance["install_attempts"]
        if item["source_role"] == "helper" and item["source_track_id"] == helper_id
    ]
    if helper_at is not None and install is None and not helper_attempts:
        raise ValueError(
            "confirmed helper track lacks a recorded publication/install attempt"
        )
    if recipient_confirmation_at is not None and not all_recipient_installs:
        raise ValueError(
            "confirmed recipient-self track lacks consumer install/availability provenance"
        )
    if install is not None:
        if helper_at is None:
            raise ValueError("accepted target install lacks helper confirmation")
        if not helper_at <= float(install["published_at_s"]) <= float(
            install["installed_at_s"]
        ) <= available_at:
            raise ValueError("helper confirmation/publication/install/availability order is invalid")
    all_event_times = [item for item in (helper_at, recipient_at, available_at) if item is not None]
    if all_event_times and horizon + 1e-12 < max(all_event_times):
        raise ValueError("evaluation horizon precedes an endpoint event")

    def event_or_miss(value: float | None) -> dict[str, Any]:
        return (
            {"status": "event", "at_s": value}
            if value is not None
            else {"status": "miss", "censor_at_s": horizon}
        )

    helper_event = event_or_miss(helper_at)
    recipient_confirmation_event = event_or_miss(recipient_confirmation_at)
    recipient_install_event = (
        {
            "status": "event",
            "at_s": recipient_at,
            "local_install_id": recipient_install["local_install_id"],
            "source_track_id": recipient_install["source_track_id"],
            "recipient_map_track_id": recipient_install["recipient_map_track_id"],
            "confirmed_at_s": recipient_install["confirmed_at_s"],
            "installed_at_s": recipient_install["installed_at_s"],
            "available_at_s": recipient_install["available_at_s"],
            "clock_id": recipient_install["clock_id"],
            "consumer_boundary": recipient_install["consumer_boundary"],
        }
        if recipient_at is not None
        else {"status": "censored", "censor_at_s": horizon}
    )
    if install is None:
        install_event: dict[str, Any] = {"status": "miss", "censor_at_s": horizon}
    else:
        install_event = {
            "status": "event",
            "at_s": available_at,
            "contribution_id": install["contribution_id"],
            "source_track_id": install["source_track_id"],
            "recipient_map_track_id": install["recipient_map_track_id"],
            "published_at_s": install["published_at_s"],
            "installed_at_s": install["installed_at_s"],
            "available_at_s": install["available_at_s"],
            "transport_mode": install["transport_mode"],
            "clock_id": install["clock_id"],
        }

    if available_at is not None and recipient_at is not None:
        status = "numeric"
    elif available_at is not None:
        status = "ego_right_censored"
    elif recipient_at is None:
        status = "both_miss"
    else:
        status = "cooperative_miss"
    association_digest = canonical_sha256(
        {
            "helper_source_track_id": helper_id,
            "recipient_source_track_id": recipient_id,
            "recipient_map_track_id": map_id,
            "scope": "evaluation_only_not_policy_state",
            "recipient_map_target_match_sha256": sorted(
                item["match_sha256"] for item in match_records
            ),
        }
    )
    result: dict[str, Any] = {
        "endpoint_name": ENDPOINT_NAME,
        "clock_id": provenance["clock_id"],
        "clock_semantics": provenance["clock_semantics"],
        "transport_mode": provenance["transport_mode"],
        "evaluation_horizon_s": horizon,
        "evaluation_horizon_semantics": "absolute_timestamp_on_clock_id_not_duration",
        "endpoint_status": status,
        "helper_source_confirmation": helper_event,
        "helper_track_recipient_install": install_event,
        "recipient_self_source_confirmation": recipient_confirmation_event,
        "recipient_self_track_recipient_install": recipient_install_event,
        "evaluation_association": {
            "scope": "evaluation_only_not_policy_state",
            "helper_source_track_id": helper_id,
            "recipient_source_track_id": recipient_id,
            "recipient_map_track_id": map_id,
            "recipient_map_target_match_sha256": sorted(
                item["match_sha256"] for item in match_records
            ),
        },
        "evaluation_association_sha256": association_digest,
    }
    if status == "numeric":
        result[ENDPOINT_NAME] = float(recipient_at) - float(available_at)
    elif status == "ego_right_censored":
        result["recipient_available_confirmed_track_margin_lower_bound_s"] = (
            horizon - float(available_at)
        )
    result["evidence_chain_sha256"] = canonical_sha256(
        {
            "provenance_sha256": provenance["provenance_sha256"],
            "evaluation_association_sha256": association_digest,
            "endpoint_without_evidence_sha256": result,
        }
    )
    return result


def _typed_rate(numerator: int, denominator: int) -> dict[str, Any]:
    # Fragmentation/duplicate event rates may exceed one event per exposed
    # source track, unlike proportions such as false-install coverage.
    if numerator < 0 or denominator < 0:
        raise ValueError("invalid metric numerator/denominator")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
        "exposure_status": "typed_zero_exposure" if denominator == 0 else "observed",
    }


def analyze_installed_track_guardrails(
    provenance: Mapping[str, Any],
    *,
    evaluation_truth_match_by_attempt_id: Mapping[str, bool | None] | None = None,
) -> dict[str, Any]:
    """Recompute structural diagnostics without treating non-targets as false installs."""

    validate_availability_record(provenance)
    observations = {
        (item["source_role"], item["source_track_id"], item["observation_sha256"])
        for item in provenance["source_observations"]
    }
    valid_local_install_ids = {
        item["local_install_id"]
        for item in provenance["recipient_local_installs"]
        if (
            "recipient",
            item["source_track_id"],
            item["source_observation_sha256"],
        )
        in observations
    }
    attempts = list(provenance["install_attempts"])
    accepted = [item for item in attempts if item["install_status"] == "accepted"]
    valid_attempt_ids = {
        item["attempt_id"]
        for item in accepted
        if (
            item["source_role"],
            item["source_track_id"],
            item["source_observation_sha256"],
        )
        in observations
    }
    false_install_count = len(accepted) - len(valid_attempt_ids)

    accepted_keys = [
        (item["contribution_id"], item["source_role"], item["source_track_id"])
        for item in accepted
    ]
    duplicate_count = len(accepted_keys) - len(set(accepted_keys))
    source_to_maps: dict[tuple[str, str], set[str]] = {}
    for item in accepted:
        source_to_maps.setdefault(
            (item["source_role"], item["source_track_id"]), set()
        ).add(
            item["recipient_map_track_id"]
        )
    fragmentation_extra = sum(max(0, len(ids) - 1) for ids in source_to_maps.values())

    map_tracks = list(provenance["recipient_map_tracks"])
    unique_map_track_ids = len(
        {item.get("recipient_map_track_id") for item in map_tracks}
    ) == len(map_tracks)
    polluted = 0
    for track in map_tracks:
        valid = False
        for event in track.get("provenance_events", []):
            kind = event.get("provenance_kind")
            ref = event.get("provenance_ref")
            if kind == "installed_source_track" and ref in valid_attempt_ids:
                valid = True
            elif kind == "recipient_local_install" and ref in valid_local_install_ids:
                valid = True
        if not valid:
            polluted += 1

    one_clock = all(item.get("clock_id") == provenance["clock_id"] for item in attempts)
    monotone = all(
        item["install_status"] != "accepted"
        or float(item["published_at_s"])
        <= float(item["installed_at_s"])
        <= float(item["available_at_s"])
        for item in attempts
    )
    unique_attempt_ids = len({item["attempt_id"] for item in attempts}) == len(attempts)
    gates = {
        "every_install_has_valid_contribution_object_source_and_recipient_track_provenance": (
            false_install_count == 0 and duplicate_count == 0
        ),
        "publish_install_available_timestamps_are_monotone_on_one_clock": one_clock and monotone,
        "every_metric_event_is_recomputable_from_immutable_provenance": (
            false_install_count == 0 and unique_attempt_ids and unique_map_track_ids
        ),
        "zero_missing_denominators_or_untyped_zero_exposure_cases": True,
    }
    metrics: dict[str, Any] = {
        "protocol_false_recipient_install_rate": _typed_rate(
            false_install_count, len(attempts)
        ),
        "duplicate_recipient_install_rate": _typed_rate(
            duplicate_count, len(accepted)
        ),
        "source_to_recipient_track_fragmentation_rate": _typed_rate(
            fragmentation_extra, len(source_to_maps)
        ),
        "recipient_map_pollution_rate": _typed_rate(polluted, len(map_tracks)),
    }

    truth = evaluation_truth_match_by_attempt_id or {}
    known_truth = [
        bool(truth[item["attempt_id"]])
        for item in accepted
        if truth.get(item["attempt_id"]) is not None
    ]
    metrics["truth_unmatched_recipient_install_rate"] = _typed_rate(
        sum(not matched for matched in known_truth), len(known_truth)
    )
    result = {
        "schema": GUARDRAIL_SCHEMA,
        "trajectory_id": provenance["trajectory_id"],
        "definitions_status": "frozen_before_exact_16",
        "numeric_threshold_status": "unset_estimation_only",
        "metrics": metrics,
        "structural_gates": gates,
        "structural_pass": all(gates.values()),
        "source_provenance_sha256": provenance["provenance_sha256"],
    }
    result["guardrail_sha256"] = canonical_sha256(result)
    return result


def summarize_policy_audits(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only after each real-row audit has been independently validated."""

    if not records:
        raise ValueError("at least one policy audit is required")
    code_hashes = {str(record["consumer_code_sha256"]) for record in records}
    if len(code_hashes) != 1:
        raise ValueError("policy consumer code changed within the tranche")
    projection_exercises = {
        canonical_sha256(record["projection_exercise"]): record["projection_exercise"]
        for record in records
    }
    if len(projection_exercises) != 1:
        raise ValueError("policy projection exercise changed within the tranche")
    placement = sum(int(record["decision_counts"]["placement"]) for record in records)
    publication = sum(int(record["decision_counts"]["publication"]) for record in records)
    first = records[0]
    return {
        "consumer_enforces_exact_projection": True,
        "consumer_code_sha256": next(iter(code_hashes)),
        "placement_decision_count": placement,
        "publication_decision_count": publication,
        "placement_features": list(first["placement_features"]),
        "publication_features": list(first["publication_features"]),
        "trajectory_audit_count": len(records),
        "projection_exercise": dict(next(iter(projection_exercises.values()))),
        "zero_object_state_trajectory_count": sum(
            bool(record["policy_state_exposure"]["zero_object_state_seen"])
            for record in records
        ),
        "zero_track_policy_state_handling_status": (
            "future_controller_blocker_missingness_contract_not_frozen"
        ),
        "environment_readiness_claimed": False,
        "audit_set_sha256": canonical_sha256(
            sorted(str(record["audit_sha256"]) for record in records)
        ),
    }


def aggregate_guardrail_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one installed-track guardrail report is required")
    metric_names = tuple(reports[0]["metrics"])
    metrics: dict[str, Any] = {}
    for name in metric_names:
        numerator = sum(int(report["metrics"][name]["numerator"]) for report in reports)
        denominator = sum(int(report["metrics"][name]["denominator"]) for report in reports)
        metrics[name] = _typed_rate(numerator, denominator)
    gate_names = tuple(reports[0]["structural_gates"])
    gates = {
        name: all(bool(report["structural_gates"][name]) for report in reports)
        for name in gate_names
    }
    result = {
        "schema": f"{GUARDRAIL_SCHEMA}.aggregate",
        "trajectory_count": len(reports),
        "definitions_status": "frozen_before_exact_16",
        "numeric_threshold_status": "unset_estimation_only",
        "metrics": metrics,
        "structural_gates": gates,
        "structural_pass": all(gates.values()),
        "trajectory_guardrail_sha256": sorted(
            str(report["guardrail_sha256"]) for report in reports
        ),
    }
    result["aggregate_sha256"] = canonical_sha256(result)
    return result
