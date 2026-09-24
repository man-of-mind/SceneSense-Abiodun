"""Bounded Run-4 bridge from fit-scene quality evidence to tensor payloads.

This module is intentionally an adapter, not an environment.  It contains no
latency or radio model, performs no file I/O, and never chooses a scene.  The
caller supplies one hidden ``fit``-scene selection, one catalog-reconciled
executed action, and already loaded/hash-bound providers.

This adapter does **not** choose fit scenes.  The future training runner must
draw from the registered fit inventory with its recorded sampling weights and
then supply that hidden selection; manually manufacturing convenient scene
selections would invalidate the training distribution even when the scene ID
exists in the surface.

The reward-request tensor uses ``Q_perc`` and payload from the existing
same-scene empirical surface.  Additional tensors use the existing marginal
train-only held-payload provider and can never request a reward.  Hidden scene
identities and endpoint provenance remain in environment-only records; the
only scene values exposed for policy-state construction are camera SI and
radar P40.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import held_payload
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract
from rl_agent.splitfusion_hybrid_sac_v1.empirical_quality_surface import (
    EXACT_GRID_ROW_EVIDENCE,
    MODELED_SAME_FRAME_EVIDENCE,
    NO_RESIDUAL_INTERVAL_STATUS,
    EmpiricalQualitySurface,
    EndpointEvidence,
    HiddenSurfaceRecord,
    PolicySceneView,
    PolicySurfaceView,
    QualityComponent,
    SurfaceBinding,
    SurfaceQueryResult,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    canonical_sha256,
)

__all__ = [
    "QualityAdapterError",
    "QualityAdapterBindingError",
    "FitSceneRejected",
    "ActionJoinError",
    "QualityUnavailable",
    "FitSceneSelectionV1",
    "QualityPayloadAdapterBindingV1",
    "RewardTensorEvidenceV1",
    "RewardTensorResultV1",
    "HeldTensorResultV1",
    "surface_binding_sha256",
    "Run4QualityPayloadAdapterV1",
]


class QualityAdapterError(ValueError):
    """Base class for quality/payload bridge failures."""


class QualityAdapterBindingError(QualityAdapterError):
    """The supplied providers or evidence hashes do not bind exactly."""


class FitSceneRejected(QualityAdapterError):
    """A hidden scene is foreign, held/test, or internally inconsistent."""


class ActionJoinError(QualityAdapterError):
    """The surface or held result does not describe the executed action."""


class QualityUnavailable(QualityAdapterError):
    """Direct Q_perc is invalid or unavailable for this exact query."""


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualityAdapterBindingError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _identity(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise FitSceneRejected(f"{name} must be a nonempty canonical string")
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise QualityAdapterError(f"{name} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise QualityAdapterError(f"{name} must be >= {minimum}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise QualityAdapterError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise QualityAdapterError(f"{name} must be finite")
    return result


def surface_binding_sha256(binding: SurfaceBinding) -> str:
    """Canonical digest of every field in the existing surface binding."""

    if type(binding) is not SurfaceBinding:
        raise QualityAdapterBindingError("binding must be exactly SurfaceBinding")
    document: Dict[str, str] = {}
    for item in fields(SurfaceBinding):
        document[item.name] = _digest(
            getattr(binding, item.name), f"surface_binding.{item.name}"
        )
    return canonical_sha256(
        {
            "record": "splitfusion_empirical_quality_surface_binding_bridge_v1",
            "value": document,
        }
    )


@dataclass(frozen=True, slots=True)
class FitSceneSelectionV1:
    """Environment-only identity of one selected fit scene.

    ``camera_si`` and ``radar_p40`` are duplicated here solely to close the
    scene join when the action-dependent quality query is made.  They are the
    only fields returned by :meth:`policy_scene`; all identifiers stay hidden.
    """

    sample_id: str
    episode_id: str
    frame_id: int
    grid_split: str
    selection_rank_within_fit: int
    inclusion_probability: float
    sampling_weight: float
    camera_si: float
    radar_p40: float
    surface_binding_sha256: str

    def __post_init__(self) -> None:
        _identity(self.sample_id, "sample_id")
        _identity(self.episode_id, "episode_id")
        object.__setattr__(self, "frame_id", _exact_int(self.frame_id, "frame_id"))
        if self.grid_split != "fit":
            raise FitSceneRejected("reward scenes must come from exactly the fit split")
        object.__setattr__(
            self,
            "selection_rank_within_fit",
            _exact_int(
                self.selection_rank_within_fit, "selection_rank_within_fit"
            ),
        )
        probability = _finite(self.inclusion_probability, "inclusion_probability")
        weight = _finite(self.sampling_weight, "sampling_weight")
        if not 0.0 < probability <= 1.0 or weight <= 0.0:
            raise FitSceneRejected("sampling design values must be positive")
        if not math.isclose(
            probability * weight, 1.0, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise FitSceneRejected(
                "sampling_weight must equal inverse inclusion_probability"
            )
        camera_si = _finite(self.camera_si, "camera_si")
        radar_p40 = _finite(self.radar_p40, "radar_p40")
        if camera_si < 0.0 or not 0.0 <= radar_p40 <= 1.0:
            raise FitSceneRejected("scene values escaped their registered domains")
        _digest(self.surface_binding_sha256, "surface_binding_sha256")
        object.__setattr__(self, "inclusion_probability", probability)
        object.__setattr__(self, "sampling_weight", weight)
        object.__setattr__(self, "camera_si", camera_si)
        object.__setattr__(self, "radar_p40", radar_p40)

    @classmethod
    def from_query(
        cls, query: SurfaceQueryResult, *, surface_binding_sha256: str
    ) -> "FitSceneSelectionV1":
        """Capture an action-independent scene selection from a fit query.

        The adapter re-queries the selected sample for the executed action and
        checks every field again; this constructor alone grants no authority.
        """

        if type(query) is not SurfaceQueryResult:
            raise FitSceneRejected("query must be exactly SurfaceQueryResult")
        if type(query.hidden) is not HiddenSurfaceRecord:
            raise FitSceneRejected("query carries a foreign hidden record")
        if type(query.policy) is not PolicySurfaceView:
            raise FitSceneRejected("query carries a foreign policy record")
        hidden = query.hidden
        scene = query.policy.scene
        if type(scene) is not PolicySceneView:
            raise FitSceneRejected("query carries a foreign policy scene")
        return cls(
            sample_id=hidden.sample_id,
            episode_id=hidden.episode_id,
            frame_id=hidden.frame_id,
            grid_split=hidden.grid_split,
            selection_rank_within_fit=hidden.selection_rank_within_split,
            inclusion_probability=hidden.inclusion_probability,
            sampling_weight=hidden.sampling_weight,
            camera_si=scene.camera_si,
            radar_p40=scene.radar_p40,
            surface_binding_sha256=surface_binding_sha256,
        )

    def hidden_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "grid_split": self.grid_split,
            "inclusion_probability": self.inclusion_probability,
            "radar_p40": self.radar_p40,
            "camera_si": self.camera_si,
            "sample_id": self.sample_id,
            "sampling_weight": self.sampling_weight,
            "selection_rank_within_fit": self.selection_rank_within_fit,
            "surface_binding_sha256": self.surface_binding_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_hidden_fit_scene_selection_v1",
                "value": self.hidden_dict(),
            }
        )

    def policy_scene(self) -> PolicySceneView:
        """Return the complete policy-visible scene view: SI and P40 only."""

        return PolicySceneView(camera_si=self.camera_si, radar_p40=self.radar_p40)


@dataclass(frozen=True, slots=True)
class QualityPayloadAdapterBindingV1:
    surface_binding_sha256: str
    surface_database_sha256: str
    surface_selection_file_sha256: str
    corrected_p40_binding_sha256: str
    corrected_p40_snapshot_sha256: str
    held_provider_binding_sha256: str
    held_inventory_sha256: str

    def __post_init__(self) -> None:
        for item in fields(self):
            _digest(getattr(self, item.name), item.name)

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_quality_payload_adapter_binding_v1",
                "value": {
                    item.name: getattr(self, item.name) for item in fields(self)
                },
            }
        )


def _endpoint_dict(endpoint: EndpointEvidence) -> Dict[str, Any]:
    if type(endpoint) is not EndpointEvidence:
        raise QualityAdapterError("surface endpoint has a foreign type")
    q_e4 = _exact_int(endpoint.q_e4, "endpoint.q_e4")
    row_key = _digest(endpoint.row_key_sha256, "endpoint.row_key_sha256")
    row = _digest(endpoint.row_sha256, "endpoint.row_sha256")
    datagrams = _exact_int(endpoint.datagram_count, "endpoint.datagram_count")
    if type(endpoint.quality_valid) is not bool:
        raise QualityAdapterError("endpoint.quality_valid must be bool")
    if type(endpoint.quality_status) is not str or not endpoint.quality_status:
        raise QualityAdapterError("endpoint.quality_status must be nonempty")
    return {
        "datagram_count": datagrams,
        "q_e4": q_e4,
        "quality_status": endpoint.quality_status,
        "quality_valid": endpoint.quality_valid,
        "row_key_sha256": row_key,
        "row_sha256": row,
    }


@dataclass(frozen=True, slots=True)
class RewardTensorEvidenceV1:
    adapter_binding_sha256: str
    fit_selection_sha256: str
    action_sha256: str
    mode_id: int
    q_e4: int
    q_perc: float
    offered_payload_bytes: int | float
    payload_evidence_class: run4_contract.PayloadEvidenceClass
    policy_scene_sha256: str
    sample_id: str
    episode_id: str
    frame_id: int
    surface_evidence_status: str
    q_perc_status: str
    endpoints: Tuple[EndpointEvidence, ...]

    def __post_init__(self) -> None:
        _digest(self.adapter_binding_sha256, "adapter_binding_sha256")
        _digest(self.fit_selection_sha256, "fit_selection_sha256")
        _digest(self.action_sha256, "action_sha256")
        object.__setattr__(self, "mode_id", _exact_int(self.mode_id, "mode_id"))
        object.__setattr__(self, "q_e4", _exact_int(self.q_e4, "q_e4"))
        q_perc = _finite(self.q_perc, "q_perc")
        payload = _finite(self.offered_payload_bytes, "offered_payload_bytes")
        if not 0.0 <= q_perc <= 1.0:
            raise QualityUnavailable("evidence Q_perc escaped [0,1]")
        if payload <= 0.0:
            raise QualityAdapterError("evidence payload must be positive")
        if not isinstance(
            self.payload_evidence_class, run4_contract.PayloadEvidenceClass
        ):
            raise QualityAdapterError(
                "payload_evidence_class must be PayloadEvidenceClass"
            )
        if (
            self.payload_evidence_class
            is run4_contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            and type(self.offered_payload_bytes) is not int
        ):
            raise QualityAdapterError("measured evidence payload must be an int")
        _digest(self.policy_scene_sha256, "policy_scene_sha256")
        _identity(self.sample_id, "sample_id")
        _identity(self.episode_id, "episode_id")
        object.__setattr__(self, "frame_id", _exact_int(self.frame_id, "frame_id"))
        if self.surface_evidence_status not in (
            EXACT_GRID_ROW_EVIDENCE,
            MODELED_SAME_FRAME_EVIDENCE,
        ):
            raise QualityAdapterError("unsupported surface evidence status")
        if type(self.q_perc_status) is not str or not self.q_perc_status:
            raise QualityAdapterError("q_perc_status must be nonempty")
        if type(self.endpoints) is not tuple:
            raise QualityAdapterError("endpoints must be an exact tuple")
        expected_count = (
            1 if self.surface_evidence_status == EXACT_GRID_ROW_EVIDENCE else 2
        )
        if len(self.endpoints) != expected_count:
            raise QualityAdapterError("endpoint count contradicts evidence status")
        for endpoint in self.endpoints:
            _endpoint_dict(endpoint)
        object.__setattr__(self, "q_perc", q_perc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_sha256": self.action_sha256,
            "adapter_binding_sha256": self.adapter_binding_sha256,
            "endpoints": [_endpoint_dict(item) for item in self.endpoints],
            "episode_id": self.episode_id,
            "fit_selection_sha256": self.fit_selection_sha256,
            "frame_id": self.frame_id,
            "mode_id": self.mode_id,
            "offered_payload_bytes": self.offered_payload_bytes,
            "payload_evidence_class": self.payload_evidence_class.value,
            "policy_scene_sha256": self.policy_scene_sha256,
            "q_e4": self.q_e4,
            "q_perc": self.q_perc,
            "q_perc_status": self.q_perc_status,
            "sample_id": self.sample_id,
            "surface_evidence_status": self.surface_evidence_status,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_reward_tensor_evidence_v1",
                "value": self.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class RewardTensorResultV1:
    """Reward-request tensor plus direct quality and policy-safe scene state."""

    action: ExecutedActionIdentity
    policy_scene: PolicySceneView
    q_perc: float
    tensor: run4_contract.HoldTensorV1
    evidence: RewardTensorEvidenceV1

    def __post_init__(self) -> None:
        if type(self.action) is not ExecutedActionIdentity:
            raise ActionJoinError("action must be exactly ExecutedActionIdentity")
        self.action.require_reconciled()
        if type(self.policy_scene) is not PolicySceneView:
            raise FitSceneRejected("policy_scene must be exactly PolicySceneView")
        quality = _finite(self.q_perc, "q_perc")
        if not 0.0 <= quality <= 1.0:
            raise QualityUnavailable("q_perc escaped [0,1]")
        if type(self.tensor) is not run4_contract.HoldTensorV1:
            raise QualityAdapterError("tensor must be exactly HoldTensorV1")
        if not self.tensor.reward_requested:
            raise QualityAdapterError("reward tensor must request a reward")
        if type(self.evidence) is not RewardTensorEvidenceV1:
            raise QualityAdapterError("evidence must be RewardTensorEvidenceV1")
        if self.tensor.payload_provenance_sha256 != self.evidence.canonical_sha256:
            raise QualityAdapterError("tensor/evidence provenance mismatch")
        if self.evidence.action_sha256 != self.action.canonical_sha256():
            raise ActionJoinError("evidence/action identity mismatch")
        if (self.evidence.mode_id, self.evidence.q_e4) != (
            self.action.mode_id,
            self.action.q_e4,
        ):
            raise ActionJoinError("evidence/action coordinate mismatch")
        if self.evidence.q_perc != quality:
            raise QualityUnavailable("result/evidence Q_perc mismatch")
        if (
            self.evidence.offered_payload_bytes
            != self.tensor.offered_payload_bytes
            or self.evidence.payload_evidence_class
            is not self.tensor.payload_evidence_class
        ):
            raise QualityAdapterError("result/evidence payload mismatch")
        scene_sha = canonical_sha256(
            {
                "record": "splitfusion_run4_policy_scene_si_p40_v1",
                "value": self.policy_scene.to_dict(),
            }
        )
        if self.evidence.policy_scene_sha256 != scene_sha:
            raise FitSceneRejected("result/evidence policy-scene mismatch")
        object.__setattr__(self, "q_perc", quality)

    def policy_scene_features(self) -> Tuple[float, float]:
        """Return only ``(camera_si, radar_p40)``; no hidden ID can leak."""

        return (self.policy_scene.camera_si, self.policy_scene.radar_p40)


@dataclass(frozen=True, slots=True)
class HeldTensorResultV1:
    """One marginal held tensor.  It categorically carries no reward."""

    action: ExecutedActionIdentity
    tensor: run4_contract.HoldTensorV1
    estimate: held_payload.HeldPayloadEstimateV1

    def __post_init__(self) -> None:
        if type(self.action) is not ExecutedActionIdentity:
            raise ActionJoinError("action must be exactly ExecutedActionIdentity")
        self.action.require_reconciled()
        if type(self.tensor) is not run4_contract.HoldTensorV1:
            raise QualityAdapterError("tensor must be exactly HoldTensorV1")
        if self.tensor.reward_requested:
            raise QualityAdapterError("held tensors must never request a reward")
        if type(self.estimate) is not held_payload.HeldPayloadEstimateV1:
            raise QualityAdapterError("estimate must be HeldPayloadEstimateV1")
        if (self.estimate.mode_id, self.estimate.q_e4) != (
            self.action.mode_id,
            self.action.q_e4,
        ):
            raise ActionJoinError("held estimate/action identity mismatch")
        if self.tensor.payload_provenance_sha256 != self.estimate.canonical_sha256:
            raise QualityAdapterError("held tensor/estimate provenance mismatch")
        if self.tensor.offered_payload_bytes != self.estimate.total_transmitted_bytes:
            raise QualityAdapterError("held tensor/estimate payload mismatch")
        expected_class = (
            run4_contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            if self.estimate.interpolation_status == held_payload.EXACT_NODE_STATUS
            else run4_contract.PayloadEvidenceClass.MODELED_SAME_SCENE_INTERPOLATION
        )
        if self.tensor.payload_evidence_class is not expected_class:
            raise QualityAdapterError("held tensor evidence class mismatch")
        if self.estimate.evidence_label != held_payload.EVIDENCE_LABEL:
            raise QualityAdapterError("held estimate evidence label mismatch")


class Run4QualityPayloadAdapterV1:
    """Join exact fit quality and held payloads without loading evidence."""

    def __init__(
        self,
        *,
        surface: EmpiricalQualitySurface,
        held_provider: held_payload.HeldPayloadProviderV1,
        expected_surface_binding_sha256: str,
        expected_held_provider_binding_sha256: str,
    ) -> None:
        if type(surface) is not EmpiricalQualitySurface:
            raise QualityAdapterBindingError(
                "surface must be exactly EmpiricalQualitySurface"
            )
        if type(held_provider) is not held_payload.HeldPayloadProviderV1:
            raise QualityAdapterBindingError(
                "held_provider must be exactly HeldPayloadProviderV1"
            )
        expected_surface = _digest(
            expected_surface_binding_sha256, "expected_surface_binding_sha256"
        )
        expected_held = _digest(
            expected_held_provider_binding_sha256,
            "expected_held_provider_binding_sha256",
        )
        observed_surface = surface_binding_sha256(surface.binding)
        if observed_surface != expected_surface:
            raise QualityAdapterBindingError("surface binding digest mismatch")
        if held_provider.binding_sha256 != expected_held:
            raise QualityAdapterBindingError("held-provider binding digest mismatch")
        if held_provider.source_database_sha256 != surface.binding.database_file_sha256:
            raise QualityAdapterBindingError(
                "held and reward payloads do not share the quality database"
            )
        if held_provider.source_selection_sha256 != surface.binding.selection_file_sha256:
            raise QualityAdapterBindingError(
                "held and reward payloads do not share the selection file"
            )
        self._surface = surface
        self._held_provider = held_provider
        self.binding = QualityPayloadAdapterBindingV1(
            surface_binding_sha256=observed_surface,
            surface_database_sha256=surface.binding.database_file_sha256,
            surface_selection_file_sha256=surface.binding.selection_file_sha256,
            corrected_p40_binding_sha256=surface.binding.corrected_p40_binding_sha256,
            corrected_p40_snapshot_sha256=surface.binding.corrected_p40_snapshot_sha256,
            held_provider_binding_sha256=held_provider.binding_sha256,
            held_inventory_sha256=held_provider.inventory_sha256,
        )

    def _revalidate_bindings(self) -> None:
        current_surface = surface_binding_sha256(self._surface.binding)
        if current_surface != self.binding.surface_binding_sha256:
            raise QualityAdapterBindingError("surface binding changed after construction")
        provider = self._held_provider
        if (
            provider.binding_sha256 != self.binding.held_provider_binding_sha256
            or provider.inventory_sha256 != self.binding.held_inventory_sha256
            or provider.source_database_sha256
            != self.binding.surface_database_sha256
            or provider.source_selection_sha256
            != self.binding.surface_selection_file_sha256
        ):
            raise QualityAdapterBindingError(
                "held-provider binding changed after construction"
            )

    @staticmethod
    def _action(action: ExecutedActionIdentity) -> ExecutedActionIdentity:
        if type(action) is not ExecutedActionIdentity:
            raise ActionJoinError("action must be exactly ExecutedActionIdentity")
        action.require_reconciled()
        return action

    @staticmethod
    def _join_selection(
        selection: FitSceneSelectionV1, result: SurfaceQueryResult
    ) -> None:
        if type(selection) is not FitSceneSelectionV1:
            raise FitSceneRejected("selection must be FitSceneSelectionV1")
        if type(result) is not SurfaceQueryResult:
            raise FitSceneRejected("surface returned a foreign query result")
        hidden = result.hidden
        scene = result.policy.scene
        observed = (
            hidden.sample_id,
            hidden.episode_id,
            hidden.frame_id,
            hidden.grid_split,
            hidden.selection_rank_within_split,
            hidden.inclusion_probability,
            hidden.sampling_weight,
            scene.camera_si,
            scene.radar_p40,
        )
        expected = (
            selection.sample_id,
            selection.episode_id,
            selection.frame_id,
            selection.grid_split,
            selection.selection_rank_within_fit,
            selection.inclusion_probability,
            selection.sampling_weight,
            selection.camera_si,
            selection.radar_p40,
        )
        if observed != expected or hidden.grid_split != "fit":
            raise FitSceneRejected("surface result does not match the selected fit scene")

    def reward_tensor(
        self,
        *,
        selection: FitSceneSelectionV1,
        action: ExecutedActionIdentity,
        tensor_seq: int,
    ) -> RewardTensorResultV1:
        """Evaluate direct Q_perc and payload for one selected fit scene."""

        self._revalidate_bindings()
        checked_action = self._action(action)
        if type(selection) is not FitSceneSelectionV1:
            raise FitSceneRejected("selection must be FitSceneSelectionV1")
        if selection.surface_binding_sha256 != self.binding.surface_binding_sha256:
            raise QualityAdapterBindingError("selection/surface binding mismatch")
        sequence = _exact_int(tensor_seq, "tensor_seq")
        result = self._surface.query_fit_q_e4(
            selection.sample_id, checked_action.mode_id, checked_action.q_e4
        )
        self._join_selection(selection, result)
        policy = result.policy
        if (
            policy.mode_id != checked_action.mode_id
            or policy.q_e4 != checked_action.q_e4
            or policy.family != checked_action.family
            or policy.quantizer != checked_action.quantizer
        ):
            raise ActionJoinError("surface result does not match the executed action")
        if policy.residual_interval_status != NO_RESIDUAL_INTERVAL_STATUS:
            raise QualityAdapterError("surface residual-interval semantics drifted")
        component = policy.component("q_perc")
        if type(component) is not QualityComponent:
            raise QualityUnavailable("Q_perc component has a foreign type")
        if not component.valid or component.value is None:
            raise QualityUnavailable("Q_perc is undefined; no reward may be issued")
        q_perc = _finite(component.value, "q_perc")
        if not 0.0 <= q_perc <= 1.0:
            raise QualityUnavailable("Q_perc escaped [0,1]")
        if policy.evidence_status == EXACT_GRID_ROW_EVIDENCE:
            evidence_class = run4_contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            if type(policy.payload.total_transmitted_bytes) is not int:
                raise QualityAdapterError("exact payload lost its integer identity")
        elif policy.evidence_status == MODELED_SAME_FRAME_EVIDENCE:
            evidence_class = (
                run4_contract.PayloadEvidenceClass.MODELED_SAME_SCENE_INTERPOLATION
            )
        else:
            raise QualityAdapterError("unsupported surface evidence status")
        payload = policy.payload.total_transmitted_bytes
        payload_value = _finite(payload, "total_transmitted_bytes")
        if payload_value <= 0.0:
            raise QualityAdapterError("reward payload must be positive")
        endpoints = result.hidden.endpoint_evidence
        if policy.evidence_status == EXACT_GRID_ROW_EVIDENCE:
            if len(endpoints) != 1 or endpoints[0].q_e4 != checked_action.q_e4:
                raise ActionJoinError("exact endpoint does not match executed q_e4")
        elif not (
            len(endpoints) == 2
            and endpoints[0].q_e4 < checked_action.q_e4 < endpoints[1].q_e4
        ):
            raise ActionJoinError("modeled endpoints do not bracket executed q_e4")
        if not all(endpoint.quality_valid for endpoint in endpoints):
            raise QualityUnavailable("Q_perc endpoints are not quality-valid")
        policy_scene = selection.policy_scene()
        policy_scene_sha = canonical_sha256(
            {
                "record": "splitfusion_run4_policy_scene_si_p40_v1",
                "value": policy_scene.to_dict(),
            }
        )
        evidence = RewardTensorEvidenceV1(
            adapter_binding_sha256=self.binding.canonical_sha256,
            fit_selection_sha256=selection.canonical_sha256,
            action_sha256=checked_action.canonical_sha256(),
            mode_id=checked_action.mode_id,
            q_e4=checked_action.q_e4,
            q_perc=q_perc,
            offered_payload_bytes=payload,
            payload_evidence_class=evidence_class,
            policy_scene_sha256=policy_scene_sha,
            sample_id=result.hidden.sample_id,
            episode_id=result.hidden.episode_id,
            frame_id=result.hidden.frame_id,
            surface_evidence_status=policy.evidence_status,
            q_perc_status=component.status,
            endpoints=endpoints,
        )
        tensor = run4_contract.HoldTensorV1(
            tensor_seq=sequence,
            offered_payload_bytes=payload,
            payload_evidence_class=evidence_class,
            payload_provenance_sha256=evidence.canonical_sha256,
            reward_requested=True,
        )
        return RewardTensorResultV1(
            action=checked_action,
            policy_scene=policy_scene,
            q_perc=q_perc,
            tensor=tensor,
            evidence=evidence,
        )

    def held_tensor(
        self,
        *,
        counter: held_payload.HeldSelectionCounterV1,
        rng: held_payload.RandomLike,
        action: ExecutedActionIdentity,
        tensor_seq: int,
    ) -> HeldTensorResultV1:
        """Build one train-only marginal held tensor with no reward flag."""

        self._revalidate_bindings()
        checked_action = self._action(action)
        sequence = _exact_int(tensor_seq, "tensor_seq")
        estimate = self._held_provider.evaluate(
            counter=counter,
            rng=rng,
            mode_id=checked_action.mode_id,
            q_e4=checked_action.q_e4,
        )
        if (
            estimate.provider_binding_sha256
            != self.binding.held_provider_binding_sha256
            or estimate.source_database_sha256
            != self.binding.surface_database_sha256
            or estimate.source_selection_sha256
            != self.binding.surface_selection_file_sha256
        ):
            raise QualityAdapterBindingError("held estimate binding mismatch")
        if (estimate.mode_id, estimate.q_e4) != (
            checked_action.mode_id,
            checked_action.q_e4,
        ):
            raise ActionJoinError("held estimate does not match the executed action")
        if estimate.interpolation_status == held_payload.EXACT_NODE_STATUS:
            evidence_class = run4_contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            if type(estimate.total_transmitted_bytes) is not int:
                raise QualityAdapterError("exact held payload lost integer identity")
        elif estimate.interpolation_status == held_payload.INTERPOLATED_STATUS:
            evidence_class = (
                run4_contract.PayloadEvidenceClass.MODELED_SAME_SCENE_INTERPOLATION
            )
        else:
            raise QualityAdapterError("unsupported held interpolation status")
        tensor = run4_contract.HoldTensorV1(
            tensor_seq=sequence,
            offered_payload_bytes=estimate.total_transmitted_bytes,
            payload_evidence_class=evidence_class,
            payload_provenance_sha256=estimate.canonical_sha256,
            reward_requested=False,
        )
        return HeldTensorResultV1(
            action=checked_action, tensor=tensor, estimate=estimate
        )
