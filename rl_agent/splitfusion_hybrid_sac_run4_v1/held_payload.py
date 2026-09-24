"""Pure, train-only marginal payload provider for Run-4 held frames.

This module deliberately does not load the quality database, selection
manifest, or any other evidence.  A caller must supply an already parsed,
hash-bound inventory of train-only same-scene payload curves.  Scene selection
uses only a caller-owned local RNG and an explicit counter identity.

The result is a marginal byte-load model for a virtual held transmission while
one semi-Markov ticket remains outstanding.  It does not emit a policy step or
reward, is not temporal evidence, and is always labelled as such.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Protocol, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    Q_E4_MAX,
    Q_E4_MIN,
)
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.contract import (
    MODE_COUNT,
    Q_E4_GRID,
)

EVIDENCE_LABEL = "MODELED_EMPIRICAL_MARGINAL_HELD_PAYLOAD_NOT_TEMPORAL"
EXACT_NODE_STATUS = "EXACT_SAME_SCENE_Q_NODE"
INTERPOLATED_STATUS = (
    "MODELED_PIECEWISE_LINEAR_FROM_EXACT_SAME_SCENE_ENDPOINTS"
)
WEIGHTING_POLICY = "INVERSE_INCLUSION_PROBABILITY_SAMPLING_WEIGHT"
TRAIN_SPLIT = "train"
SUPPORTED_MODE_IDS = tuple(range(MODE_COUNT))


class HeldPayloadError(ValueError):
    """Base class for held-payload contract failures."""


class SourceRejected(HeldPayloadError):
    """A curve is not train-only or has unsupported/missing evidence."""


class BindingMismatch(HeldPayloadError):
    """Caller-pinned evidence bindings do not close exactly."""


class UnsupportedAction(HeldPayloadError):
    """The executed mode or q is outside the registered support."""


class IdentifierConflict(HeldPayloadError):
    """One identity names conflicting evidence or held-frame requests."""


class LocalRngRequired(HeldPayloadError):
    """The caller did not provide a usable local RNG stream."""


class RandomLike(Protocol):
    def random(self) -> float: ...


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise HeldPayloadError("value is not canonical-JSON encodable") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _digest(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BindingMismatch(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identity(value: Any, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise IdentifierConflict(f"{field} must be a nonempty canonical string")
    return value


def _exact_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise HeldPayloadError(f"{field} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise HeldPayloadError(f"{field} must be >= {minimum}")
    return result


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise HeldPayloadError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise HeldPayloadError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PayloadNodeV1:
    """One exact same-scene q node from the caller-bound source."""

    q_e4: int
    total_transmitted_bytes: int
    source_row_sha256: str

    def __post_init__(self) -> None:
        q_e4 = _exact_int(self.q_e4, "q_e4")
        if not Q_E4_MIN <= q_e4 <= Q_E4_MAX:
            raise UnsupportedAction(
                f"q_e4 {q_e4} is outside [{Q_E4_MIN}, {Q_E4_MAX}]"
            )
        payload = _exact_int(
            self.total_transmitted_bytes, "total_transmitted_bytes", minimum=1
        )
        _digest(self.source_row_sha256, "source_row_sha256")
        object.__setattr__(self, "q_e4", q_e4)
        object.__setattr__(self, "total_transmitted_bytes", payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q_e4": self.q_e4,
            "source_row_sha256": self.source_row_sha256,
            "total_transmitted_bytes": self.total_transmitted_bytes,
        }


@dataclass(frozen=True, slots=True)
class ScenePayloadCurveV1:
    """All exact q nodes for one train scene and one discrete mode."""

    sample_id: str
    episode_id: str
    frame_id: int
    selection_rank_within_train: int
    source_split: str
    inclusion_probability: float
    sampling_weight: float
    mode_id: int
    scene_source_sha256: str
    source_selection_sha256: str
    source_database_sha256: str
    nodes: Tuple[PayloadNodeV1, ...]

    def __post_init__(self) -> None:
        _identity(self.sample_id, "sample_id")
        _identity(self.episode_id, "episode_id")
        frame_id = _exact_int(self.frame_id, "frame_id")
        rank = _exact_int(
            self.selection_rank_within_train, "selection_rank_within_train"
        )
        if type(self.source_split) is not str or self.source_split != TRAIN_SPLIT:
            raise SourceRejected(
                f"source_split must be exactly {TRAIN_SPLIT!r}; got {self.source_split!r}"
            )
        probability = _finite(self.inclusion_probability, "inclusion_probability")
        weight = _finite(self.sampling_weight, "sampling_weight")
        if not 0.0 < probability <= 1.0 or weight <= 0.0:
            raise SourceRejected("sampling design values must be positive and finite")
        if not math.isclose(
            probability * weight, 1.0, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise SourceRejected(
                "sampling_weight must equal inverse inclusion_probability"
            )
        mode_id = _exact_int(self.mode_id, "mode_id")
        if mode_id not in SUPPORTED_MODE_IDS:
            raise UnsupportedAction(f"unsupported mode_id {mode_id}")
        _digest(self.scene_source_sha256, "scene_source_sha256")
        _digest(self.source_selection_sha256, "source_selection_sha256")
        _digest(self.source_database_sha256, "source_database_sha256")
        if type(self.nodes) is not tuple or any(
            type(node) is not PayloadNodeV1 for node in self.nodes
        ):
            raise SourceRejected("nodes must be an exact tuple of PayloadNodeV1")
        if tuple(node.q_e4 for node in self.nodes) != tuple(Q_E4_GRID):
            raise SourceRejected(
                "same-scene curve does not contain the complete registered q grid"
            )
        payloads = tuple(node.total_transmitted_bytes for node in self.nodes)
        if any(lower <= upper for lower, upper in zip(payloads, payloads[1:])):
            raise SourceRejected(
                "same-scene payload must be strictly decreasing across q nodes"
            )
        if len({node.source_row_sha256 for node in self.nodes}) != len(self.nodes):
            raise IdentifierConflict("curve reuses a source row digest")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "selection_rank_within_train", rank)
        object.__setattr__(self, "inclusion_probability", probability)
        object.__setattr__(self, "sampling_weight", weight)
        object.__setattr__(self, "mode_id", mode_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "inclusion_probability": self.inclusion_probability,
            "mode_id": self.mode_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "sample_id": self.sample_id,
            "sampling_weight": self.sampling_weight,
            "scene_source_sha256": self.scene_source_sha256,
            "selection_rank_within_train": self.selection_rank_within_train,
            "source_database_sha256": self.source_database_sha256,
            "source_selection_sha256": self.source_selection_sha256,
            "source_split": self.source_split,
        }

    @property
    def curve_sha256(self) -> str:
        return _sha256(
            {
                "record": "splitfusion_run4_scene_payload_curve_v1",
                "value": self.to_dict(),
            }
        )


def payload_curve_inventory_sha256(
    curves: Iterable[ScenePayloadCurveV1],
) -> str:
    """Return the canonical inventory digest without reading any evidence."""

    material = tuple(curves)
    if not material or any(type(curve) is not ScenePayloadCurveV1 for curve in material):
        raise SourceRejected("curve inventory must contain ScenePayloadCurveV1 records")
    ordered = sorted(material, key=lambda item: (item.sample_id, item.mode_id))
    return _sha256(
        {
            "curves": [curve.to_dict() for curve in ordered],
            "evidence_label": EVIDENCE_LABEL,
            "q_e4_grid": list(Q_E4_GRID),
            "record": "splitfusion_run4_held_payload_curve_inventory_v1",
            "supported_mode_ids": list(SUPPORTED_MODE_IDS),
            "weighting_policy": WEIGHTING_POLICY,
        }
    )


@dataclass(frozen=True, slots=True)
class HeldSelectionCounterV1:
    """Caller-owned identity for exactly one held-frame marginal draw."""

    session_id: str
    decision_seq: int
    held_ordinal: int
    rng_stream_id: str

    def __post_init__(self) -> None:
        _identity(self.session_id, "session_id")
        _identity(self.rng_stream_id, "rng_stream_id")
        object.__setattr__(
            self, "decision_seq", _exact_int(self.decision_seq, "decision_seq")
        )
        object.__setattr__(
            self,
            "held_ordinal",
            _exact_int(self.held_ordinal, "held_ordinal", minimum=1),
        )

    @property
    def physical_key(self) -> Tuple[str, int, int]:
        return (self.session_id, self.decision_seq, self.held_ordinal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_seq": self.decision_seq,
            "held_ordinal": self.held_ordinal,
            "rng_stream_id": self.rng_stream_id,
            "session_id": self.session_id,
        }


@dataclass(frozen=True, slots=True)
class EndpointEvidenceV1:
    q_e4: int
    total_transmitted_bytes: int
    source_row_sha256: str

    @classmethod
    def from_node(cls, node: PayloadNodeV1) -> "EndpointEvidenceV1":
        return cls(
            q_e4=node.q_e4,
            total_transmitted_bytes=node.total_transmitted_bytes,
            source_row_sha256=node.source_row_sha256,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q_e4": self.q_e4,
            "source_row_sha256": self.source_row_sha256,
            "total_transmitted_bytes": self.total_transmitted_bytes,
        }


@dataclass(frozen=True, slots=True)
class HeldPayloadEstimateV1:
    """Identity-bound marginal payload estimate for one executed held action."""

    evidence_label: str
    provider_binding_sha256: str
    selection_identity_sha256: str
    counter: HeldSelectionCounterV1
    rng_draw: float
    selected_sample_id: str
    selected_episode_id: str
    selected_frame_id: int
    selected_scene_source_sha256: str
    selected_curve_sha256: str
    source_selection_sha256: str
    source_database_sha256: str
    inclusion_probability: float
    sampling_weight: float
    marginal_selection_probability: float
    mode_id: int
    q_e4: int
    total_transmitted_bytes: int | float
    interpolation_status: str
    endpoints: Tuple[EndpointEvidenceV1, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counter": self.counter.to_dict(),
            "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
            "evidence_label": self.evidence_label,
            "inclusion_probability": self.inclusion_probability,
            "interpolation_status": self.interpolation_status,
            "marginal_selection_probability": self.marginal_selection_probability,
            "mode_id": self.mode_id,
            "provider_binding_sha256": self.provider_binding_sha256,
            "q_e4": self.q_e4,
            "rng_draw": self.rng_draw,
            "sampling_weight": self.sampling_weight,
            "selected_curve_sha256": self.selected_curve_sha256,
            "selected_episode_id": self.selected_episode_id,
            "selected_frame_id": self.selected_frame_id,
            "selected_sample_id": self.selected_sample_id,
            "selected_scene_source_sha256": self.selected_scene_source_sha256,
            "selection_identity_sha256": self.selection_identity_sha256,
            "source_database_sha256": self.source_database_sha256,
            "source_selection_sha256": self.source_selection_sha256,
            "total_transmitted_bytes": self.total_transmitted_bytes,
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha256(
            {
                "record": "splitfusion_run4_held_payload_estimate_v1",
                "value": self.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class _Scene:
    sample_id: str
    episode_id: str
    frame_id: int
    selection_rank_within_train: int
    inclusion_probability: float
    sampling_weight: float
    scene_source_sha256: str
    curves_by_mode: Dict[int, ScenePayloadCurveV1]


@dataclass(frozen=True, slots=True)
class _Issued:
    request_sha256: str
    result: HeldPayloadEstimateV1


class HeldPayloadProviderV1:
    """Validated in-memory marginal provider with no filesystem dependency."""

    def __init__(
        self,
        curves: Sequence[ScenePayloadCurveV1],
        *,
        expected_inventory_sha256: str,
    ) -> None:
        if type(curves) not in (tuple, list) or not curves:
            raise SourceRejected("curves must be a nonempty materialized sequence")
        material = tuple(curves)
        if any(type(curve) is not ScenePayloadCurveV1 for curve in material):
            raise SourceRejected("curve inventory contains a foreign record type")
        expected = _digest(expected_inventory_sha256, "expected_inventory_sha256")
        observed = payload_curve_inventory_sha256(material)
        if observed != expected:
            raise BindingMismatch(
                f"payload curve inventory drift: expected {expected}, observed {observed}"
            )

        selection_bindings = {curve.source_selection_sha256 for curve in material}
        database_bindings = {curve.source_database_sha256 for curve in material}
        if len(selection_bindings) != 1 or len(database_bindings) != 1:
            raise BindingMismatch("mixed selection/database bindings are forbidden")
        self.source_selection_sha256 = next(iter(selection_bindings))
        self.source_database_sha256 = next(iter(database_bindings))

        grouped: Dict[str, list[ScenePayloadCurveV1]] = {}
        endpoint_owners: Dict[str, Tuple[str, int, int]] = {}
        curve_keys: set[Tuple[str, int]] = set()
        for curve in material:
            key = (curve.sample_id, curve.mode_id)
            if key in curve_keys:
                raise IdentifierConflict(f"duplicate scene/mode curve: {key}")
            curve_keys.add(key)
            grouped.setdefault(curve.sample_id, []).append(curve)
            for node in curve.nodes:
                owner = (curve.sample_id, curve.mode_id, node.q_e4)
                previous = endpoint_owners.setdefault(node.source_row_sha256, owner)
                if previous != owner:
                    raise IdentifierConflict(
                        "one source row digest identifies multiple payload endpoints"
                    )

        scenes = []
        physical_owners: Dict[Tuple[str, int], str] = {}
        rank_owners: Dict[int, str] = {}
        scene_digest_owners: Dict[str, str] = {}
        curves_by_key: Dict[Tuple[str, int], ScenePayloadCurveV1] = {}
        for sample_id, scene_curves in grouped.items():
            scene_curves.sort(key=lambda item: item.mode_id)
            if tuple(curve.mode_id for curve in scene_curves) != SUPPORTED_MODE_IDS:
                raise SourceRejected(
                    f"scene {sample_id!r} does not cover all registered modes"
                )
            first = scene_curves[0]
            shared = (
                first.episode_id,
                first.frame_id,
                first.selection_rank_within_train,
                first.inclusion_probability,
                first.sampling_weight,
                first.scene_source_sha256,
                first.source_selection_sha256,
                first.source_database_sha256,
                first.source_split,
            )
            if any(
                (
                    curve.episode_id,
                    curve.frame_id,
                    curve.selection_rank_within_train,
                    curve.inclusion_probability,
                    curve.sampling_weight,
                    curve.scene_source_sha256,
                    curve.source_selection_sha256,
                    curve.source_database_sha256,
                    curve.source_split,
                )
                != shared
                for curve in scene_curves[1:]
            ):
                raise IdentifierConflict(
                    f"scene identity conflicts across modes: {sample_id!r}"
                )
            physical_key = (first.episode_id, first.frame_id)
            previous = physical_owners.setdefault(physical_key, sample_id)
            if previous != sample_id:
                raise IdentifierConflict("one episode/frame has multiple sample IDs")
            rank_key = first.selection_rank_within_train
            previous = rank_owners.setdefault(rank_key, sample_id)
            if previous != sample_id:
                raise IdentifierConflict("one train selection rank has multiple samples")
            previous = scene_digest_owners.setdefault(
                first.scene_source_sha256, sample_id
            )
            if previous != sample_id:
                raise IdentifierConflict("one scene digest has multiple sample IDs")
            by_mode = {curve.mode_id: curve for curve in scene_curves}
            curves_by_key.update(
                {(sample_id, mode_id): curve for mode_id, curve in by_mode.items()}
            )
            scenes.append(
                _Scene(
                    sample_id=sample_id,
                    episode_id=first.episode_id,
                    frame_id=first.frame_id,
                    selection_rank_within_train=first.selection_rank_within_train,
                    inclusion_probability=first.inclusion_probability,
                    sampling_weight=first.sampling_weight,
                    scene_source_sha256=first.scene_source_sha256,
                    curves_by_mode=by_mode,
                )
            )
        scenes.sort(key=lambda item: item.sample_id)
        self._scenes = tuple(scenes)
        self._curves_by_key = curves_by_key
        self._total_sampling_weight = sum(scene.sampling_weight for scene in scenes)
        if not math.isfinite(self._total_sampling_weight) or self._total_sampling_weight <= 0:
            raise SourceRejected("aggregate sampling weight is invalid")
        self.inventory_sha256 = observed
        self.binding_sha256 = _sha256(
            {
                "evidence_label": EVIDENCE_LABEL,
                "inventory_sha256": observed,
                "q_e4_grid": list(Q_E4_GRID),
                "record": "splitfusion_run4_held_payload_provider_binding_v1",
                "source_database_sha256": self.source_database_sha256,
                "source_selection_sha256": self.source_selection_sha256,
                "supported_mode_ids": list(SUPPORTED_MODE_IDS),
                "weighting_policy": WEIGHTING_POLICY,
            }
        )
        self._issued: Dict[Tuple[str, int, int], _Issued] = {}

    @property
    def scene_count(self) -> int:
        return len(self._scenes)

    @property
    def issued_count(self) -> int:
        return len(self._issued)

    @staticmethod
    def _validate_action(mode_id: Any, q_e4: Any) -> Tuple[int, int]:
        checked_mode = _exact_int(mode_id, "mode_id")
        checked_q = _exact_int(q_e4, "q_e4")
        if checked_mode not in SUPPORTED_MODE_IDS:
            raise UnsupportedAction(f"unsupported mode_id {checked_mode}")
        if not Q_E4_MIN <= checked_q <= Q_E4_MAX:
            raise UnsupportedAction(
                f"q_e4 {checked_q} is outside [{Q_E4_MIN}, {Q_E4_MAX}]"
            )
        return checked_mode, checked_q

    @staticmethod
    def _draw(rng: RandomLike) -> float:
        if isinstance(rng, types.ModuleType):
            raise LocalRngRequired("module/global RNGs are forbidden; pass a local RNG")
        method = getattr(rng, "random", None)
        if not callable(method):
            raise LocalRngRequired("caller must provide a local RNG with random()")
        try:
            value = method()
        except Exception as exc:
            raise LocalRngRequired("caller-provided local RNG failed") from exc
        draw = _finite(value, "local RNG draw")
        if not 0.0 <= draw < 1.0:
            raise LocalRngRequired("local RNG draw must lie in [0,1)")
        return draw

    def _select_scene(self, draw: float) -> _Scene:
        threshold = draw * self._total_sampling_weight
        cumulative = 0.0
        for scene in self._scenes:
            cumulative += scene.sampling_weight
            if threshold < cumulative:
                return scene
        return self._scenes[-1]

    @staticmethod
    def _evaluate_curve(
        curve: ScenePayloadCurveV1, q_e4: int
    ) -> Tuple[int | float, str, Tuple[EndpointEvidenceV1, ...]]:
        exact = next((node for node in curve.nodes if node.q_e4 == q_e4), None)
        if exact is not None:
            return (
                exact.total_transmitted_bytes,
                EXACT_NODE_STATUS,
                (EndpointEvidenceV1.from_node(exact),),
            )
        lower = max((node for node in curve.nodes if node.q_e4 < q_e4), key=lambda n: n.q_e4)
        upper = min((node for node in curve.nodes if node.q_e4 > q_e4), key=lambda n: n.q_e4)
        alpha = (q_e4 - lower.q_e4) / (upper.q_e4 - lower.q_e4)
        payload = lower.total_transmitted_bytes + alpha * (
            upper.total_transmitted_bytes - lower.total_transmitted_bytes
        )
        if not upper.total_transmitted_bytes < payload < lower.total_transmitted_bytes:
            raise SourceRejected("interpolated payload escaped its strict same-scene bracket")
        if not math.isfinite(payload) or payload <= 0:
            raise SourceRejected("interpolated payload is nonpositive or nonfinite")
        return (
            payload,
            INTERPOLATED_STATUS,
            (
                EndpointEvidenceV1.from_node(lower),
                EndpointEvidenceV1.from_node(upper),
            ),
        )

    def evaluate(
        self,
        *,
        counter: HeldSelectionCounterV1,
        rng: RandomLike,
        mode_id: Any,
        q_e4: Any,
    ) -> HeldPayloadEstimateV1:
        """Select a weighted train scene and evaluate the exact executed action.

        A repeated physical counter is idempotent only for the identical
        counter/action request.  Reusing it with another RNG stream or action
        is an identifier conflict and is rejected before consuming the RNG.
        """

        if type(counter) is not HeldSelectionCounterV1:
            raise IdentifierConflict("counter must be HeldSelectionCounterV1")
        checked_mode, checked_q = self._validate_action(mode_id, q_e4)
        request_document = {
            "counter": counter.to_dict(),
            "mode_id": checked_mode,
            "provider_binding_sha256": self.binding_sha256,
            "q_e4": checked_q,
            "record": "splitfusion_run4_held_payload_request_v1",
        }
        request_sha = _sha256(request_document)
        prior = self._issued.get(counter.physical_key)
        if prior is not None:
            if prior.request_sha256 != request_sha:
                raise IdentifierConflict(
                    "held-frame counter was reused for a conflicting request"
                )
            return prior.result

        draw = self._draw(rng)
        scene = self._select_scene(draw)
        curve = self._curves_by_key[(scene.sample_id, checked_mode)]
        payload, status, endpoints = self._evaluate_curve(curve, checked_q)
        selection_identity_sha = _sha256(
            {
                "counter": counter.to_dict(),
                "provider_binding_sha256": self.binding_sha256,
                "record": "splitfusion_run4_marginal_scene_selection_v1",
                "rng_draw": draw,
                "selected_curve_sha256": curve.curve_sha256,
                "selected_sample_id": scene.sample_id,
            }
        )
        result = HeldPayloadEstimateV1(
            evidence_label=EVIDENCE_LABEL,
            provider_binding_sha256=self.binding_sha256,
            selection_identity_sha256=selection_identity_sha,
            counter=counter,
            rng_draw=draw,
            selected_sample_id=scene.sample_id,
            selected_episode_id=scene.episode_id,
            selected_frame_id=scene.frame_id,
            selected_scene_source_sha256=scene.scene_source_sha256,
            selected_curve_sha256=curve.curve_sha256,
            source_selection_sha256=self.source_selection_sha256,
            source_database_sha256=self.source_database_sha256,
            inclusion_probability=scene.inclusion_probability,
            sampling_weight=scene.sampling_weight,
            marginal_selection_probability=(
                scene.sampling_weight / self._total_sampling_weight
            ),
            mode_id=checked_mode,
            q_e4=checked_q,
            total_transmitted_bytes=payload,
            interpolation_status=status,
            endpoints=endpoints,
        )
        self._issued[counter.physical_key] = _Issued(request_sha, result)
        return result
