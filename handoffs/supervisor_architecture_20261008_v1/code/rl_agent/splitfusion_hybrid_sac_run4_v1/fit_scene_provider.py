"""Train-only, checkpointable fit-scene provider for Run-4.

The provider is deliberately an in-memory boundary.  It never discovers or
loads evidence.  A caller must first load and verify the frozen quality
surface, intersect it with the registered temporal-block partition, build the
391-scene inventory below, and supply a separately pinned verification
envelope which closes the quality and held-payload bindings.

Only ``(camera_si, radar_p40)`` leaves this module as policy-visible scene
state.  Scene identity, selection weights, partition membership and source
digests remain environment-side evidence.
"""

from __future__ import annotations

import math
import numbers
import random
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import held_payload, quality_adapter
from rl_agent.splitfusion_hybrid_sac_v1.empirical_quality_surface import (
    PolicySceneView,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    canonical_sha256,
)

__all__ = [
    "EXPECTED_TRAIN_SCENE_COUNT",
    "REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256",
    "REGISTERED_TRAIN_SCENE_INVENTORY_SHA256",
    "FitSceneProviderError",
    "InventoryRejected",
    "EnvelopeBindingError",
    "SceneDrawError",
    "ProviderStateError",
    "TrainSceneInventoryRecordV1",
    "RegisteredTrainSceneInventoryV1",
    "VerifiedFitSceneEnvelopeV1",
    "FitSceneDrawV1",
    "FitSceneProviderBindingV1",
    "FitSceneProviderStateV1",
    "Run4FitSceneProviderV1",
]


EXPECTED_TRAIN_SCENE_COUNT = 391
TRAIN_SPLIT = "train"
FIT_VALIDATION_SPLIT = "fit_validation"
PARTITION_HASH_DOMAIN = "splitfusion.empirical_fit_partition.v1"
PARTITION_VALIDATION_MODULUS = 5
PARTITION_VALIDATION_RESIDUE = 0

# Existing immutable partition and the canonical 391-row intersection of that
# partition with the quality-valid fit inventory.  The latter was derived once
# from the hash-pinned selection/database/surface; this module never reopens it.
REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256 = (
    "44dad342d09cc39cab37e069627b8b9019e8cb9221fff25e1a8a8f240e8e1061"
)
REGISTERED_TRAIN_SCENE_INVENTORY_SHA256 = (
    "1d0d102fafb8797fd4b69e8524dd301229ddbc6f9bc76072048545c1d022ebbd"
)


class FitSceneProviderError(ValueError):
    """Base class for train-scene provider failures."""


class InventoryRejected(FitSceneProviderError):
    """The supplied scene inventory is not the registered 391-scene set."""


class EnvelopeBindingError(FitSceneProviderError):
    """The verified caller envelope does not close all provider bindings."""


class SceneDrawError(FitSceneProviderError):
    """A scene draw is foreign, stale, or inconsistent with its provider."""


class ProviderStateError(FitSceneProviderError):
    """A checkpoint cannot be restored without changing the random stream."""


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EnvelopeBindingError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _identity(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise FitSceneProviderError(f"{name} must be a nonempty canonical string")
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise FitSceneProviderError(f"{name} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise FitSceneProviderError(f"{name} must be >= {minimum}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise FitSceneProviderError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise FitSceneProviderError(f"{name} must be finite")
    return result


def _partition_split(frame_id: int) -> str:
    block = frame_id // 100
    domain = f"{PARTITION_HASH_DOMAIN}:{block}".encode("ascii")
    import hashlib

    value = int.from_bytes(hashlib.sha256(domain).digest(), "big")
    return (
        FIT_VALIDATION_SPLIT
        if value % PARTITION_VALIDATION_MODULUS == PARTITION_VALIDATION_RESIDUE
        else TRAIN_SPLIT
    )


@dataclass(frozen=True, slots=True)
class TrainSceneInventoryRecordV1:
    """One hidden fit-scene selection with a re-derived train assignment."""

    selection: quality_adapter.FitSceneSelectionV1
    partition_split: str
    temporal_block: int

    def __post_init__(self) -> None:
        if type(self.selection) is not quality_adapter.FitSceneSelectionV1:
            raise InventoryRejected(
                "selection must be exactly FitSceneSelectionV1"
            )
        if self.selection.grid_split != "fit":
            raise InventoryRejected("quality evidence must come from the fit split")
        block = _exact_int(self.temporal_block, "temporal_block")
        if block != self.selection.frame_id // 100:
            raise InventoryRejected("temporal block is not frame_id // 100")
        derived = _partition_split(self.selection.frame_id)
        if self.partition_split != derived:
            raise InventoryRejected("partition label contradicts the registered rule")
        if self.partition_split != TRAIN_SPLIT:
            raise InventoryRejected(
                "fit-validation identities are categorically unavailable to training"
            )
        object.__setattr__(self, "temporal_block", block)

    @classmethod
    def from_selection(
        cls, selection: quality_adapter.FitSceneSelectionV1
    ) -> "TrainSceneInventoryRecordV1":
        if type(selection) is not quality_adapter.FitSceneSelectionV1:
            raise InventoryRejected(
                "selection must be exactly FitSceneSelectionV1"
            )
        return cls(
            selection=selection,
            partition_split=_partition_split(selection.frame_id),
            temporal_block=selection.frame_id // 100,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_split": self.partition_split,
            "selection": self.selection.hidden_dict(),
            "temporal_block": self.temporal_block,
        }


@dataclass(frozen=True, slots=True)
class RegisteredTrainSceneInventoryV1:
    """The exact registered 391-scene training inventory, never validation."""

    records: Tuple[TrainSceneInventoryRecordV1, ...]
    registered_partition_sha256: str
    selection_manifest_sha256: str
    database_sha256: str
    surface_binding_sha256: str

    def __post_init__(self) -> None:
        if type(self.records) is not tuple or any(
            type(item) is not TrainSceneInventoryRecordV1 for item in self.records
        ):
            raise InventoryRejected(
                "records must be an exact tuple of TrainSceneInventoryRecordV1"
            )
        if len(self.records) != EXPECTED_TRAIN_SCENE_COUNT:
            raise InventoryRejected(
                f"train inventory must contain {EXPECTED_TRAIN_SCENE_COUNT} scenes"
            )
        for name in (
            "registered_partition_sha256",
            "selection_manifest_sha256",
            "database_sha256",
            "surface_binding_sha256",
        ):
            _digest(getattr(self, name), name)
        if (
            self.registered_partition_sha256
            != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
        ):
            raise InventoryRejected("registered fit-partition binding drift")

        sample_ids = tuple(item.selection.sample_id for item in self.records)
        if sample_ids != tuple(sorted(sample_ids)):
            raise InventoryRejected("train records must be sorted by sample_id")
        if len(set(sample_ids)) != len(sample_ids):
            raise InventoryRejected("train inventory duplicated a sample_id")
        physical = tuple(
            (item.selection.episode_id, item.selection.frame_id)
            for item in self.records
        )
        ranks = tuple(
            item.selection.selection_rank_within_fit for item in self.records
        )
        if len(set(physical)) != len(physical):
            raise InventoryRejected("one physical frame has multiple sample IDs")
        if len(set(ranks)) != len(ranks):
            raise InventoryRejected("one fit selection rank has multiple samples")
        if any(
            item.selection.surface_binding_sha256 != self.surface_binding_sha256
            for item in self.records
        ):
            raise InventoryRejected("scene/surface binding mismatch")
        total_weight = sum(item.selection.sampling_weight for item in self.records)
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise InventoryRejected("aggregate scene sampling weight is invalid")
        if self.canonical_sha256 != REGISTERED_TRAIN_SCENE_INVENTORY_SHA256:
            raise InventoryRejected("registered 391-scene inventory digest drift")

    @classmethod
    def from_verified_selections(
        cls,
        selections: Sequence[quality_adapter.FitSceneSelectionV1],
        *,
        registered_partition_sha256: str,
        selection_manifest_sha256: str,
        database_sha256: str,
        surface_binding_sha256: str,
    ) -> "RegisteredTrainSceneInventoryV1":
        if type(selections) not in (tuple, list):
            raise InventoryRejected("selections must be a materialized tuple/list")
        records = tuple(
            sorted(
                (TrainSceneInventoryRecordV1.from_selection(item) for item in selections),
                key=lambda item: item.selection.sample_id,
            )
        )
        return cls(
            records=records,
            registered_partition_sha256=registered_partition_sha256,
            selection_manifest_sha256=selection_manifest_sha256,
            database_sha256=database_sha256,
            surface_binding_sha256=surface_binding_sha256,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_sha256": self.database_sha256,
            "records": [item.to_dict() for item in self.records],
            "record": "splitfusion_run4_registered_train_scene_inventory_v1",
            "registered_partition_sha256": self.registered_partition_sha256,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "surface_binding_sha256": self.surface_binding_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def total_sampling_weight(self) -> float:
        return sum(item.selection.sampling_weight for item in self.records)


@dataclass(frozen=True, slots=True)
class VerifiedFitSceneEnvelopeV1:
    """Caller-issued verification envelope closing scene and adapter sources."""

    inventory: RegisteredTrainSceneInventoryV1
    quality_adapter_binding_sha256: str
    held_provider_binding_sha256: str
    held_provider_inventory_sha256: str
    held_scene_identity_inventory_sha256: str
    held_provider_scene_count: int
    verification_report_sha256: str
    verification_status: str = "VERIFIED_391_TRAIN_SCENES_ONLY"

    def __post_init__(self) -> None:
        if type(self.inventory) is not RegisteredTrainSceneInventoryV1:
            raise EnvelopeBindingError(
                "inventory must be RegisteredTrainSceneInventoryV1"
            )
        for name in (
            "quality_adapter_binding_sha256",
            "held_provider_binding_sha256",
            "held_provider_inventory_sha256",
            "held_scene_identity_inventory_sha256",
            "verification_report_sha256",
        ):
            _digest(getattr(self, name), name)
        if (
            self.held_scene_identity_inventory_sha256
            != self.inventory.canonical_sha256
        ):
            raise EnvelopeBindingError(
                "held-provider scene identities do not close to the train inventory"
            )
        count = _exact_int(
            self.held_provider_scene_count, "held_provider_scene_count", minimum=1
        )
        if count != EXPECTED_TRAIN_SCENE_COUNT:
            raise EnvelopeBindingError(
                "held provider must cover all 391 registered train scenes"
            )
        if self.verification_status != "VERIFIED_391_TRAIN_SCENES_ONLY":
            raise EnvelopeBindingError("verification status is not accepted")
        object.__setattr__(self, "held_provider_scene_count", count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "held_provider_binding_sha256": self.held_provider_binding_sha256,
            "held_provider_inventory_sha256": self.held_provider_inventory_sha256,
            "held_provider_scene_count": self.held_provider_scene_count,
            "held_scene_identity_inventory_sha256": (
                self.held_scene_identity_inventory_sha256
            ),
            "inventory_sha256": self.inventory.canonical_sha256,
            "quality_adapter_binding_sha256": self.quality_adapter_binding_sha256,
            "record": "splitfusion_run4_verified_fit_scene_envelope_v1",
            "verification_report_sha256": self.verification_report_sha256,
            "verification_status": self.verification_status,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FitSceneDrawV1:
    """Environment-hidden selected scene plus a policy-safe scene projection."""

    provider_binding_sha256: str
    draw_ordinal: int
    rng_draw: float
    selection: quality_adapter.FitSceneSelectionV1

    def __post_init__(self) -> None:
        _digest(self.provider_binding_sha256, "provider_binding_sha256")
        object.__setattr__(
            self,
            "draw_ordinal",
            _exact_int(self.draw_ordinal, "draw_ordinal", minimum=1),
        )
        draw = _finite(self.rng_draw, "rng_draw")
        if not 0.0 <= draw < 1.0:
            raise SceneDrawError("rng_draw must lie in [0,1)")
        if type(self.selection) is not quality_adapter.FitSceneSelectionV1:
            raise SceneDrawError("selection must be FitSceneSelectionV1")
        object.__setattr__(self, "rng_draw", draw)

    def policy_scene(self) -> PolicySceneView:
        """Return SI/P40 only; hidden identity never enters the actor state."""

        return self.selection.policy_scene()

    def policy_scene_features(self) -> Tuple[float, float]:
        scene = self.policy_scene()
        return (scene.camera_si, scene.radar_p40)

    def to_hidden_dict(self) -> Dict[str, Any]:
        return {
            "draw_ordinal": self.draw_ordinal,
            "provider_binding_sha256": self.provider_binding_sha256,
            "rng_draw": self.rng_draw,
            "selection": self.selection.hidden_dict(),
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_hidden_fit_scene_draw_v1",
                "value": self.to_hidden_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class FitSceneProviderBindingV1:
    envelope_sha256: str
    inventory_sha256: str
    quality_adapter_binding_sha256: str
    scene_rng_seed: int
    scene_rng_stream_id: str
    held_rng_seed: int
    held_rng_stream_id: str

    def __post_init__(self) -> None:
        for name in (
            "envelope_sha256",
            "inventory_sha256",
            "quality_adapter_binding_sha256",
        ):
            _digest(getattr(self, name), name)
        for name in ("scene_rng_seed", "held_rng_seed"):
            value = _exact_int(getattr(self, name), name)
            if value >= 1 << 63:
                raise EnvelopeBindingError(f"{name} must be below 2^63")
            object.__setattr__(self, name, value)
        _identity(self.scene_rng_stream_id, "scene_rng_stream_id")
        _identity(self.held_rng_stream_id, "held_rng_stream_id")
        if self.scene_rng_stream_id == self.held_rng_stream_id:
            raise EnvelopeBindingError("scene and held RNG streams must be distinct")

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_fit_scene_provider_binding_v1",
                "value": {
                    item.name: getattr(self, item.name) for item in fields(self)
                },
            }
        )


@dataclass(frozen=True, slots=True)
class FitSceneProviderStateV1:
    provider_binding_sha256: str
    selected_count: int
    held_draw_count: int
    scene_rng_state: tuple
    held_rng_state: tuple
    active_draw: Optional[FitSceneDrawV1]

    def __post_init__(self) -> None:
        _digest(self.provider_binding_sha256, "provider_binding_sha256")
        object.__setattr__(
            self, "selected_count", _exact_int(self.selected_count, "selected_count")
        )
        object.__setattr__(
            self, "held_draw_count", _exact_int(self.held_draw_count, "held_draw_count")
        )
        if type(self.scene_rng_state) is not tuple or type(self.held_rng_state) is not tuple:
            raise ProviderStateError("RNG states must be exact tuples")
        if self.active_draw is not None and type(self.active_draw) is not FitSceneDrawV1:
            raise ProviderStateError("active_draw must be FitSceneDrawV1 or None")


class Run4FitSceneProviderV1:
    """Weighted train-scene sampler and exact quality/held-payload join."""

    def __init__(
        self,
        *,
        envelope: VerifiedFitSceneEnvelopeV1,
        adapter: quality_adapter.Run4QualityPayloadAdapterV1,
        expected_envelope_sha256: str,
        scene_rng_seed: int,
        held_rng_seed: int,
        scene_rng_stream_id: str,
        held_rng_stream_id: str,
    ) -> None:
        if type(envelope) is not VerifiedFitSceneEnvelopeV1:
            raise EnvelopeBindingError("envelope must be VerifiedFitSceneEnvelopeV1")
        if type(adapter) is not quality_adapter.Run4QualityPayloadAdapterV1:
            raise EnvelopeBindingError(
                "adapter must be exactly Run4QualityPayloadAdapterV1"
            )
        expected = _digest(expected_envelope_sha256, "expected_envelope_sha256")
        if envelope.canonical_sha256 != expected:
            raise EnvelopeBindingError("verified fit-scene envelope digest drift")
        adapter_binding = adapter.binding.canonical_sha256
        if adapter_binding != envelope.quality_adapter_binding_sha256:
            raise EnvelopeBindingError("quality-adapter binding mismatch")
        bound_held_provider = getattr(adapter, "_held_provider", None)
        if type(bound_held_provider) is not held_payload.HeldPayloadProviderV1:
            raise EnvelopeBindingError("adapter lost its exact held-payload provider")
        if bound_held_provider.scene_count != envelope.held_provider_scene_count:
            raise EnvelopeBindingError("held-provider scene-count attestation is false")
        # HeldPayloadProviderV1 intentionally exposes estimates rather than its
        # source identities.  At this one construction boundary, inspect its
        # already-validated immutable scene snapshot so a 391-row *different*
        # population cannot satisfy a count-only attestation.
        bound_held_scenes = tuple(getattr(bound_held_provider, "_scenes", ()))
        expected_held = tuple(
            (
                record.selection.sample_id,
                record.selection.episode_id,
                record.selection.frame_id,
                rank,
                record.selection.inclusion_probability,
                record.selection.sampling_weight,
                record.selection.canonical_sha256,
            )
            for rank, record in enumerate(envelope.inventory.records)
        )
        observed_held = tuple(
            (
                scene.sample_id,
                scene.episode_id,
                scene.frame_id,
                scene.selection_rank_within_train,
                scene.inclusion_probability,
                scene.sampling_weight,
                scene.scene_source_sha256,
            )
            for scene in bound_held_scenes
        )
        if observed_held != expected_held:
            raise EnvelopeBindingError(
                "held-provider scene identities do not equal the 391-scene train inventory"
            )
        if (
            adapter.binding.surface_binding_sha256
            != envelope.inventory.surface_binding_sha256
            or adapter.binding.surface_database_sha256
            != envelope.inventory.database_sha256
            or adapter.binding.surface_selection_file_sha256
            != envelope.inventory.selection_manifest_sha256
            or adapter.binding.held_provider_binding_sha256
            != envelope.held_provider_binding_sha256
            or adapter.binding.held_inventory_sha256
            != envelope.held_provider_inventory_sha256
        ):
            raise EnvelopeBindingError("envelope source bindings do not close")

        self._envelope = envelope
        self._adapter = adapter
        self._records = envelope.inventory.records
        self._by_sample = {
            item.selection.sample_id: item for item in self._records
        }
        self._total_weight = envelope.inventory.total_sampling_weight
        self.binding = FitSceneProviderBindingV1(
            envelope_sha256=envelope.canonical_sha256,
            inventory_sha256=envelope.inventory.canonical_sha256,
            quality_adapter_binding_sha256=adapter_binding,
            scene_rng_seed=scene_rng_seed,
            scene_rng_stream_id=scene_rng_stream_id,
            held_rng_seed=held_rng_seed,
            held_rng_stream_id=held_rng_stream_id,
        )
        self._scene_rng = random.Random(self.binding.scene_rng_seed)
        self._held_rng = random.Random(self.binding.held_rng_seed)
        self._selected_count = 0
        self._held_draw_count = 0
        self._active_draw: Optional[FitSceneDrawV1] = None

    @property
    def selected_count(self) -> int:
        return self._selected_count

    @property
    def active_draw(self) -> Optional[FitSceneDrawV1]:
        return self._active_draw

    def _revalidate_bindings(self) -> None:
        if self._envelope.inventory.canonical_sha256 != self.binding.inventory_sha256:
            raise EnvelopeBindingError("train inventory changed after construction")
        if self._envelope.canonical_sha256 != self.binding.envelope_sha256:
            raise EnvelopeBindingError("verification envelope changed after construction")
        if self._adapter.binding.canonical_sha256 != self.binding.quality_adapter_binding_sha256:
            raise EnvelopeBindingError("quality-adapter binding changed after construction")

    def _select_record(self, draw: float) -> TrainSceneInventoryRecordV1:
        threshold = draw * self._total_weight
        cumulative = 0.0
        for record in self._records:
            cumulative += record.selection.sampling_weight
            if threshold < cumulative:
                return record
        return self._records[-1]

    def select_scene(self) -> FitSceneDrawV1:
        """Select exactly one registered training scene by recorded weight."""

        self._revalidate_bindings()
        draw = self._scene_rng.random()
        record = self._select_record(draw)
        selected = FitSceneDrawV1(
            provider_binding_sha256=self.binding.canonical_sha256,
            draw_ordinal=self._selected_count + 1,
            rng_draw=draw,
            selection=record.selection,
        )
        self._selected_count += 1
        self._active_draw = selected
        return selected

    def _require_active(self, draw: FitSceneDrawV1) -> FitSceneDrawV1:
        if type(draw) is not FitSceneDrawV1:
            raise SceneDrawError("draw must be exactly FitSceneDrawV1")
        if draw.provider_binding_sha256 != self.binding.canonical_sha256:
            raise SceneDrawError("draw/provider binding mismatch")
        if self._active_draw is None or (
            draw.canonical_sha256 != self._active_draw.canonical_sha256
        ):
            raise SceneDrawError("draw is not the provider's current active scene")
        registered = self._by_sample.get(draw.selection.sample_id)
        if registered is None or registered.selection != draw.selection:
            raise SceneDrawError("draw does not match the registered train inventory")
        return draw

    def reward_tensor(
        self,
        *,
        draw: FitSceneDrawV1,
        action: ExecutedActionIdentity,
        tensor_seq: int,
    ) -> quality_adapter.RewardTensorResultV1:
        """Join the exact selected scene into the empirical reward tensor."""

        self._revalidate_bindings()
        checked = self._require_active(draw)
        result = self._adapter.reward_tensor(
            selection=checked.selection,
            action=action,
            tensor_seq=tensor_seq,
        )
        if result.evidence.fit_selection_sha256 != checked.selection.canonical_sha256:
            raise SceneDrawError("reward tensor did not retain the selected scene")
        if result.policy_scene_features() != checked.policy_scene_features():
            raise SceneDrawError("reward tensor changed policy-visible scene values")
        return result

    def held_tensor(
        self,
        *,
        counter: held_payload.HeldSelectionCounterV1,
        action: ExecutedActionIdentity,
        tensor_seq: int,
    ) -> quality_adapter.HeldTensorResultV1:
        """Draw a train-only marginal held frame and close it to the same inventory."""

        self._revalidate_bindings()
        before = self._held_rng.getstate()
        result = self._adapter.held_tensor(
            counter=counter,
            rng=self._held_rng,
            action=action,
            tensor_seq=tensor_seq,
        )
        after = self._held_rng.getstate()
        if after != before:
            self._held_draw_count += 1
        estimate = result.estimate
        registered = self._by_sample.get(estimate.selected_sample_id)
        if registered is None:
            raise SceneDrawError("held provider selected a non-training scene")
        selection = registered.selection
        observed = (
            estimate.selected_episode_id,
            estimate.selected_frame_id,
            estimate.inclusion_probability,
            estimate.sampling_weight,
            estimate.selected_scene_source_sha256,
        )
        expected = (
            selection.episode_id,
            selection.frame_id,
            selection.inclusion_probability,
            selection.sampling_weight,
            selection.canonical_sha256,
        )
        if observed != expected:
            raise SceneDrawError("held selection does not match its train inventory row")
        return result

    def state_dict(self) -> FitSceneProviderStateV1:
        """Return all local random state required for exact continuation."""

        self._revalidate_bindings()
        return FitSceneProviderStateV1(
            provider_binding_sha256=self.binding.canonical_sha256,
            selected_count=self._selected_count,
            held_draw_count=self._held_draw_count,
            scene_rng_state=self._scene_rng.getstate(),
            held_rng_state=self._held_rng.getstate(),
            active_draw=self._active_draw,
        )

    def load_state_dict(self, state: FitSceneProviderStateV1) -> None:
        """Atomically restore a checkpoint from this exact provider binding."""

        self._revalidate_bindings()
        if type(state) is not FitSceneProviderStateV1:
            raise ProviderStateError("state must be FitSceneProviderStateV1")
        if state.provider_binding_sha256 != self.binding.canonical_sha256:
            raise ProviderStateError("checkpoint/provider binding mismatch")
        if state.active_draw is not None:
            if state.active_draw.provider_binding_sha256 != self.binding.canonical_sha256:
                raise ProviderStateError("checkpoint active draw has a foreign binding")
            record = self._by_sample.get(state.active_draw.selection.sample_id)
            if record is None or record.selection != state.active_draw.selection:
                raise ProviderStateError("checkpoint active draw is not registered")
            if state.active_draw.draw_ordinal != state.selected_count:
                raise ProviderStateError("checkpoint active draw/count mismatch")
        elif state.selected_count != 0:
            raise ProviderStateError("nonempty checkpoint lost its active draw")

        scene_probe = random.Random()
        held_probe = random.Random()
        try:
            scene_probe.setstate(state.scene_rng_state)
            held_probe.setstate(state.held_rng_state)
        except Exception as exc:
            raise ProviderStateError("checkpoint contains an invalid RNG state") from exc

        # Commit only after every check and both RNG states have parsed.
        self._scene_rng.setstate(state.scene_rng_state)
        self._held_rng.setstate(state.held_rng_state)
        self._selected_count = state.selected_count
        self._held_draw_count = state.held_draw_count
        self._active_draw = state.active_draw
