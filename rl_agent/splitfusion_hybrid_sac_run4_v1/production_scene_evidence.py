"""Exact production loader for Run-4 scene quality and held payloads.

The existing quality surface and temporal-block partition are already
hash-pinned, but the production Run-4 path must still close one important
join: the scene sampled for a reward and the marginal scene sampled for the
second (held) tensor must come from the exact same 391-scene *training*
inventory.  Test helpers previously used mechanically generated payload
curves, which are deliberately ineligible here.

This module builds every held-payload curve from exact rows of the pinned
quality database.  Loading is explicit and read-only; importing this module
performs no I/O, RNG operation, CUDA initialization, or process launch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.corrected_p40_sidecar import (
    load_exact_corrected_p40_sidecar,
)
from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from rl_agent.splitfusion_hybrid_sac_v1.empirical_quality_surface import (
    EXACT_GRID_ROW_EVIDENCE,
    EmpiricalQualitySurface,
    load_empirical_quality_surface,
)
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.contract import (
    MODE_COUNT,
    Q_E4_GRID,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256

from . import fit_scene_provider, held_payload, quality_adapter, scientific_basis


SCHEMA_ID = "splitfusion.run4.production_scene_evidence.v1"
SCHEMA_VERSION = 1
EXPECTED_TRAIN_SCENE_COUNT = fit_scene_provider.EXPECTED_TRAIN_SCENE_COUNT
EXPECTED_CURVE_COUNT = EXPECTED_TRAIN_SCENE_COUNT * MODE_COUNT

# Registered only after the exact builder ran against independently verified,
# hash-pinned inputs.  Public loading recomputes every value and fails closed
# on any drift; these constants authorize this evidence bundle, not arbitrary
# counterfactual or held-out-scene data.
REGISTERED_HELD_CURVE_INVENTORY_SHA256: Optional[str] = (
    "21eee0c7923fd6805b9657fe82e8b9425c6f063980e6446398e2dd8418609394"
)
REGISTERED_HELD_PROVIDER_BINDING_SHA256: Optional[str] = (
    "5a459c22f8c3b3f173d77ebb0b992e27156a4ed105c98c1630c462c1ee0796f9"
)
REGISTERED_QUALITY_ADAPTER_BINDING_SHA256: Optional[str] = (
    "4df93e88df821556c9e0aa521ec5b6b3a119071213ffa8234dc44aa5b53e56b7"
)
REGISTERED_VERIFICATION_REPORT_SHA256: Optional[str] = (
    "f1635d5dca7a65eaf0ef6732b1249d35ad04e9c6374bb8d623d2b658d7e1ac47"
)
REGISTERED_SCENE_ENVELOPE_SHA256: Optional[str] = (
    "f6e11ce7d21cb56162f9a0d850f135ed3353d99d198afea4d35c8419e5f2383b"
)


class ProductionSceneEvidenceError(ValueError):
    """The exact production scene/held-payload evidence did not close."""


class ProductionSceneEvidenceUnavailable(ProductionSceneEvidenceError):
    """Reviewed production digests have not yet been registered."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _weighted_quantile(
    values: Sequence[Tuple[float, float, str]], quantile: float
) -> float:
    """Deterministic inverse-inclusion-weighted empirical quantile."""

    if not 0.0 <= quantile <= 1.0 or not values:
        raise ProductionSceneEvidenceError("invalid weighted quantile request")
    ordered = sorted(values, key=lambda item: (item[0], item[2]))
    total = sum(item[1] for item in ordered)
    if not math.isfinite(total) or total <= 0.0:
        raise ProductionSceneEvidenceError("invalid aggregate scene weight")
    threshold = quantile * total
    cumulative = 0.0
    for value, weight, _identity in ordered:
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0.0:
            raise ProductionSceneEvidenceError("non-finite scene scaling input")
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


@dataclass(frozen=True, slots=True)
class CameraSiScalingEvidenceV1:
    """Fit-only robust SI scaling; backlog scaling is calibrated separately."""

    weighted_median: float
    weighted_q25: float
    weighted_q75: float
    robust_scale_iqr: float
    scene_count: int
    scene_inventory_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "weighted_median",
            "weighted_q25",
            "weighted_q75",
            "robust_scale_iqr",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ProductionSceneEvidenceError(f"{name} must be finite")
        if self.robust_scale_iqr <= 0.0:
            raise ProductionSceneEvidenceError("camera SI IQR must be positive")
        if not self.weighted_q25 <= self.weighted_median <= self.weighted_q75:
            raise ProductionSceneEvidenceError("camera SI quantiles are unordered")
        if type(self.scene_count) is not int or self.scene_count != (
            EXPECTED_TRAIN_SCENE_COUNT
        ):
            raise ProductionSceneEvidenceError("camera scaling scene count drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "FIT_ONLY_INVERSE_INCLUSION_WEIGHTED_MEDIAN_AND_IQR",
            "robust_scale_iqr": self.robust_scale_iqr,
            "scene_count": self.scene_count,
            "scene_inventory_sha256": self.scene_inventory_sha256,
            "weighted_median": self.weighted_median,
            "weighted_q25": self.weighted_q25,
            "weighted_q75": self.weighted_q75,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {"record": "run4_camera_si_scaling_evidence_v1", "value": self.to_dict()}
        )


@dataclass(frozen=True, slots=True)
class ProductionSceneVerificationV1:
    """Canonical report produced before the envelope is constructed."""

    scientific_basis_sha256: str
    fit_partition_sha256: str
    surface_binding_sha256: str
    selection_manifest_sha256: str
    database_sha256: str
    train_inventory_sha256: str
    held_curve_inventory_sha256: str
    held_provider_binding_sha256: str
    quality_adapter_binding_sha256: str
    camera_si_scaling: CameraSiScalingEvidenceV1
    train_scene_count: int
    validation_scene_count: int
    curve_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_si_scaling": self.camera_si_scaling.to_dict(),
            "curve_count": self.curve_count,
            "database_sha256": self.database_sha256,
            "fit_partition_sha256": self.fit_partition_sha256,
            "held_curve_inventory_sha256": self.held_curve_inventory_sha256,
            "held_provider_binding_sha256": self.held_provider_binding_sha256,
            "quality_adapter_binding_sha256": self.quality_adapter_binding_sha256,
            "q_e4_grid": list(Q_E4_GRID),
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "scientific_basis_sha256": self.scientific_basis_sha256,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "surface_binding_sha256": self.surface_binding_sha256,
            "train_inventory_sha256": self.train_inventory_sha256,
            "train_scene_count": self.train_scene_count,
            "validation_scene_count": self.validation_scene_count,
            "verification_status": "EXACT_PINNED_TRAIN_SCENE_AND_PAYLOAD_JOIN",
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(slots=True)
class ProductionSceneEvidenceV1:
    """Owned resource bundle; callers must close the SQLite surface."""

    surface: EmpiricalQualitySurface
    inventory: fit_scene_provider.RegisteredTrainSceneInventoryV1
    held_provider: held_payload.HeldPayloadProviderV1
    adapter: quality_adapter.Run4QualityPayloadAdapterV1
    envelope: fit_scene_provider.VerifiedFitSceneEnvelopeV1
    verification: ProductionSceneVerificationV1

    def close(self) -> None:
        self.surface.close()

    def __enter__(self) -> "ProductionSceneEvidenceV1":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _build_curves(
    *,
    surface: EmpiricalQualitySurface,
    inventory: fit_scene_provider.RegisteredTrainSceneInventoryV1,
) -> tuple[held_payload.ScenePayloadCurveV1, ...]:
    curves: list[held_payload.ScenePayloadCurveV1] = []
    for train_rank, record in enumerate(inventory.records):
        selection = record.selection
        for mode_id in range(MODE_COUNT):
            nodes: list[held_payload.PayloadNodeV1] = []
            for q_e4 in Q_E4_GRID:
                query = surface.query_fit_q_e4(
                    selection.sample_id, mode_id, q_e4
                )
                if (
                    query.hidden.sample_id != selection.sample_id
                    or query.hidden.episode_id != selection.episode_id
                    or query.hidden.frame_id != selection.frame_id
                    or query.policy.mode_id != mode_id
                    or query.policy.q_e4 != q_e4
                    or query.policy.evidence_status != EXACT_GRID_ROW_EVIDENCE
                    or len(query.hidden.endpoint_evidence) != 1
                    or query.hidden.endpoint_evidence[0].q_e4 != q_e4
                ):
                    raise ProductionSceneEvidenceError(
                        "exact scene/mode/q payload query changed identity"
                    )
                payload = query.policy.payload.total_transmitted_bytes
                if type(payload) is not int or payload <= 0:
                    raise ProductionSceneEvidenceError(
                        "an exact held-payload node must be a positive integer"
                    )
                nodes.append(
                    held_payload.PayloadNodeV1(
                        q_e4=q_e4,
                        total_transmitted_bytes=payload,
                        source_row_sha256=(
                            query.hidden.endpoint_evidence[0].row_sha256
                        ),
                    )
                )
            curves.append(
                held_payload.ScenePayloadCurveV1(
                    sample_id=selection.sample_id,
                    episode_id=selection.episode_id,
                    frame_id=selection.frame_id,
                    selection_rank_within_train=train_rank,
                    source_split=TRAIN_SPLIT,
                    inclusion_probability=selection.inclusion_probability,
                    sampling_weight=selection.sampling_weight,
                    mode_id=mode_id,
                    scene_source_sha256=selection.canonical_sha256,
                    source_selection_sha256=inventory.selection_manifest_sha256,
                    source_database_sha256=inventory.database_sha256,
                    nodes=tuple(nodes),
                )
            )
    if len(curves) != EXPECTED_CURVE_COUNT:
        raise ProductionSceneEvidenceError("held-payload curve count drifted")
    return tuple(curves)


def _camera_scaling(
    inventory: fit_scene_provider.RegisteredTrainSceneInventoryV1,
) -> CameraSiScalingEvidenceV1:
    weighted = tuple(
        (
            record.selection.camera_si,
            record.selection.sampling_weight,
            record.selection.sample_id,
        )
        for record in inventory.records
    )
    q25 = _weighted_quantile(weighted, 0.25)
    median = _weighted_quantile(weighted, 0.50)
    q75 = _weighted_quantile(weighted, 0.75)
    return CameraSiScalingEvidenceV1(
        weighted_median=median,
        weighted_q25=q25,
        weighted_q75=q75,
        robust_scale_iqr=q75 - q25,
        scene_count=len(inventory.records),
        scene_inventory_sha256=inventory.canonical_sha256,
    )


def _assemble_unregistered(repo_root: Path) -> ProductionSceneEvidenceV1:
    """Build exact material; public callers additionally need registered hashes."""

    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise ProductionSceneEvidenceError("repo_root must be an absolute Path")
    scientific_basis.verify_quality_source(repo_root)
    partition = load_registered_empirical_fit_partition(project_root=repo_root)
    sidecar = load_exact_corrected_p40_sidecar(root=repo_root)
    surface = load_empirical_quality_surface(sidecar, project_root=repo_root)
    try:
        surface_binding = quality_adapter.surface_binding_sha256(surface.binding)
        train_ids = tuple(
            sorted(
                item.sample_id
                for item in partition.scene_assignments
                if item.split == TRAIN_SPLIT
            )
        )
        validation_ids = tuple(
            sorted(
                item.sample_id
                for item in partition.scene_assignments
                if item.split == FIT_VALIDATION_SPLIT
            )
        )
        selections = tuple(
            quality_adapter.FitSceneSelectionV1.from_query(
                surface.query_fit_q_e4(sample_id, 0, 0),
                surface_binding_sha256=surface_binding,
            )
            for sample_id in train_ids
        )
        inventory = (
            fit_scene_provider.RegisteredTrainSceneInventoryV1.from_verified_selections(
                selections,
                registered_partition_sha256=(
                    fit_scene_provider.REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
                ),
                selection_manifest_sha256=surface.binding.selection_file_sha256,
                database_sha256=surface.binding.database_file_sha256,
                surface_binding_sha256=surface_binding,
            )
        )
        curves = _build_curves(surface=surface, inventory=inventory)
        curve_inventory = held_payload.payload_curve_inventory_sha256(curves)
        held_provider = held_payload.HeldPayloadProviderV1(
            curves, expected_inventory_sha256=curve_inventory
        )
        adapter = quality_adapter.Run4QualityPayloadAdapterV1(
            surface=surface,
            held_provider=held_provider,
            expected_surface_binding_sha256=surface_binding,
            expected_held_provider_binding_sha256=held_provider.binding_sha256,
        )
        scaling = _camera_scaling(inventory)
        verification = ProductionSceneVerificationV1(
            scientific_basis_sha256=scientific_basis.SCIENTIFIC_BASIS_SHA256,
            fit_partition_sha256=(
                fit_scene_provider.REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
            ),
            surface_binding_sha256=surface_binding,
            selection_manifest_sha256=inventory.selection_manifest_sha256,
            database_sha256=inventory.database_sha256,
            train_inventory_sha256=inventory.canonical_sha256,
            held_curve_inventory_sha256=curve_inventory,
            held_provider_binding_sha256=held_provider.binding_sha256,
            quality_adapter_binding_sha256=adapter.binding.canonical_sha256,
            camera_si_scaling=scaling,
            train_scene_count=len(train_ids),
            validation_scene_count=len(validation_ids),
            curve_count=len(curves),
        )
        envelope = fit_scene_provider.VerifiedFitSceneEnvelopeV1(
            inventory=inventory,
            quality_adapter_binding_sha256=adapter.binding.canonical_sha256,
            held_provider_binding_sha256=held_provider.binding_sha256,
            held_provider_inventory_sha256=held_provider.inventory_sha256,
            held_scene_identity_inventory_sha256=inventory.canonical_sha256,
            held_provider_scene_count=held_provider.scene_count,
            verification_report_sha256=verification.canonical_sha256,
        )
        return ProductionSceneEvidenceV1(
            surface=surface,
            inventory=inventory,
            held_provider=held_provider,
            adapter=adapter,
            envelope=envelope,
            verification=verification,
        )
    except Exception:
        surface.close()
        raise


def _registered_values() -> tuple[str, ...]:
    values = (
        REGISTERED_HELD_CURVE_INVENTORY_SHA256,
        REGISTERED_HELD_PROVIDER_BINDING_SHA256,
        REGISTERED_QUALITY_ADAPTER_BINDING_SHA256,
        REGISTERED_VERIFICATION_REPORT_SHA256,
        REGISTERED_SCENE_ENVELOPE_SHA256,
    )
    if any(value is None for value in values):
        raise ProductionSceneEvidenceUnavailable(
            "reviewed production scene-evidence hashes are not registered"
        )
    return tuple(str(value) for value in values)


def load_production_scene_evidence(
    *, repository_root: Optional[Path] = None
) -> ProductionSceneEvidenceV1:
    """Load and verify the exact train-only scene/payload evidence bundle."""

    expected = _registered_values()
    root = _repo_root() if repository_root is None else Path(repository_root)
    root = root.resolve(strict=True)
    evidence = _assemble_unregistered(root)
    observed = (
        evidence.held_provider.inventory_sha256,
        evidence.held_provider.binding_sha256,
        evidence.adapter.binding.canonical_sha256,
        evidence.verification.canonical_sha256,
        evidence.envelope.canonical_sha256,
    )
    if observed != expected:
        evidence.close()
        labels = (
            "held curve inventory",
            "held provider binding",
            "quality adapter binding",
            "verification report",
            "scene envelope",
        )
        mismatch = next(
            label
            for label, actual, wanted in zip(labels, observed, expected)
            if actual != wanted
        )
        raise ProductionSceneEvidenceError(f"{mismatch} digest drifted")
    return evidence


__all__ = [
    "CameraSiScalingEvidenceV1",
    "ProductionSceneEvidenceError",
    "ProductionSceneEvidenceUnavailable",
    "ProductionSceneEvidenceV1",
    "ProductionSceneVerificationV1",
    "REGISTERED_HELD_CURVE_INVENTORY_SHA256",
    "REGISTERED_HELD_PROVIDER_BINDING_SHA256",
    "REGISTERED_QUALITY_ADAPTER_BINDING_SHA256",
    "REGISTERED_SCENE_ENVELOPE_SHA256",
    "REGISTERED_VERIFICATION_REPORT_SHA256",
    "load_production_scene_evidence",
]
