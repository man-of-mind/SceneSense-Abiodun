"""Reviewed scientific basis for the Run-4 learning objective.

The exact continuous-q surface was generated with a file named
``reward_spec.json``.  That file actually has two logically separate roles:

* it defines the perception-quality target ``Q_perc``; and
* it carries an old, inert scalar-reward hypothesis from the grid campaign.

Run-4 approves only the first role.  Its scalar reward is exclusively the
170-ms contract in :mod:`run4_contract`.  Keeping the two identities explicit
prevents the grid's historical 200-ms metadata or zero latency weight from
silently replacing the reviewed Run-4 reward.

Importing this module performs no I/O.  ``verify_quality_source`` is the sole
filesystem entry point and must be called by the future composite verifier.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    canonical_sha256,
)

from . import run4_contract as contract

QUALITY_SOURCE_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_quality_grid_v1/"
    "20260918_exact_continuous_q_grid_a1b_full/reward_spec.json"
)
QUALITY_SOURCE_FILE_SHA256 = (
    "d5d1e0d2d435076dd53c740f8b0e632620144194c32baf7d24db6d5043fc74d9"
)

QUALITY_DEFINITION_ID = "splitfusion_run4_qperc_v1"
QUALITY_APPROVAL_STATUS = (
    "APPROVED_FOR_RUN4_SEMI_EMPIRICAL_TRAINING_QUALITY_TARGET_ONLY"
)
QUALITY_APPROVAL_SCOPE = (
    "Q_perc definition and exact quality-surface values only; the source "
    "file's scalar reward, deadline, latency weight, failure reward, gamma "
    "and switching terms are explicitly not adopted"
)


class ScientificBasisError(ValueError):
    """The reviewed Run-4 scientific basis or its source drifted."""


@dataclass(frozen=True, slots=True)
class QualityDefinitionV1:
    localization_combiner: str = "WEIGHTED_GEOMETRIC_MEAN"
    tau_person_m: float = 1.2
    tau_vehicle_m: float = 1.0
    localization_person_weight: float = 0.6
    localization_vehicle_weight: float = 0.4
    segmentation_person_weight: float = 0.6
    segmentation_vehicle_weight: float = 0.4
    segmentation_reference_person_iou: float = 0.52789408
    segmentation_reference_vehicle_iou: float = 0.899012847
    segmentation_modulation_beta: float = 0.3

    def __post_init__(self) -> None:
        scalars = {
            name: value
            for name, value in self.to_dict().items()
            if name != "localization_combiner"
        }
        if any(not math.isfinite(float(value)) for value in scalars.values()):
            raise ScientificBasisError("quality definition contains non-finite data")
        if not math.isclose(
            self.localization_person_weight + self.localization_vehicle_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ScientificBasisError("localization weights must sum to one")
        if not math.isclose(
            self.segmentation_person_weight + self.segmentation_vehicle_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ScientificBasisError("segmentation weights must sum to one")
        if not 0.0 <= self.segmentation_modulation_beta <= 1.0:
            raise ScientificBasisError("segmentation beta must lie in [0,1]")
        if self.tau_person_m <= 0.0 or self.tau_vehicle_m <= 0.0:
            raise ScientificBasisError("localization tolerances must be positive")
        if (
            self.segmentation_reference_person_iou <= 0.0
            or self.segmentation_reference_vehicle_iou <= 0.0
        ):
            raise ScientificBasisError("segmentation references must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "localization_combiner": self.localization_combiner,
            "localization_person_weight": self.localization_person_weight,
            "localization_vehicle_weight": self.localization_vehicle_weight,
            "segmentation_modulation_beta": self.segmentation_modulation_beta,
            "segmentation_person_weight": self.segmentation_person_weight,
            "segmentation_reference_person_iou": (
                self.segmentation_reference_person_iou
            ),
            "segmentation_reference_vehicle_iou": (
                self.segmentation_reference_vehicle_iou
            ),
            "segmentation_vehicle_weight": self.segmentation_vehicle_weight,
            "tau_person_m": self.tau_person_m,
            "tau_vehicle_m": self.tau_vehicle_m,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_quality_definition_v1",
                "value": self.to_dict(),
            }
        )


QUALITY_DEFINITION = QualityDefinitionV1()

SCIENTIFIC_BASIS_DESCRIPTOR: Mapping[str, Any] = {
    "basis_id": "splitfusion_run4_scientific_basis_v1",
    "basis_version": 1,
    "quality": {
        "approval_scope": QUALITY_APPROVAL_SCOPE,
        "approval_status": QUALITY_APPROVAL_STATUS,
        "definition_id": QUALITY_DEFINITION_ID,
        "definition_sha256": QUALITY_DEFINITION.canonical_sha256,
        "source_file_sha256": QUALITY_SOURCE_FILE_SHA256,
        "source_relative_path": QUALITY_SOURCE_RELATIVE_PATH,
    },
    "scalar_reward": {
        "deadline_ms": contract.REWARD_DEADLINE_MS,
        "failure_reward": contract.REGISTERED_FAILURE_REWARD,
        "latency_weight": contract.REWARD_LATENCY_WEIGHT,
        "schema_id": contract.REWARD_SCHEMA_ID,
        "schema_sha256": contract.REWARD_SCHEMA_SHA256,
        "success": "Q_perc - 0.25 * action_open_to_feedback_ms / 170",
        "timeout_boundary": "latency strictly greater than 170 ms",
    },
    "separation_rule": (
        "the quality-grid source authenticates Q_perc values but never "
        "authorizes Run-4 scalar reward semantics"
    ),
}
SCIENTIFIC_BASIS_SHA256 = canonical_sha256(SCIENTIFIC_BASIS_DESCRIPTOR)


def _expect(document: Mapping[str, Any], name: str, expected: Any) -> None:
    if document.get(name) != expected:
        raise ScientificBasisError(
            f"quality source field {name!r} drifted: "
            f"expected {expected!r}, observed {document.get(name)!r}"
        )


def verify_quality_source(repo_root: Path) -> str:
    """Verify the pinned source bytes and only the approved quality fields."""

    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise ScientificBasisError("repo_root must be an absolute pathlib.Path")
    path = repo_root / QUALITY_SOURCE_RELATIVE_PATH
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScientificBasisError(f"cannot read quality source {path}") from exc
    observed = hashlib.sha256(raw).hexdigest()
    if observed != QUALITY_SOURCE_FILE_SHA256:
        raise ScientificBasisError(
            f"quality source SHA-256 drift: {observed} != "
            f"{QUALITY_SOURCE_FILE_SHA256}"
        )
    if not isinstance(document, Mapping):
        raise ScientificBasisError("quality source must be a JSON object")
    approved = QUALITY_DEFINITION
    for name, expected in (
        ("localization_combiner", approved.localization_combiner),
        ("tau_person_m", approved.tau_person_m),
        ("tau_vehicle_m", approved.tau_vehicle_m),
        ("w_loc_person", approved.localization_person_weight),
        ("w_loc_vehicle", approved.localization_vehicle_weight),
        ("w_seg_person", approved.segmentation_person_weight),
        ("w_seg_vehicle", approved.segmentation_vehicle_weight),
        (
            "seg_reference_person_iou",
            approved.segmentation_reference_person_iou,
        ),
        (
            "seg_reference_vehicle_iou",
            approved.segmentation_reference_vehicle_iou,
        ),
        (
            "segmentation_modulation_beta",
            approved.segmentation_modulation_beta,
        ),
    ):
        _expect(document, name, expected)
    return observed


__all__ = [
    "QUALITY_APPROVAL_SCOPE",
    "QUALITY_APPROVAL_STATUS",
    "QUALITY_DEFINITION",
    "QUALITY_DEFINITION_ID",
    "QUALITY_SOURCE_FILE_SHA256",
    "QUALITY_SOURCE_RELATIVE_PATH",
    "SCIENTIFIC_BASIS_DESCRIPTOR",
    "SCIENTIFIC_BASIS_SHA256",
    "QualityDefinitionV1",
    "ScientificBasisError",
    "verify_quality_source",
]
