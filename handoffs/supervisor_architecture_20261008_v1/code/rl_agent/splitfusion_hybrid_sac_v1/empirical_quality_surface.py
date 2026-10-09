"""Hash-bound, frame-conditioned empirical quality/payload surface.

This module is the deliberately small bridge between the completed exact
``12 mode x 11 q x 768 frame`` offline grid and a later policy environment.
It is *not* a scene-to-quality regressor.  For one hidden selected frame and
one mode it returns the measured row at a measured q node, or a piecewise
linear estimate between the two adjacent measured q nodes of that exact same
frame.  Consequently it never borrows quality from another scene, never
extrapolates, never enforces a monotone quality curve, and never turns an
undefined component into zero.

The policy-facing view contains only the two approved scene descriptors
(``camera_si`` and corrected ``radar_p40``), the requested action and the
resulting scalar evidence.  Frame/sample/episode/split identities, sampling
weights and endpoint row digests remain in a separate hidden record owned by
the environment.

Importing this module performs no filesystem access.  Loading is explicit,
read-only, and pinned to the one completed evidence bundle below.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from .action_contract import (
    Q_E4_MAX,
    Q_E4_MIN,
    Q_MAX,
    Q_MIN,
    round_half_up_q_e4,
)
from .offline_quality_grid.contract import (
    EXPECTED_GRID_ROWS,
    FIT_SELECTION_COUNT,
    HELD_SCENE_SELECTION_COUNT,
    Q_E4_GRID,
    TOTAL_SELECTED_FRAMES,
    canonical_json_bytes,
    mode_inventory,
)
from .offline_quality_grid.manifest import validate_completion_manifest
from .offline_quality_grid.quality import load_reward_spec
from .offline_quality_grid.selection import validate_selection_manifest

__all__ = [
    "BUNDLE_RELATIVE_PATH",
    "COMPLETE_FILE_SHA256",
    "DATABASE_FILE_SHA256",
    "EXACT_GRID_ROW_EVIDENCE",
    "MODELED_SAME_FRAME_EVIDENCE",
    "NO_RESIDUAL_INTERVAL_STATUS",
    "PAYLOAD_HELD_P95_RELATIVE_ERROR_MAX",
    "QUALITY_MEDIAN_ABSOLUTE_ERROR_MAX",
    "QUALITY_P90_ABSOLUTE_ERROR_MAX",
    "QUALITY_WEIGHTED_MEAN_ABSOLUTE_ERROR_MAX",
    "CorrectedP40Provider",
    "EmpiricalQualitySurface",
    "EndpointEvidence",
    "EvidenceBindingError",
    "GridInventoryError",
    "HiddenSurfaceRecord",
    "InvalidSurfaceQuery",
    "PayloadEstimate",
    "PolicySceneView",
    "PolicySurfaceView",
    "QualityComponent",
    "QualificationReport",
    "SplitAccessError",
    "SurfaceBinding",
    "SurfaceError",
    "SurfaceQueryResult",
    "load_empirical_quality_surface",
]


# ---------------------------------------------------------------------------
# Immutable evidence bindings.  Both the embedded document identities and the
# exact file bytes are pinned; replacing a file and recomputing its self-hash
# cannot authorize a different grid.
# ---------------------------------------------------------------------------

BUNDLE_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_quality_grid_v1/"
    "20260918_exact_continuous_q_grid_a1b_full"
)
COMPLETE_FILE_SHA256 = "de73546968bd6d632a4b48120c4980936111896206adeeafddd05efe79324c44"
RUN_MANIFEST_FILE_SHA256 = "8869d085e585bd4cb4d8231abdfea4b358095578e85a3cbfb811bf073bfc9b36"
RUN_MANIFEST_SHA256 = "9f4d53f15381ff538fe1775abc5cd38c84da0101b6263d69e2fef667a61e020e"
RUN_BINDING_SHA256 = "c43ccc1b5604eb2b20314c47842d6162986e2c5288f555297585434a2407fde3"
SELECTION_FILE_SHA256 = "a25b93d35a5bb70e062d275daf8767ce73a5ab7ba34079ab6a468bda5ffe4c4d"
SELECTION_MANIFEST_SHA256 = "9a03f415edf6f5482115bbec7644c9e6174d13dad9bd1505576e3e6ef93df905"
REWARD_SPEC_FILE_SHA256 = "d5d1e0d2d435076dd53c740f8b0e632620144194c32baf7d24db6d5043fc74d9"
DATABASE_FILE_SHA256 = "8b2c0754f135873cc339dc62cc7e0e09e02e3a85426d0027c2cd38b6a2a68c88"

EXACT_GRID_ROW_EVIDENCE = "MEASURED_EXACT_OFFLINE_GRID_ROW"
MODELED_SAME_FRAME_EVIDENCE = (
    "MODELED_PIECEWISE_LINEAR_FROM_EXACT_SAME_FRAME_ENDPOINTS"
)
NO_RESIDUAL_INTERVAL_STATUS = (
    "POINT_ESTIMATE_ONLY_NO_CALIBRATED_EMPIRICAL_RESIDUAL_INTERVAL"
)

# Preregistered before the held-scene qualification is inspected.
PAYLOAD_HELD_P95_RELATIVE_ERROR_MAX = 0.05
QUALITY_WEIGHTED_MEAN_ABSOLUTE_ERROR_MAX = 0.05
QUALITY_MEDIAN_ABSOLUTE_ERROR_MAX = 0.02
QUALITY_P90_ABSOLUTE_ERROR_MAX = 0.15

_QUALITY_FIELDS: Tuple[Tuple[str, str, Optional[str], str], ...] = (
    ("seg_vehicle_iou", "seg_vehicle_iou", "seg_vehicle_valid", "seg_vehicle_status"),
    ("seg_person_iou", "seg_person_iou", "seg_person_valid", "seg_person_status"),
    ("vehicle_recall", "loc_vehicle_recall", "loc_vehicle_valid", "loc_vehicle_status"),
    ("person_recall", "loc_person_recall", "loc_person_valid", "loc_person_status"),
    (
        "vehicle_xy_error_m",
        "loc_vehicle_matched_xy_median_m",
        None,
        "loc_vehicle_status",
    ),
    (
        "person_xy_error_m",
        "loc_person_matched_xy_median_m",
        None,
        "loc_person_status",
    ),
    ("q_seg", "q_seg", None, "quality_status"),
    ("q_loc", "q_loc", "quality_valid", "quality_status"),
    ("q_perc", "q_perc", "quality_valid", "quality_status"),
)
_QUALIFICATION_QUALITY_NAMES = ("q_seg", "q_loc", "q_perc")
_PAYLOAD_FIELDS = (
    "scientific_inner_payload_bytes",
    "total_transmitted_bytes",
    "udp_application_bytes",
)
_MODE_IDENTITY = {
    mode_id: (family, quantizer)
    for mode_id, family, quantizer in mode_inventory()
}


class SurfaceError(ValueError):
    """Base error for empirical-surface contract failures."""


class EvidenceBindingError(SurfaceError):
    """An immutable source artifact or corrected-P40 binding drifted."""


class GridInventoryError(SurfaceError):
    """The pinned database does not contain the declared exact grid."""


class InvalidSurfaceQuery(SurfaceError):
    """A mode, q, sample, or interpolation request is invalid."""


class SplitAccessError(SurfaceError):
    """Held-scene evidence was offered to the fit/training API."""


class CorrectedP40Provider(Protocol):
    """Minimal adapter expected from the independently repaired P40 sidecar.

    The surface does not derive or repair P40.  It snapshots this provider,
    demands exactly the selected 768 sample IDs, and records both the
    provider's immutable binding and a digest of the snapshot it consumed.
    """

    @property
    def binding_sha256(self) -> str: ...

    @property
    def sample_ids(self) -> Sequence[str]: ...

    def lookup(self, sample_id: str) -> float: ...


@dataclass(frozen=True, slots=True)
class SurfaceBinding:
    complete_file_sha256: str
    run_manifest_file_sha256: str
    run_manifest_sha256: str
    run_binding_sha256: str
    selection_file_sha256: str
    selection_manifest_sha256: str
    reward_spec_file_sha256: str
    database_file_sha256: str
    corrected_p40_binding_sha256: str
    corrected_p40_snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class PolicySceneView:
    """The complete scene state visible to the policy from this surface."""

    camera_si: float
    radar_p40: float

    def to_dict(self) -> Dict[str, float]:
        return {"camera_si": self.camera_si, "radar_p40": self.radar_p40}


@dataclass(frozen=True, slots=True)
class QualityComponent:
    name: str
    value: Optional[float]
    valid: bool
    status: str

    def __post_init__(self) -> None:
        if self.value is None:
            if self.valid:
                raise GridInventoryError(f"{self.name}: null component cannot be valid")
        elif not self.valid or not math.isfinite(self.value):
            raise GridInventoryError(f"{self.name}: numeric component must be finite and valid")


@dataclass(frozen=True, slots=True)
class PayloadEstimate:
    scientific_inner_payload_bytes: int | float
    total_transmitted_bytes: int | float
    udp_application_bytes: int | float
    datagram_count: Optional[int]
    discrete_count_status: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "datagram_count": self.datagram_count,
            "discrete_count_status": self.discrete_count_status,
            "scientific_inner_payload_bytes": self.scientific_inner_payload_bytes,
            "total_transmitted_bytes": self.total_transmitted_bytes,
            "udp_application_bytes": self.udp_application_bytes,
        }


@dataclass(frozen=True, slots=True)
class PolicySurfaceView:
    """Policy-safe surface output; deliberately contains no frame identity."""

    mode_id: int
    family: str
    quantizer: str
    q_e4: int
    q_exec: float
    scene: PolicySceneView
    payload: PayloadEstimate
    quality: Tuple[QualityComponent, ...]
    evidence_status: str
    residual_interval_status: str = NO_RESIDUAL_INTERVAL_STATUS

    def component(self, name: str) -> QualityComponent:
        matches = tuple(item for item in self.quality if item.name == name)
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def to_dict(self) -> Dict[str, Any]:
        """Canonical policy handoff with no hidden identifiers or GT counts."""

        return {
            "evidence_status": self.evidence_status,
            "family": self.family,
            "mode_id": self.mode_id,
            "payload": self.payload.as_dict(),
            "q_e4": self.q_e4,
            "q_exec": self.q_exec,
            "quality": [
                {
                    "name": item.name,
                    "status": item.status,
                    "valid": item.valid,
                    "value": item.value,
                }
                for item in self.quality
            ],
            "quantizer": self.quantizer,
            "residual_interval_status": self.residual_interval_status,
            "scene": self.scene.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EndpointEvidence:
    q_e4: int
    row_key_sha256: str
    row_sha256: str
    datagram_count: int
    quality_valid: bool
    quality_status: str


@dataclass(frozen=True, slots=True)
class HiddenSurfaceRecord:
    """Environment-only provenance; never pass this object into the policy."""

    episode_id: str
    sample_id: str
    frame_id: int
    grid_split: str
    selection_rank_within_split: int
    inclusion_probability: float
    sampling_weight: float
    endpoint_evidence: Tuple[EndpointEvidence, ...]


@dataclass(frozen=True, slots=True)
class SurfaceQueryResult:
    policy: PolicySurfaceView
    hidden: HiddenSurfaceRecord


@dataclass(frozen=True, slots=True)
class QualificationReport:
    """Immutable canonical report; ``document()`` returns a defensive copy."""

    canonical_json: bytes
    report_sha256: str

    def document(self) -> Dict[str, Any]:
        return json.loads(self.canonical_json.decode("utf-8"))


@dataclass(frozen=True, slots=True)
class _Frame:
    episode_id: str
    sample_id: str
    frame_id: int
    grid_split: str
    selection_rank_within_split: int
    inclusion_probability: float
    sampling_weight: float
    camera_si: float


@dataclass(frozen=True, slots=True)
class _GridRow:
    sample_id: str
    grid_split: str
    mode_id: int
    family: str
    quantizer: str
    q_e4: int
    scientific_inner_payload_bytes: int
    total_transmitted_bytes: int
    udp_application_bytes: int
    datagram_count: int
    quality: Tuple[QualityComponent, ...]
    quality_valid: bool
    quality_status: str
    row_key_sha256: str
    row_sha256: str

    def component(self, name: str) -> QualityComponent:
        for component in self.quality:
            if component.name == name:
                return component
        raise KeyError(name)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvidenceBindingError(f"cannot read pinned artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _read_pinned(path: Path, expected: str, label: str) -> bytes:
    observed = _sha256_file(path)
    if observed != expected:
        raise EvidenceBindingError(
            f"{label} SHA-256 drift: expected {expected}, observed {observed}"
        )
    return path.read_bytes()


def _json_document(raw: bytes, label: str) -> Dict[str, Any]:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBindingError(f"{label} is not valid UTF-8 JSON") from exc
    if type(document) is not dict:
        raise EvidenceBindingError(f"{label} must be a JSON object")
    return document


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise GridInventoryError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise GridInventoryError(f"{label} must be finite")
    return result


def _digest_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EvidenceBindingError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise EvidenceBindingError(f"{label} must be hexadecimal") from exc
    return value


def _quality_components(row: Mapping[str, Any]) -> Tuple[QualityComponent, ...]:
    result = []
    for name, value_field, valid_field, status_field in _QUALITY_FIELDS:
        value_raw = row[value_field]
        value = None if value_raw is None else _finite(value_raw, value_field)
        valid = value is not None if valid_field is None else bool(row[valid_field])
        status = str(row[status_field])
        if value is None and valid_field is not None and valid:
            # A localization class may be valid with zero TP, in which case its
            # matched-error summary is intentionally undefined.  Recall does
            # not take this path; q fields do not either.
            valid = False
            status = f"UNDEFINED_NUMERIC_COMPONENT__{status}"
        result.append(QualityComponent(name, value, valid, status))
    return tuple(result)


def _project_row(raw: bytes | str) -> _GridRow:
    try:
        row = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GridInventoryError("stored row_json is malformed") from exc
    if type(row) is not dict:
        raise GridInventoryError("stored row_json is not an object")
    return _GridRow(
        sample_id=str(row["sample_id"]),
        grid_split=str(row["grid_split"]),
        mode_id=int(row["mode_id"]),
        family=str(row["family"]),
        quantizer=str(row["quantizer"]),
        q_e4=int(row["q_e4"]),
        scientific_inner_payload_bytes=int(row["scientific_inner_payload_bytes"]),
        total_transmitted_bytes=int(row["total_transmitted_bytes"]),
        udp_application_bytes=int(row["udp_application_bytes"]),
        datagram_count=int(row["datagram_count"]),
        quality=_quality_components(row),
        quality_valid=bool(row["quality_valid"]),
        quality_status=str(row["quality_status"]),
        row_key_sha256=str(row["row_key_sha256"]),
        row_sha256=str(row["row_sha256"]),
    )


def _endpoint(row: _GridRow) -> EndpointEvidence:
    return EndpointEvidence(
        q_e4=row.q_e4,
        row_key_sha256=row.row_key_sha256,
        row_sha256=row.row_sha256,
        datagram_count=row.datagram_count,
        quality_valid=row.quality_valid,
        quality_status=row.quality_status,
    )


def _validate_query_q(q: Any) -> int:
    if isinstance(q, bool) or not isinstance(q, numbers.Real):
        raise InvalidSurfaceQuery("q must be a finite real number")
    value = float(q)
    if not math.isfinite(value) or not Q_MIN <= value <= Q_MAX:
        raise InvalidSurfaceQuery(f"q must lie in [{Q_MIN}, {Q_MAX}]")
    return round_half_up_q_e4(value)


def _validate_q_e4(q_e4: Any) -> int:
    if isinstance(q_e4, bool) or not isinstance(q_e4, numbers.Integral):
        raise InvalidSurfaceQuery("q_e4 must be an exact integer")
    value = int(q_e4)
    if not Q_E4_MIN <= value <= Q_E4_MAX:
        raise InvalidSurfaceQuery(f"q_e4 must lie in [{Q_E4_MIN}, {Q_E4_MAX}]")
    return value


def _lerp(lower: float, upper: float, alpha: float) -> float:
    return lower + alpha * (upper - lower)


def _surface_from_rows(
    *, frame: _Frame, radar_p40: float, rows: Sequence[_GridRow], q_e4: int
) -> SurfaceQueryResult:
    """Pure exact/bracket interpolation used by the store and unit tests."""

    if len(rows) != len(Q_E4_GRID) or tuple(row.q_e4 for row in rows) != Q_E4_GRID:
        raise GridInventoryError("same-frame mode rows do not cover the exact q grid")
    exact = next((row for row in rows if row.q_e4 == q_e4), None)
    if exact is not None:
        payload = PayloadEstimate(
            exact.scientific_inner_payload_bytes,
            exact.total_transmitted_bytes,
            exact.udp_application_bytes,
            exact.datagram_count,
            "EXACT_MEASURED_DATAGRAM_COUNT",
        )
        quality = exact.quality
        endpoints = (_endpoint(exact),)
        evidence_status = EXACT_GRID_ROW_EVIDENCE
    else:
        lower = max((row for row in rows if row.q_e4 < q_e4), key=lambda row: row.q_e4)
        upper = min((row for row in rows if row.q_e4 > q_e4), key=lambda row: row.q_e4)
        alpha = (q_e4 - lower.q_e4) / (upper.q_e4 - lower.q_e4)
        modeled_payload: Dict[str, float] = {}
        for name in _PAYLOAD_FIELDS:
            low = float(getattr(lower, name))
            high = float(getattr(upper, name))
            if not low > high:
                raise GridInventoryError(
                    f"payload is not strictly decreasing at interpolation endpoints: {name}"
                )
            value = _lerp(low, high, alpha)
            if not high < value < low:
                raise GridInventoryError(f"modeled payload escaped its strict bracket: {name}")
            modeled_payload[name] = value
        payload = PayloadEstimate(
            modeled_payload["scientific_inner_payload_bytes"],
            modeled_payload["total_transmitted_bytes"],
            modeled_payload["udp_application_bytes"],
            None,
            "NOT_INTERPOLATED_ENDPOINT_COUNTS_RETAINED_IN_HIDDEN_PROVENANCE",
        )
        modeled_quality = []
        for lower_component, upper_component in zip(lower.quality, upper.quality):
            if lower_component.name != upper_component.name:
                raise GridInventoryError("quality component order drift across endpoints")
            name = lower_component.name
            if lower_component.valid and upper_component.valid:
                assert lower_component.value is not None and upper_component.value is not None
                modeled_quality.append(
                    QualityComponent(
                        name,
                        _lerp(lower_component.value, upper_component.value, alpha),
                        True,
                        MODELED_SAME_FRAME_EVIDENCE,
                    )
                )
            else:
                statuses = f"{lower_component.status}|{upper_component.status}"
                modeled_quality.append(
                    QualityComponent(
                        name,
                        None,
                        False,
                        "UNDEFINED_ENDPOINT_SUPPORT_NOT_ZERO_IMPUTED__" + statuses,
                    )
                )
        quality = tuple(modeled_quality)
        endpoints = (_endpoint(lower), _endpoint(upper))
        evidence_status = MODELED_SAME_FRAME_EVIDENCE

    mode_id = rows[0].mode_id
    expected_mode = _MODE_IDENTITY.get(mode_id)
    if expected_mode != (rows[0].family, rows[0].quantizer):
        raise GridInventoryError("mode identity drift")
    policy = PolicySurfaceView(
        mode_id=mode_id,
        family=rows[0].family,
        quantizer=rows[0].quantizer,
        q_e4=q_e4,
        q_exec=q_e4 / 10_000.0,
        scene=PolicySceneView(frame.camera_si, radar_p40),
        payload=payload,
        quality=quality,
        evidence_status=evidence_status,
    )
    hidden = HiddenSurfaceRecord(
        episode_id=frame.episode_id,
        sample_id=frame.sample_id,
        frame_id=frame.frame_id,
        grid_split=frame.grid_split,
        selection_rank_within_split=frame.selection_rank_within_split,
        inclusion_probability=frame.inclusion_probability,
        sampling_weight=frame.sampling_weight,
        endpoint_evidence=endpoints,
    )
    return SurfaceQueryResult(policy=policy, hidden=hidden)


def _weighted_summary(values: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "weight_sum": 0.0,
            "weighted_mae": None,
            "weighted_mean": None,
            "weighted_median": None,
            "weighted_p90": None,
            "weighted_p95": None,
            "maximum": None,
        }
    ordered = sorted((float(value), float(weight)) for value, weight in values)
    total_weight = sum(weight for _, weight in ordered)
    if not total_weight > 0:
        raise GridInventoryError("qualification weights must sum to a positive value")

    def quantile(probability: float) -> float:
        target = probability * total_weight
        cumulative = 0.0
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= target:
                return value
        return ordered[-1][0]

    weighted_mean = sum(v * w for v, w in ordered) / total_weight
    return {
        "count": len(ordered),
        "maximum": ordered[-1][0],
        "weight_sum": total_weight,
        # These inputs are non-negative absolute errors, hence their weighted
        # mean is the weighted mean absolute error (MAE).  Retain both labels
        # so the report is unambiguous to readers and machine consumers.
        "weighted_mae": weighted_mean,
        "weighted_mean": weighted_mean,
        "weighted_median": quantile(0.50),
        "weighted_p90": quantile(0.90),
        "weighted_p95": quantile(0.95),
    }


class EmpiricalQualitySurface:
    """Read-only access to exact and same-frame interpolated grid evidence."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        frames: Mapping[str, _Frame],
        corrected_p40: Mapping[str, float],
        binding: SurfaceBinding,
        database_path: Path,
    ) -> None:
        self._connection = connection
        frame_snapshot = dict(frames)
        self._fit_frames = MappingProxyType(
            {key: value for key, value in frame_snapshot.items() if value.grid_split == "fit"}
        )
        self._held_frames = MappingProxyType(
            {
                key: value
                for key, value in frame_snapshot.items()
                if value.grid_split == "held_scene"
            }
        )
        # Combined inventory is environment-private and exists only for the
        # immutable database audit/qualification.  Training lookup below uses
        # the physically separate fit mapping and cannot see held keys.
        self._frames = MappingProxyType(frame_snapshot)
        self._corrected_p40 = MappingProxyType(dict(corrected_p40))
        self.binding = binding
        self.database_path = Path(database_path)
        self._closed = False
        self._row_cache: Dict[Tuple[str, int], Tuple[_GridRow, ...]] = {}
        self._qualification_cache: Optional[QualificationReport] = None

    @classmethod
    def load(
        cls, bundle_dir: Path, corrected_p40_provider: CorrectedP40Provider
    ) -> "EmpiricalQualitySurface":
        bundle = Path(bundle_dir).resolve(strict=True)
        complete_raw = _read_pinned(bundle / "COMPLETE.json", COMPLETE_FILE_SHA256, "COMPLETE")
        manifest_raw = _read_pinned(
            bundle / "run_manifest.json", RUN_MANIFEST_FILE_SHA256, "run manifest"
        )
        selection_raw = _read_pinned(
            bundle / "selection_manifest.json", SELECTION_FILE_SHA256, "selection manifest"
        )
        _read_pinned(bundle / "reward_spec.json", REWARD_SPEC_FILE_SHA256, "reward spec")
        database_path = bundle / "quality_rows.sqlite3"
        database_digest = _sha256_file(database_path)
        if database_digest != DATABASE_FILE_SHA256:
            raise EvidenceBindingError(
                "quality database SHA-256 drift: "
                f"expected {DATABASE_FILE_SHA256}, observed {database_digest}"
            )

        complete = _json_document(complete_raw, "COMPLETE")
        manifest = _json_document(manifest_raw, "run manifest")
        selection = _json_document(selection_raw, "selection manifest")
        validate_completion_manifest(manifest)
        validate_selection_manifest(selection)
        load_reward_spec(bundle / "reward_spec.json", REWARD_SPEC_FILE_SHA256)
        required_complete = {
            "schema": "splitfusion_exact_offline_quality_grid_complete_v1",
            "status": "COMPLETE",
            "rows": EXPECTED_GRID_ROWS,
            "expected_rows": EXPECTED_GRID_ROWS,
            "run_manifest_file_sha256": RUN_MANIFEST_FILE_SHA256,
            "run_manifest_sha256": RUN_MANIFEST_SHA256,
            "run_binding_sha256": RUN_BINDING_SHA256,
        }
        for key, value in required_complete.items():
            if complete.get(key) != value:
                raise EvidenceBindingError(f"COMPLETE binding drift at {key}")
        if (
            manifest.get("run_manifest_sha256") != RUN_MANIFEST_SHA256
            or manifest.get("run_binding_sha256") != RUN_BINDING_SHA256
            or manifest.get("selection_manifest_sha256") != SELECTION_MANIFEST_SHA256
            or manifest.get("reward_spec_sha256") != REWARD_SPEC_FILE_SHA256
            or manifest.get("execution_status") != "COMPLETE"
            or manifest.get("expected_rows") != EXPECTED_GRID_ROWS
        ):
            raise EvidenceBindingError("run manifest immutable binding drift")
        artifacts = manifest.get("artifact_sha256")
        if not isinstance(artifacts, Mapping) or any(
            artifacts.get(name) != digest
            for name, digest in (
                ("quality_rows.sqlite3", DATABASE_FILE_SHA256),
                ("selection_manifest.json", SELECTION_FILE_SHA256),
                ("reward_spec.json", REWARD_SPEC_FILE_SHA256),
            )
        ):
            raise EvidenceBindingError("run manifest artifact inventory drift")
        if selection.get("selection_manifest_sha256") != SELECTION_MANIFEST_SHA256:
            raise EvidenceBindingError("selection manifest embedded digest drift")

        frames: Dict[str, _Frame] = {}
        for row in selection["selected_frames"]:
            sample_id = str(row["sample_id"])
            camera_si = _finite(row["camera_si"], f"{sample_id}.camera_si")
            if row.get("camera_si_valid") is not True or camera_si < 0:
                raise GridInventoryError(f"{sample_id}: invalid camera SI")
            frame = _Frame(
                episode_id=str(row["episode_id"]),
                sample_id=sample_id,
                frame_id=int(row["frame_id"]),
                grid_split=str(row["grid_split"]),
                selection_rank_within_split=int(row["selection_rank_within_split"]),
                inclusion_probability=_finite(
                    row["inclusion_probability"], f"{sample_id}.inclusion_probability"
                ),
                sampling_weight=_finite(row["sampling_weight"], f"{sample_id}.sampling_weight"),
                camera_si=camera_si,
            )
            if frame.grid_split not in ("fit", "held_scene"):
                raise GridInventoryError(f"{sample_id}: unknown grid split")
            if not 0 < frame.inclusion_probability <= 1 or frame.sampling_weight <= 0:
                raise GridInventoryError(f"{sample_id}: invalid sampling design values")
            if sample_id in frames:
                raise GridInventoryError(f"duplicate selected sample: {sample_id}")
            frames[sample_id] = frame
        if (
            len(frames) != TOTAL_SELECTED_FRAMES
            or sum(frame.grid_split == "fit" for frame in frames.values()) != FIT_SELECTION_COUNT
            or sum(frame.grid_split == "held_scene" for frame in frames.values())
            != HELD_SCENE_SELECTION_COUNT
        ):
            raise GridInventoryError("selected frame/split cardinality drift")

        p40_binding = _digest_hex(
            getattr(corrected_p40_provider, "binding_sha256", None),
            "corrected P40 provider binding",
        )
        try:
            provider_ids_first = tuple(corrected_p40_provider.sample_ids)
            provider_ids_second = tuple(corrected_p40_provider.sample_ids)
        except Exception as exc:  # provider boundary: normalize to one fail-closed type
            raise EvidenceBindingError(f"cannot enumerate corrected P40 provider: {exc}") from exc
        if provider_ids_first != provider_ids_second or len(set(provider_ids_first)) != len(
            provider_ids_first
        ):
            raise EvidenceBindingError("corrected P40 provider is mutable or duplicated")
        if set(provider_ids_first) != set(frames):
            missing = len(set(frames).difference(provider_ids_first))
            foreign = len(set(provider_ids_first).difference(frames))
            raise EvidenceBindingError(
                f"corrected P40 inventory mismatch: missing={missing}, foreign={foreign}"
            )
        p40: Dict[str, float] = {}
        for sample_id in sorted(frames):
            try:
                first = _finite(corrected_p40_provider.lookup(sample_id), f"{sample_id}.radar_p40")
                second = _finite(corrected_p40_provider.lookup(sample_id), f"{sample_id}.radar_p40")
            except Exception as exc:
                if isinstance(exc, SurfaceError):
                    raise
                raise EvidenceBindingError(f"corrected P40 lookup failed for {sample_id}: {exc}") from exc
            if first != second or not 0.0 <= first <= 1.0:
                raise EvidenceBindingError(
                    f"corrected P40 is mutable or outside [0,1] for {sample_id}"
                )
            p40[sample_id] = first
        p40_snapshot_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "provider_binding_sha256": p40_binding,
                    "record": "splitfusion_corrected_p40_surface_snapshot_v1",
                    "values": [[sample_id, p40[sample_id]] for sample_id in sorted(p40)],
                }
            )
        ).hexdigest()

        uri = f"file:{database_path}?mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=30.0)
            connection.execute("PRAGMA query_only=ON")
        except sqlite3.Error as exc:
            raise EvidenceBindingError(f"cannot open immutable quality database: {exc}") from exc
        try:
            metadata = connection.execute(
                "SELECT run_manifest_sha256, run_binding_sha256, "
                "selection_manifest_sha256, reward_spec_sha256 FROM metadata "
                "WHERE singleton=1"
            ).fetchall()
            if metadata != [
                (
                    manifest["initial_run_manifest_sha256"],
                    RUN_BINDING_SHA256,
                    SELECTION_MANIFEST_SHA256,
                    REWARD_SPEC_FILE_SHA256,
                )
            ]:
                raise EvidenceBindingError("SQLite metadata binding drift")
            binding = SurfaceBinding(
                COMPLETE_FILE_SHA256,
                RUN_MANIFEST_FILE_SHA256,
                RUN_MANIFEST_SHA256,
                RUN_BINDING_SHA256,
                SELECTION_FILE_SHA256,
                SELECTION_MANIFEST_SHA256,
                REWARD_SPEC_FILE_SHA256,
                DATABASE_FILE_SHA256,
                p40_binding,
                p40_snapshot_sha,
            )
            surface = cls(
                connection=connection,
                frames=frames,
                corrected_p40=p40,
                binding=binding,
                database_path=database_path,
            )
            surface._audit_inventory_and_payload_monotonicity()
            return surface
        except Exception:
            connection.close()
            raise

    @property
    def fit_frame_count(self) -> int:
        return len(self._fit_frames)

    @property
    def held_scene_frame_count(self) -> int:
        return len(self._held_frames)

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True
            self._row_cache.clear()

    def __enter__(self) -> "EmpiricalQualitySurface":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise SurfaceError("empirical surface is closed")

    def _audit_inventory_and_payload_monotonicity(self) -> None:
        self._ensure_open()
        count = int(self._connection.execute("SELECT COUNT(*) FROM quality_rows").fetchone()[0])
        if count != EXPECTED_GRID_ROWS:
            raise GridInventoryError(f"quality row count {count} != {EXPECTED_GRID_ROWS}")
        seen_groups = 0
        current_key: Optional[Tuple[str, int]] = None
        rows: list[_GridRow] = []

        def finish_group() -> None:
            nonlocal seen_groups, rows
            if current_key is None:
                return
            if tuple(row.q_e4 for row in rows) != Q_E4_GRID:
                raise GridInventoryError(f"q inventory drift for {current_key}")
            for name in _PAYLOAD_FIELDS:
                values = [getattr(row, name) for row in rows]
                if any(lower <= upper for lower, upper in zip(values, values[1:])):
                    raise GridInventoryError(
                        f"payload is not strictly nonincreasing for {current_key}: {name}"
                    )
            seen_groups += 1
            rows = []

        cursor = self._connection.execute(
            "SELECT row_json FROM quality_rows ORDER BY sample_id, mode_id, q_e4"
        )
        for (raw,) in cursor:
            row = _project_row(raw)
            key = (row.sample_id, row.mode_id)
            if key != current_key:
                finish_group()
                current_key = key
            frame = self._frames.get(row.sample_id)
            if frame is None or row.grid_split != frame.grid_split:
                raise GridInventoryError(f"row has foreign sample/split: {row.sample_id}")
            if _MODE_IDENTITY.get(row.mode_id) != (row.family, row.quantizer):
                raise GridInventoryError(f"row mode identity drift: {key}")
            rows.append(row)
        finish_group()
        expected_groups = TOTAL_SELECTED_FRAMES * len(_MODE_IDENTITY)
        if seen_groups != expected_groups:
            raise GridInventoryError(
                f"frame-mode group count {seen_groups} != {expected_groups}"
            )

    def _rows_for(self, sample_id: str, mode_id: int) -> Tuple[_GridRow, ...]:
        self._ensure_open()
        key = (sample_id, mode_id)
        cached = self._row_cache.get(key)
        if cached is not None:
            return cached
        if mode_id not in _MODE_IDENTITY:
            raise InvalidSurfaceQuery(f"unknown mode_id {mode_id}")
        rows = tuple(
            _project_row(raw)
            for (raw,) in self._connection.execute(
                "SELECT row_json FROM quality_rows WHERE sample_id=? AND mode_id=? "
                "ORDER BY q_e4",
                key,
            )
        )
        if len(rows) != len(Q_E4_GRID) or tuple(row.q_e4 for row in rows) != Q_E4_GRID:
            raise GridInventoryError(f"incomplete q support for sample/mode {key}")
        # Small bounded manual cache: deterministic FIFO insertion order.
        if len(self._row_cache) >= 128:
            self._row_cache.pop(next(iter(self._row_cache)))
        self._row_cache[key] = rows
        return rows

    def _query(self, sample_id: str, mode_id: int, q_e4: int) -> SurfaceQueryResult:
        frame = self._frames.get(sample_id)
        if frame is None:
            raise InvalidSurfaceQuery(f"unknown selected sample_id: {sample_id}")
        rows = self._rows_for(sample_id, mode_id)
        return _surface_from_rows(
            frame=frame,
            radar_p40=self._corrected_p40[sample_id],
            rows=rows,
            q_e4=q_e4,
        )

    def query_fit(self, sample_id: str, mode_id: int, q: Any) -> SurfaceQueryResult:
        """Training API.  It categorically refuses held-scene samples."""

        frame = self._fit_frames.get(sample_id)
        if frame is None and sample_id in self._held_frames:
            raise SplitAccessError("held_scene evidence is evaluation-only")
        if frame is None:
            raise InvalidSurfaceQuery(f"unknown fit sample_id: {sample_id}")
        return self._query(sample_id, mode_id, _validate_query_q(q))

    def query_fit_q_e4(
        self, sample_id: str, mode_id: int, q_e4: Any
    ) -> SurfaceQueryResult:
        frame = self._fit_frames.get(sample_id)
        if frame is None and sample_id in self._held_frames:
            raise SplitAccessError("held_scene evidence is evaluation-only")
        if frame is None:
            raise InvalidSurfaceQuery(f"unknown fit sample_id: {sample_id}")
        return self._query(sample_id, mode_id, _validate_q_e4(q_e4))

    def evaluate_held(
        self, sample_id: str, mode_id: int, q: Any
    ) -> SurfaceQueryResult:
        """Explicit evaluation-only access to the untouched held-scene split."""

        frame = self._held_frames.get(sample_id)
        if frame is None and sample_id in self._fit_frames:
            raise SplitAccessError("evaluate_held requires a held_scene sample")
        if frame is None:
            raise InvalidSurfaceQuery(f"unknown held_scene sample_id: {sample_id}")
        return self._query(sample_id, mode_id, _validate_query_q(q))

    def qualify_leave_one_q_out(self) -> QualificationReport:
        """Grouped same-frame LOQO qualification over fit and held_scene.

        Every interior measured q node is predicted from its immediately
        adjacent measured nodes for the same frame and mode.  Selection weights
        are used for the mean and all reported empirical quantiles.
        """

        self._ensure_open()
        if self._qualification_cache is not None:
            return self._qualification_cache
        quality_errors: Dict[Tuple[str, int, int, str], list[Tuple[float, float]]] = {}
        payload_relative: Dict[Tuple[str, int, int, str], list[Tuple[float, float]]] = {}
        payload_absolute: Dict[Tuple[str, int, int, str], list[Tuple[float, float]]] = {}
        undefined: Dict[Tuple[str, int, int, str], int] = {}
        global_quality: Dict[Tuple[str, str], list[Tuple[float, float]]] = {}
        global_payload: Dict[Tuple[str, str], list[Tuple[float, float]]] = {}

        current_key: Optional[Tuple[str, int]] = None
        rows: list[_GridRow] = []

        def assess_group() -> None:
            if current_key is None:
                return
            if tuple(row.q_e4 for row in rows) != Q_E4_GRID:
                raise GridInventoryError(f"qualification q inventory drift for {current_key}")
            frame = self._frames[current_key[0]]
            weight = frame.sampling_weight
            for index in range(1, len(rows) - 1):
                lower, actual, upper = rows[index - 1], rows[index], rows[index + 1]
                alpha = (actual.q_e4 - lower.q_e4) / (upper.q_e4 - lower.q_e4)
                base = (frame.grid_split, actual.mode_id, actual.q_e4)
                for field in _PAYLOAD_FIELDS:
                    prediction = _lerp(
                        float(getattr(lower, field)), float(getattr(upper, field)), alpha
                    )
                    observed = float(getattr(actual, field))
                    absolute = abs(prediction - observed)
                    relative = absolute / observed
                    payload_absolute.setdefault((*base, field), []).append((absolute, weight))
                    payload_relative.setdefault((*base, field), []).append((relative, weight))
                    global_payload.setdefault((frame.grid_split, field), []).append(
                        (relative, weight)
                    )
                for field in _QUALIFICATION_QUALITY_NAMES:
                    low = lower.component(field)
                    target = actual.component(field)
                    high = upper.component(field)
                    key = (*base, field)
                    if low.valid and target.valid and high.valid:
                        assert low.value is not None and target.value is not None and high.value is not None
                        error = abs(_lerp(low.value, high.value, alpha) - target.value)
                        quality_errors.setdefault(key, []).append((error, weight))
                        global_quality.setdefault((frame.grid_split, field), []).append(
                            (error, weight)
                        )
                    else:
                        undefined[key] = undefined.get(key, 0) + 1

        for (raw,) in self._connection.execute(
            "SELECT row_json FROM quality_rows ORDER BY sample_id, mode_id, q_e4"
        ):
            row = _project_row(raw)
            key = (row.sample_id, row.mode_id)
            if key != current_key:
                assess_group()
                rows = []
                current_key = key
            rows.append(row)
        assess_group()

        slices = []
        for split in ("fit", "held_scene"):
            for mode_id in sorted(_MODE_IDENTITY):
                for q_e4 in Q_E4_GRID[1:-1]:
                    base = (split, mode_id, q_e4)
                    slices.append(
                        {
                            "grid_split": split,
                            "mode_id": mode_id,
                            "q_e4": q_e4,
                            "payload_absolute_error": {
                                field: _weighted_summary(payload_absolute.get((*base, field), ()))
                                for field in _PAYLOAD_FIELDS
                            },
                            "payload_relative_error": {
                                field: _weighted_summary(payload_relative.get((*base, field), ()))
                                for field in _PAYLOAD_FIELDS
                            },
                            "quality_absolute_error": {
                                field: _weighted_summary(quality_errors.get((*base, field), ()))
                                for field in _QUALIFICATION_QUALITY_NAMES
                            },
                            "quality_undefined_count": {
                                field: undefined.get((*base, field), 0)
                                for field in _QUALIFICATION_QUALITY_NAMES
                            },
                        }
                    )

        quality_aggregate: Dict[str, Dict[str, Any]] = {}
        payload_aggregate: Dict[str, Dict[str, Any]] = {}
        quality_gate_results = []
        quality_component_gates: Dict[str, Dict[str, bool]] = {}
        for split in ("fit", "held_scene"):
            quality_aggregate[split] = {}
            payload_aggregate[split] = {}
            quality_component_gates[split] = {}
            for field in _QUALIFICATION_QUALITY_NAMES:
                summary = _weighted_summary(global_quality.get((split, field), ()))
                quality_aggregate[split][field] = summary
                component_passed = bool(
                    summary["count"] > 0
                    and summary["weighted_mean"] <= QUALITY_WEIGHTED_MEAN_ABSOLUTE_ERROR_MAX
                    and summary["weighted_median"] <= QUALITY_MEDIAN_ABSOLUTE_ERROR_MAX
                    and summary["weighted_p90"] <= QUALITY_P90_ABSOLUTE_ERROR_MAX
                )
                quality_component_gates[split][field] = component_passed
                quality_gate_results.append(component_passed)
            for field in _PAYLOAD_FIELDS:
                payload_aggregate[split][field] = _weighted_summary(
                    global_payload.get((split, field), ())
                )
        held_payload_gate = all(
            payload_aggregate["held_scene"][field]["count"] > 0
            and payload_aggregate["held_scene"][field]["weighted_p95"]
            <= PAYLOAD_HELD_P95_RELATIVE_ERROR_MAX
            for field in _PAYLOAD_FIELDS
        )
        monotone_gate = True  # exact full-grid audit is mandatory at construction
        full_component_qualified = bool(all(quality_gate_results))
        reward_target_q_perc_qualified = all(
            quality_component_gates[split]["q_perc"]
            for split in ("fit", "held_scene")
        )
        passed = bool(full_component_qualified and held_payload_gate and monotone_gate)
        document = {
            "binding": {
                name: getattr(self.binding, name)
                for name in self.binding.__dataclass_fields__
            },
            "gate_results": {
                "all_quality_fit_and_held": full_component_qualified,
                "full_component_surface_qualified": full_component_qualified,
                "held_payload_p95": held_payload_gate,
                "payload_anchor_monotone_fraction": 1.0,
                "payload_anchor_monotone": monotone_gate,
                "quality_component_by_split": quality_component_gates,
                "reward_target_q_perc_qualified": reward_target_q_perc_qualified,
            },
            "method": (
                "leave each interior q node out and linearly interpolate from the "
                "immediately adjacent exact nodes of the same hidden frame and mode"
            ),
            "payload_relative_error_aggregate": payload_aggregate,
            "quality_absolute_error_aggregate": quality_aggregate,
            "q_perc_scope": (
                "Qualified only as the scalar target derived by the exact pinned "
                f"RewardSpecV1 SHA-256 {REWARD_SPEC_FILE_SHA256}. It does not qualify "
                "off-anchor beta, class-weight, tolerance, or localization-combiner "
                "recalibration; those require rederivation from exact evidence."
            ),
            "record": "splitfusion_empirical_quality_surface_loqo_v1",
            "slices_by_split_mode_q": slices,
            "status": "PASS" if passed else "FAIL",
            "thresholds_preregistered_before_held_inspection": {
                "payload_held_weighted_p95_relative_error_max": PAYLOAD_HELD_P95_RELATIVE_ERROR_MAX,
                "quality_weighted_mean_absolute_error_max": QUALITY_WEIGHTED_MEAN_ABSOLUTE_ERROR_MAX,
                "quality_weighted_median_absolute_error_max": QUALITY_MEDIAN_ABSOLUTE_ERROR_MAX,
                "quality_weighted_p90_absolute_error_max": QUALITY_P90_ABSOLUTE_ERROR_MAX,
                "payload_anchor_monotone_fraction_required": 1.0,
            },
            "weighting": "selection sampling_weight used for mean and quantiles",
        }
        canonical = canonical_json_bytes(document)
        report = QualificationReport(canonical, hashlib.sha256(canonical).hexdigest())
        self._qualification_cache = report
        return report


def load_empirical_quality_surface(
    corrected_p40_provider: CorrectedP40Provider,
    *,
    project_root: Optional[Path] = None,
) -> EmpiricalQualitySurface:
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    return EmpiricalQualitySurface.load(root / BUNDLE_RELATIVE_PATH, corrected_p40_provider)
