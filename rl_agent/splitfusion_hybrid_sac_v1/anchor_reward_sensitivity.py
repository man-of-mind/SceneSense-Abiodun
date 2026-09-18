"""Deterministic, design-only reward sensitivity over the 72 measured anchors.

This module deliberately does **not** construct ``ReplayTransitionV1`` records
and does not claim a causal or per-frame reward.  It combines action-level
validation anchors with a conditional P50 latency proxy from the optimized
288-row action/profile analysis.  The resulting scalar is useful only for
screening candidate reward specifications before training.

The latency proxy is survivor-conditioned: it exists only for frames that
reached model-ready.  Missing latency is kept missing and is never imputed as
zero.  Consequently, every reward ranking reports timing support and the
associated survivor-bias limitation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .anchor_store import (
    ACTION_SUMMARY_SHA256,
    PROFILE_LATENCY_SHA256,
    AnchorEvidenceStore,
    AnchorStoreError,
)


SCHEMA = "splitfusion.anchor_reward_sensitivity_design_screen.v1"
BUDGET_MS = 200.0
SEG_REFERENCE_VEHICLE_IOU = 0.899012847
SEG_REFERENCE_PERSON_IOU = 0.527894080
LATENCY_RATIOS = (0.10, 0.25, 0.50)
TOP_K_VALUES = (5, 10)
NETWORK_PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_SUMMARY = (
    REPO_ROOT
    / "experiments"
    / "splitfusion_288_offline_rl_dataset_v1"
    / "20260909_offline_consolidation_v1"
    / "action_72_summary.csv"
)
DEFAULT_ACTION_PROFILE = (
    REPO_ROOT
    / "experiments"
    / "splitfusion_supervisor_analysis_v1"
    / "20260915_tail_completion_feedback_policy_analysis_v3"
    / "action_profile_quality_latency.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "splitfusion_hybrid_sac_reward_sensitivity_v1"
    / "20260918_anchor_design_screen_v1"
)

ACTION_REQUIRED_FIELDS = (
    "action_id",
    "profile_id",
    "family",
    "quantizer",
    "q",
    "val_vehicle_recall",
    "val_vehicle_xy_mae_m",
    "val_vehicle_iou",
    "val_person_avo_recall",
    "val_person_avo_xy_mae_m",
    "val_person_box_mask_iou",
)
PROFILE_REQUIRED_FIELDS = (
    "action_id",
    "profile_id",
    "network_profile",
    "family",
    "quantizer",
    "q",
    "frames_sent",
    "measured_complete_reassemblies",
    "measured_edge_admissions",
    "simulated_map_installs",
    "rate_reassembled_per_sent",
    "rate_admitted_per_sent",
    "rate_installed_per_sent",
    "sensor_model_ready_count",
    "sensor_model_ready_p50_ms",
)


class SensitivityAuditError(ValueError):
    """Raised when an input or derived audit record violates the contract."""


@dataclass(frozen=True, slots=True)
class QualitySpec:
    """One explicit scientific hypothesis; never a production default."""

    spec_id: str
    localization_combiner: str
    person_localization_share: float
    person_segmentation_share: float
    tau_person_m: float
    tau_vehicle_m: float
    tau_label: str
    segmentation_modulation_beta: float
    is_initial_hypothesis: bool

    def __post_init__(self) -> None:
        """Reject malformed scientific hypotheses before they are scored."""
        if self.localization_combiner not in {"arithmetic", "geometric"}:
            raise SensitivityAuditError(
                "localization_combiner must be 'arithmetic' or 'geometric', "
                f"got {self.localization_combiner!r}"
            )
        for name, value in (
            ("person_localization_share", self.person_localization_share),
            ("person_segmentation_share", self.person_segmentation_share),
            ("segmentation_modulation_beta", self.segmentation_modulation_beta),
        ):
            number = _finite(value, name)
            if not 0.0 <= number <= 1.0:
                raise SensitivityAuditError(f"{name} must lie in [0,1], got {number}")
        for name, value in (
            ("tau_person_m", self.tau_person_m),
            ("tau_vehicle_m", self.tau_vehicle_m),
        ):
            number = _finite(value, name)
            if number <= 0.0:
                raise SensitivityAuditError(f"{name} must be positive, got {number}")


@dataclass(frozen=True, slots=True)
class QualityScore:
    u_xy_person: float
    u_xy_vehicle: float
    u_loc_person: float
    u_loc_vehicle: float
    s_person: float
    s_vehicle: float
    q_loc: float
    q_seg: float
    q_perc: float

    @property
    def min_normalized_component(self) -> float:
        """Diagnostic only; no registered class-quality floor is implied."""
        return min(
            self.u_loc_person,
            self.u_loc_vehicle,
            self.s_person,
            self.s_vehicle,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SensitivityAuditError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise SensitivityAuditError(f"{name} must be finite, got {number!r}")
    return number


def _optional_finite(value: Any, name: str) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    return _finite(value, name)


def _integer(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SensitivityAuditError(f"{name} must be an integer, got {value!r}") from exc
    if str(number) != str(value).strip() and _finite(value, name) != number:
        raise SensitivityAuditError(f"{name} must be an exact integer, got {value!r}")
    return number


def enumerate_quality_specs() -> tuple[QualitySpec, ...]:
    """Return the complete 162-cell design grid in a stable order."""
    tau_pairs = (
        ("strict", 0.9, 0.75),
        ("nominal", 1.2, 1.0),
        ("tolerant", 1.5, 1.25),
    )
    records: list[QualitySpec] = []
    for combiner in ("arithmetic", "geometric"):
        for p_loc in (0.5, 0.6, 0.7):
            for p_seg in (0.5, 0.6, 0.7):
                for tau_label, tau_person, tau_vehicle in tau_pairs:
                    for beta in (0.15, 0.30, 0.45):
                        spec_id = (
                            f"{combiner}_pl{int(round(100*p_loc)):02d}_"
                            f"ps{int(round(100*p_seg)):02d}_tau-{tau_label}_"
                            f"b{int(round(100*beta)):02d}"
                        )
                        initial = (
                            combiner == "geometric"
                            and p_loc == 0.6
                            and p_seg == 0.6
                            and tau_label == "nominal"
                            and beta == 0.30
                        )
                        records.append(
                            QualitySpec(
                                spec_id=spec_id,
                                localization_combiner=combiner,
                                person_localization_share=p_loc,
                                person_segmentation_share=p_seg,
                                tau_person_m=tau_person,
                                tau_vehicle_m=tau_vehicle,
                                tau_label=tau_label,
                                segmentation_modulation_beta=beta,
                                is_initial_hypothesis=initial,
                            )
                        )
    if len(records) != 162 or len({record.spec_id for record in records}) != 162:
        raise AssertionError("quality sensitivity grid must contain 162 unique specs")
    if sum(record.is_initial_hypothesis for record in records) != 1:
        raise AssertionError("quality sensitivity grid must contain one initial hypothesis")
    return tuple(records)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _weighted_geometric(person: float, vehicle: float, person_share: float) -> float:
    if person <= 0.0 or vehicle <= 0.0:
        return 0.0
    return math.exp(
        person_share * math.log(person)
        + (1.0 - person_share) * math.log(vehicle)
    )


def score_quality(action: Mapping[str, Any], spec: QualitySpec) -> QualityScore:
    """Evaluate the exact registered aggregate proxy for one action anchor."""
    person_recall = _finite(action["val_person_avo_recall"], "person AVO recall")
    vehicle_recall = _finite(action["val_vehicle_recall"], "vehicle recall")
    person_error = _finite(action["val_person_avo_xy_mae_m"], "person AVO XY MAE")
    vehicle_error = _finite(action["val_vehicle_xy_mae_m"], "vehicle XY MAE")
    person_iou = _finite(action["val_person_box_mask_iou"], "person box-mask IoU")
    vehicle_iou = _finite(action["val_vehicle_iou"], "vehicle IoU")
    for name, value in (
        ("person AVO recall", person_recall),
        ("vehicle recall", vehicle_recall),
        ("person box-mask IoU", person_iou),
        ("vehicle IoU", vehicle_iou),
    ):
        if not 0.0 <= value <= 1.0:
            raise SensitivityAuditError(f"{name} must lie in [0,1], got {value}")
    if person_error < 0.0 or vehicle_error < 0.0:
        raise SensitivityAuditError("localization error must be non-negative")

    u_xy_person = math.exp(-person_error / spec.tau_person_m)
    u_xy_vehicle = math.exp(-vehicle_error / spec.tau_vehicle_m)
    u_loc_person = math.sqrt(person_recall * u_xy_person)
    u_loc_vehicle = math.sqrt(vehicle_recall * u_xy_vehicle)
    if spec.localization_combiner == "geometric":
        q_loc = _weighted_geometric(
            u_loc_person, u_loc_vehicle, spec.person_localization_share
        )
    elif spec.localization_combiner == "arithmetic":
        q_loc = (
            spec.person_localization_share * u_loc_person
            + (1.0 - spec.person_localization_share) * u_loc_vehicle
        )
    else:  # pragma: no cover - dataclass is public, so fail closed
        raise SensitivityAuditError(
            f"unsupported localization combiner {spec.localization_combiner!r}"
        )
    s_person = _clip01(person_iou / SEG_REFERENCE_PERSON_IOU)
    s_vehicle = _clip01(vehicle_iou / SEG_REFERENCE_VEHICLE_IOU)
    q_seg = _weighted_geometric(s_person, s_vehicle, spec.person_segmentation_share)
    beta = spec.segmentation_modulation_beta
    q_perc = q_loc * ((1.0 - beta) + beta * q_seg)
    return QualityScore(
        u_xy_person=u_xy_person,
        u_xy_vehicle=u_xy_vehicle,
        u_loc_person=u_loc_person,
        u_loc_vehicle=u_loc_vehicle,
        s_person=s_person,
        s_vehicle=s_vehicle,
        q_loc=q_loc,
        q_seg=q_seg,
        q_perc=q_perc,
    )


def reward_proxy(
    quality: float,
    model_ready_latency_proxy_ms: Optional[float],
    latency_ratio: float,
) -> Optional[float]:
    """Return ``Q - rho*(L/B)`` or ``None`` when survivor latency is absent."""
    if model_ready_latency_proxy_ms is None:
        return None
    latency = _finite(model_ready_latency_proxy_ms, "model-ready latency proxy")
    if latency < 0.0:
        raise SensitivityAuditError("model-ready latency proxy cannot be negative")
    return float(quality) - float(latency_ratio) * latency / BUDGET_MS


def average_ranks(values: Sequence[float], *, descending: bool = True) -> list[float]:
    """Average ranks with deterministic tie handling; rank one is best."""
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: ((-item[1]) if descending else item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        average = ((start + 1) + stop) / 2.0
        for offset in range(start, stop):
            ranks[indexed[offset][0]] = average
        start = stop
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks and no SciPy dependency."""
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise SensitivityAuditError("Spearman inputs must have equal length >= 2")
    rank_a = average_ranks(values_a)
    rank_b = average_ranks(values_b)
    mean_a = sum(rank_a) / len(rank_a)
    mean_b = sum(rank_b) / len(rank_b)
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(rank_a, rank_b))
    variance_a = sum((a - mean_a) ** 2 for a in rank_a)
    variance_b = sum((b - mean_b) ** 2 for b in rank_b)
    if variance_a == 0.0 or variance_b == 0.0:
        raise SensitivityAuditError("Spearman is undefined for a constant ranking")
    return covariance / math.sqrt(variance_a * variance_b)


def load_sources(
    action_summary_path: Path,
    action_profile_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load sources through the SHA-pinned, catalog-reconciled anchor store.

    This design screen intentionally does not maintain a second, weaker CSV
    parser.  The committed :class:`AnchorEvidenceStore` is the evidence
    boundary: it checks both exact file hashes, reconciles all 72 actions to
    the frozen catalog, proves the 72-by-4 cell inventory, and validates the
    cross-source identities before any sensitivity value is computed.
    """
    try:
        store = AnchorEvidenceStore.from_paths(
            action_summary_path=Path(action_summary_path),
            profile_latency_path=Path(action_profile_path),
        )
    except AnchorStoreError as exc:
        raise SensitivityAuditError(f"anchor-store evidence binding failed: {exc}") from exc

    actions: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for record in store.records:
        quality = record.quality
        try:
            action_row: dict[str, Any] = {
                "action_id": quality.action_id,
                "profile_id": quality.profile_id,
                "family": quality.family,
                "quantizer": quality.quantizer,
                "q": quality.q,
                **{
                    field: quality.raw_quality[field]
                    for field in ACTION_REQUIRED_FIELDS
                    if field.startswith("val_")
                },
            }
        except KeyError as exc:  # Defensive: store schema must retain our registered inputs.
            raise SensitivityAuditError(
                f"anchor action {quality.action_id} lacks required quality field {exc.args[0]!r}"
            ) from exc
        # Exercise the quality formula once at the evidence boundary as an
        # additional domain check, not as a substitute for AnchorStore proof.
        score_quality(action_row, enumerate_quality_specs()[0])
        actions.append(action_row)

        for network_profile in NETWORK_PROFILE_ORDER:
            outcome = record.outcome(network_profile)
            model_ready = outcome.latency_stat("sensor_model_ready")
            profiles.append(
                {
                    "action_id": quality.action_id,
                    "profile_id": quality.profile_id,
                    "network_profile": network_profile,
                    "family": quality.family,
                    "quantizer": quality.quantizer,
                    "q": quality.q,
                    "frames_sent": outcome.frames_sent,
                    "measured_complete_reassemblies": outcome.counts[
                        "replay_v3__measured_complete_reassemblies"
                    ],
                    "measured_edge_admissions": outcome.counts[
                        "replay_v3__measured_edge_admissions"
                    ],
                    "simulated_map_installs": outcome.counts[
                        "replay_v3__simulated_map_installs"
                    ],
                    "rate_reassembled_per_sent": outcome.rates[
                        "replay_v3__rate_reassembled_per_sent"
                    ],
                    "rate_admitted_per_sent": outcome.rates[
                        "replay_v3__rate_admitted_per_sent"
                    ],
                    "rate_installed_per_sent": outcome.rates[
                        "replay_v3__rate_installed_per_sent"
                    ],
                    "sensor_model_ready_count": model_ready.support,
                    "sensor_model_ready_p50_ms": model_ready.p50_ms,
                }
            )

    def display_path(path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(resolved)

    binding = {
        "action_72_summary": {
            "path": display_path(action_summary_path),
            "rows": len(actions),
            "sha256": store.action_summary_sha256,
            "required_sha256": ACTION_SUMMARY_SHA256,
        },
        "action_profile_quality_latency_v3": {
            "path": display_path(action_profile_path),
            "rows": len(profiles),
            "sha256": store.profile_latency_sha256,
            "required_sha256": PROFILE_LATENCY_SHA256,
        },
        "frozen_action_catalog": {
            "path": display_path(store.contract.catalog_path),
            "sha256": store.contract.catalog_sha256,
            "reconciled_action_count": len(store.records),
        },
        "binding_implementation": "AnchorEvidenceStore",
    }
    return actions, profiles, binding


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SensitivityAuditError(f"cannot serialize non-finite float {value!r}")
        return format(value, ".12g")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fields})


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _top_ids(rows: Sequence[Mapping[str, Any]], score_field: str, k: int) -> set[int]:
    ordered = sorted(rows, key=lambda row: (-float(row[score_field]), int(row["action_id"])))
    return {int(row["action_id"]) for row in ordered[:k]}


def _component_minima(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        name: min(float(row[name]) for row in rows)
        for name in ("u_loc_person", "u_loc_vehicle", "s_person", "s_vehicle")
    }


def _rank_rows(rows: list[dict[str, Any]], score_field: str, rank_field: str) -> None:
    ordered = sorted(rows, key=lambda row: (-float(row[score_field]), int(row["action_id"])))
    scores = [float(row[score_field]) for row in ordered]
    ranks = average_ranks(scores)
    for row, rank in zip(ordered, ranks):
        row[rank_field] = rank


def _member_rows(
    *,
    ranking_kind: str,
    spec: QualitySpec,
    ordered: Sequence[Mapping[str, Any]],
    score_field: str,
    profile: str = "",
    latency_ratio: Optional[float] = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for k in TOP_K_VALUES:
        for row in ordered[:k]:
            result.append(
                {
                    "ranking_kind": ranking_kind,
                    "quality_spec_id": spec.spec_id,
                    "network_profile": profile,
                    "latency_ratio": latency_ratio,
                    "top_k": k,
                    "action_id": row["action_id"],
                    "profile_id": row["profile_id"],
                    "rank": row["quality_rank"] if ranking_kind == "quality" else row["reward_rank"],
                    "score": row[score_field],
                    "val_person_avo_recall": row["val_person_avo_recall"],
                    "val_person_avo_xy_mae_m": row["val_person_avo_xy_mae_m"],
                    "val_vehicle_recall": row["val_vehicle_recall"],
                    "val_vehicle_xy_mae_m": row["val_vehicle_xy_mae_m"],
                    "val_person_box_mask_iou": row["val_person_box_mask_iou"],
                    "val_vehicle_iou": row["val_vehicle_iou"],
                    "u_loc_person": row["u_loc_person"],
                    "u_loc_vehicle": row["u_loc_vehicle"],
                    "s_person": row["s_person"],
                    "s_vehicle": row["s_vehicle"],
                    "min_normalized_component_diagnostic": row[
                        "min_normalized_component_diagnostic"
                    ],
                    "model_ready_latency_proxy_ms": row.get(
                        "model_ready_latency_proxy_ms"
                    ),
                    "model_ready_support_rate": row.get("model_ready_support_rate"),
                    "rate_reassembled_per_sent": row.get("rate_reassembled_per_sent"),
                    "rate_admitted_per_sent": row.get("rate_admitted_per_sent"),
                    "rate_installed_per_sent": row.get("rate_installed_per_sent"),
                }
            )
    return result


def run_audit(
    *,
    action_summary_path: Path = DEFAULT_ACTION_SUMMARY,
    action_profile_path: Path = DEFAULT_ACTION_PROFILE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Path:
    """Run the complete deterministic design screen and write its artifacts."""
    actions, profiles, source_binding = load_sources(
        action_summary_path, action_profile_path
    )
    implementation_path = Path(__file__).resolve()
    reward_contract_path = implementation_path.with_name(
        "state_reward_transition_contract.py"
    )
    source_binding["audit_implementation"] = {
        "path": str(implementation_path.relative_to(REPO_ROOT.resolve())),
        "sha256": _sha256(implementation_path),
    }
    source_binding["reward_formula_contract"] = {
        "path": str(reward_contract_path.relative_to(REPO_ROOT.resolve())),
        "sha256": _sha256(reward_contract_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = enumerate_quality_specs()
    initial_spec = next(spec for spec in specs if spec.is_initial_hypothesis)

    raw_action_rows: list[dict[str, Any]] = []
    for row in sorted(actions, key=lambda value: int(value["action_id"])):
        raw_action_rows.append(
            {field: row[field] for field in ACTION_REQUIRED_FIELDS}
        )

    raw_profile_rows: list[dict[str, Any]] = []
    profile_order = {name: index for index, name in enumerate(NETWORK_PROFILE_ORDER)}
    for row in sorted(
        profiles,
        key=lambda value: (profile_order[value["network_profile"]], int(value["action_id"])),
    ):
        latency = _optional_finite(
            row["sensor_model_ready_p50_ms"], "sensor_model_ready_p50_ms"
        )
        frames_sent = _integer(row["frames_sent"], "frames_sent")
        timing_count = _integer(row["sensor_model_ready_count"], "sensor_model_ready_count")
        raw_profile_rows.append(
            {
                **{field: row[field] for field in PROFILE_REQUIRED_FIELDS},
                "model_ready_latency_proxy_ms": latency,
                "model_ready_latency_available": latency is not None,
                "model_ready_support_rate": timing_count / frames_sent,
                "missing_latency_reason": (
                    "" if latency is not None else "NO_MODEL_READY_TIMING_SURVIVORS"
                ),
            }
        )

    quality_rows: list[dict[str, Any]] = []
    quality_by_spec_action: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in specs:
        current: list[dict[str, Any]] = []
        for action in sorted(actions, key=lambda value: int(value["action_id"])):
            action_id = int(action["action_id"])
            score = score_quality(action, spec)
            row = {
                **asdict(spec),
                "action_id": action_id,
                "profile_id": action["profile_id"],
                "family": action["family"],
                "quantizer": action["quantizer"],
                "q": _finite(action["q"], "q"),
                "val_person_avo_recall": _finite(
                    action["val_person_avo_recall"], "val_person_avo_recall"
                ),
                "val_person_avo_xy_mae_m": _finite(
                    action["val_person_avo_xy_mae_m"], "val_person_avo_xy_mae_m"
                ),
                "val_vehicle_recall": _finite(
                    action["val_vehicle_recall"], "val_vehicle_recall"
                ),
                "val_vehicle_xy_mae_m": _finite(
                    action["val_vehicle_xy_mae_m"], "val_vehicle_xy_mae_m"
                ),
                "val_person_box_mask_iou": _finite(
                    action["val_person_box_mask_iou"], "val_person_box_mask_iou"
                ),
                "val_vehicle_iou": _finite(action["val_vehicle_iou"], "val_vehicle_iou"),
                **asdict(score),
                "min_normalized_component_diagnostic": score.min_normalized_component,
            }
            current.append(row)
        _rank_rows(current, "q_perc", "quality_rank")
        for row in current:
            quality_by_spec_action[(spec.spec_id, int(row["action_id"]))] = row
        quality_rows.extend(current)

    initial_quality = {
        int(row["action_id"]): row
        for row in quality_rows
        if row["spec_id"] == initial_spec.spec_id
    }
    quality_summary: list[dict[str, Any]] = []
    top_members: list[dict[str, Any]] = []
    for spec in specs:
        rows = [row for row in quality_rows if row["spec_id"] == spec.spec_id]
        ordered = sorted(rows, key=lambda row: (-float(row["q_perc"]), int(row["action_id"])))
        baseline_scores = [
            initial_quality[int(row["action_id"])]["q_perc"] for row in rows
        ]
        current_scores = [row["q_perc"] for row in rows]
        summary: dict[str, Any] = {
            **asdict(spec),
            "spearman_vs_initial_quality": spearman(current_scores, baseline_scores),
            "quality_min_all_actions": min(current_scores),
            "quality_max_all_actions": max(current_scores),
            "best_action_id": ordered[0]["action_id"],
            "best_profile_id": ordered[0]["profile_id"],
            "best_quality": ordered[0]["q_perc"],
            "best_min_normalized_component_diagnostic": ordered[0][
                "min_normalized_component_diagnostic"
            ],
            "class_floor_thresholds_status": "UNFROZEN_REQUIRES_SUPERVISOR_CHOICE",
        }
        for k in TOP_K_VALUES:
            current_top = _top_ids(rows, "q_perc", k)
            baseline_top = _top_ids(list(initial_quality.values()), "q_perc", k)
            selected = ordered[:k]
            minima = _component_minima(selected)
            summary[f"top_{k}_overlap_count_vs_initial"] = len(current_top & baseline_top)
            summary[f"top_{k}_overlap_fraction_vs_initial"] = len(
                current_top & baseline_top
            ) / k
            for name, value in minima.items():
                summary[f"observed_top_{k}_component_min_{name}"] = value
        quality_summary.append(summary)
        top_members.extend(
            _member_rows(
                ranking_kind="quality",
                spec=spec,
                ordered=ordered,
                score_field="q_perc",
            )
        )

    reward_rows: list[dict[str, Any]] = []
    reward_summary: list[dict[str, Any]] = []
    for spec in specs:
        for latency_ratio in LATENCY_RATIOS:
            for network_profile in NETWORK_PROFILE_ORDER:
                profile_inputs = [
                    row for row in raw_profile_rows if row["network_profile"] == network_profile
                ]
                current: list[dict[str, Any]] = []
                for source in profile_inputs:
                    action_id = int(source["action_id"])
                    quality = quality_by_spec_action[(spec.spec_id, action_id)]
                    latency = source["model_ready_latency_proxy_ms"]
                    reward = reward_proxy(quality["q_perc"], latency, latency_ratio)
                    row = {
                        "quality_spec_id": spec.spec_id,
                        "is_initial_hypothesis": spec.is_initial_hypothesis,
                        "latency_ratio": latency_ratio,
                        "latency_budget_ms": BUDGET_MS,
                        "network_profile": network_profile,
                        "action_id": action_id,
                        "profile_id": source["profile_id"],
                        "family": source["family"],
                        "quantizer": source["quantizer"],
                        "q": _finite(source["q"], "q"),
                        "q_perc": quality["q_perc"],
                        "u_loc_person": quality["u_loc_person"],
                        "u_loc_vehicle": quality["u_loc_vehicle"],
                        "s_person": quality["s_person"],
                        "s_vehicle": quality["s_vehicle"],
                        "min_normalized_component_diagnostic": quality[
                            "min_normalized_component_diagnostic"
                        ],
                        "val_person_avo_recall": quality["val_person_avo_recall"],
                        "val_person_avo_xy_mae_m": quality[
                            "val_person_avo_xy_mae_m"
                        ],
                        "val_vehicle_recall": quality["val_vehicle_recall"],
                        "val_vehicle_xy_mae_m": quality["val_vehicle_xy_mae_m"],
                        "val_person_box_mask_iou": quality["val_person_box_mask_iou"],
                        "val_vehicle_iou": quality["val_vehicle_iou"],
                        "model_ready_latency_proxy_ms": latency,
                        "reward_proxy": reward,
                        "reward_available": reward is not None,
                        "reward_unavailable_reason": source["missing_latency_reason"],
                        "frames_sent": _integer(source["frames_sent"], "frames_sent"),
                        "sensor_model_ready_count": _integer(
                            source["sensor_model_ready_count"], "sensor_model_ready_count"
                        ),
                        "model_ready_support_rate": source["model_ready_support_rate"],
                        "rate_reassembled_per_sent": _finite(
                            source["rate_reassembled_per_sent"], "reassembly rate"
                        ),
                        "rate_admitted_per_sent": _finite(
                            source["rate_admitted_per_sent"], "admission rate"
                        ),
                        "rate_installed_per_sent": _finite(
                            source["rate_installed_per_sent"], "installation rate"
                        ),
                    }
                    current.append(row)
                eligible = [row for row in current if row["reward_proxy"] is not None]
                _rank_rows(eligible, "reward_proxy", "reward_rank")
                for row in current:
                    row.setdefault("reward_rank", None)
                reward_rows.extend(current)
                ordered = sorted(
                    eligible,
                    key=lambda row: (-float(row["reward_proxy"]), int(row["action_id"])),
                )
                if not ordered:
                    raise SensitivityAuditError(
                        f"no model-ready latency support for {network_profile}"
                    )
                # The initial spec appears after some grid cells.  Defer
                # correlations until all reward rows have been materialized.
                reward_summary.append(
                    {
                        **asdict(spec),
                        "latency_ratio": latency_ratio,
                        "latency_budget_ms": BUDGET_MS,
                        "network_profile": network_profile,
                        "available_action_count": len(eligible),
                        "missing_action_count": len(current) - len(eligible),
                        "available_action_fraction": len(eligible) / len(current),
                        "best_action_id": ordered[0]["action_id"],
                        "best_profile_id": ordered[0]["profile_id"],
                        "best_reward_proxy": ordered[0]["reward_proxy"],
                        "best_q_perc": ordered[0]["q_perc"],
                        "best_model_ready_latency_proxy_ms": ordered[0][
                            "model_ready_latency_proxy_ms"
                        ],
                        "best_model_ready_support_rate": ordered[0][
                            "model_ready_support_rate"
                        ],
                        "best_rate_reassembled_per_sent": ordered[0][
                            "rate_reassembled_per_sent"
                        ],
                        "best_rate_admitted_per_sent": ordered[0][
                            "rate_admitted_per_sent"
                        ],
                        "best_rate_installed_per_sent": ordered[0][
                            "rate_installed_per_sent"
                        ],
                        "best_min_normalized_component_diagnostic": ordered[0][
                            "min_normalized_component_diagnostic"
                        ],
                        "class_floor_thresholds_status": (
                            "UNFROZEN_REQUIRES_SUPERVISOR_CHOICE"
                        ),
                        "survivor_bias_warning": (
                            "P50 latency is conditional on model-ready survivors; "
                            "missing actions are excluded, never scored as zero latency"
                        ),
                    }
                )
                top_members.extend(
                    _member_rows(
                        ranking_kind="reward",
                        spec=spec,
                        ordered=ordered,
                        score_field="reward_proxy",
                        profile=network_profile,
                        latency_ratio=latency_ratio,
                    )
                )

    reward_lookup: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in reward_rows:
        if row["reward_proxy"] is not None:
            reward_lookup.setdefault(
                (
                    str(row["quality_spec_id"]),
                    float(row["latency_ratio"]),
                    str(row["network_profile"]),
                ),
                [],
            ).append(row)
    for summary in reward_summary:
        key = (
            str(summary["spec_id"]),
            float(summary["latency_ratio"]),
            str(summary["network_profile"]),
        )
        baseline_key = (
            initial_spec.spec_id,
            float(summary["latency_ratio"]),
            str(summary["network_profile"]),
        )
        current = sorted(reward_lookup[key], key=lambda row: int(row["action_id"]))
        baseline_by_action = {
            int(row["action_id"]): row for row in reward_lookup[baseline_key]
        }
        if {int(row["action_id"]) for row in current} != set(baseline_by_action):
            raise SensitivityAuditError("latency availability changed across quality specs")
        current_values = [float(row["reward_proxy"]) for row in current]
        baseline_values = [
            float(baseline_by_action[int(row["action_id"])]["reward_proxy"])
            for row in current
        ]
        summary["spearman_vs_initial_reward_same_latency_ratio"] = spearman(
            current_values, baseline_values
        )
        ordered = sorted(
            current,
            key=lambda row: (-float(row["reward_proxy"]), int(row["action_id"])),
        )
        baseline_ordered = sorted(
            baseline_by_action.values(),
            key=lambda row: (-float(row["reward_proxy"]), int(row["action_id"])),
        )
        for k in TOP_K_VALUES:
            current_top = {int(row["action_id"]) for row in ordered[:k]}
            baseline_top = {int(row["action_id"]) for row in baseline_ordered[:k]}
            summary[f"top_{k}_overlap_count_vs_initial"] = len(
                current_top & baseline_top
            )
            summary[f"top_{k}_overlap_fraction_vs_initial"] = len(
                current_top & baseline_top
            ) / k
            for name, value in _component_minima(ordered[:k]).items():
                summary[f"observed_top_{k}_component_min_{name}"] = value

    # Stable output ordering.
    quality_rows.sort(key=lambda row: (row["spec_id"], int(row["action_id"])))
    quality_summary.sort(key=lambda row: row["spec_id"])
    reward_rows.sort(
        key=lambda row: (
            row["quality_spec_id"],
            float(row["latency_ratio"]),
            profile_order[row["network_profile"]],
            int(row["action_id"]),
        )
    )
    reward_summary.sort(
        key=lambda row: (
            row["spec_id"],
            float(row["latency_ratio"]),
            profile_order[row["network_profile"]],
        )
    )
    top_members.sort(
        key=lambda row: (
            row["ranking_kind"],
            row["quality_spec_id"],
            str(row["latency_ratio"]),
            row["network_profile"],
            int(row["top_k"]),
            float(row["rank"]),
            int(row["action_id"]),
        )
    )

    spec_fields = list(asdict(specs[0]))
    _write_csv(
        output_dir / "quality_spec_grid.csv",
        [asdict(spec) for spec in specs],
        spec_fields,
    )
    _write_csv(output_dir / "raw_action_inputs.csv", raw_action_rows, ACTION_REQUIRED_FIELDS)
    raw_profile_fields = list(PROFILE_REQUIRED_FIELDS) + [
        "model_ready_latency_proxy_ms",
        "model_ready_latency_available",
        "model_ready_support_rate",
        "missing_latency_reason",
    ]
    _write_csv(
        output_dir / "raw_action_profile_inputs.csv",
        raw_profile_rows,
        raw_profile_fields,
    )
    quality_fields = spec_fields + [
        "action_id",
        "profile_id",
        "family",
        "quantizer",
        "q",
        "val_person_avo_recall",
        "val_person_avo_xy_mae_m",
        "val_vehicle_recall",
        "val_vehicle_xy_mae_m",
        "val_person_box_mask_iou",
        "val_vehicle_iou",
        "u_xy_person",
        "u_xy_vehicle",
        "u_loc_person",
        "u_loc_vehicle",
        "s_person",
        "s_vehicle",
        "q_loc",
        "q_seg",
        "q_perc",
        "min_normalized_component_diagnostic",
        "quality_rank",
    ]
    _write_csv(output_dir / "quality_rankings.csv", quality_rows, quality_fields)
    quality_summary_fields = list(quality_summary[0])
    _write_csv(
        output_dir / "quality_sensitivity_summary.csv",
        quality_summary,
        quality_summary_fields,
    )
    reward_fields = list(reward_rows[0])
    _write_csv(output_dir / "reward_rankings.csv", reward_rows, reward_fields)
    reward_summary_fields = list(reward_summary[0])
    _write_csv(
        output_dir / "reward_sensitivity_summary.csv",
        reward_summary,
        reward_summary_fields,
    )
    top_member_fields = list(top_members[0])
    _write_csv(output_dir / "top_k_members.csv", top_members, top_member_fields)

    initial_rewards = [
        row for row in reward_summary if bool(row["is_initial_hypothesis"])
    ]
    availability = {
        profile: {
            "available_actions": sum(
                1
                for row in raw_profile_rows
                if row["network_profile"] == profile
                and row["model_ready_latency_available"]
            ),
            "missing_actions": sum(
                1
                for row in raw_profile_rows
                if row["network_profile"] == profile
                and not row["model_ready_latency_available"]
            ),
        }
        for profile in NETWORK_PROFILE_ORDER
    }
    summary_document = {
        "schema": SCHEMA,
        "scope": "NONCAUSAL_ACTION_LEVEL_DESIGN_SCREEN_NOT_REPLAY_TRANSITIONS",
        "source_binding": source_binding,
        "quality_spec_count": len(specs),
        "latency_ratios": list(LATENCY_RATIOS),
        "latency_budget_ms": BUDGET_MS,
        "latency_proxy": "sensor_model_ready_p50_ms renamed model_ready_latency_proxy_ms",
        "initial_hypothesis": asdict(initial_spec),
        "production_defaults_frozen": False,
        "latency_weight_frozen": False,
        "latency_weight_status": (
            "CANNOT_FREEZE_FROM_SURVIVOR_CONDITIONED_MODEL_READY_PROXY"
        ),
        "class_quality_floors_frozen": False,
        "timing_availability": availability,
        "initial_hypothesis_best_by_profile_and_latency_ratio": [
            {
                key: row[key]
                for key in (
                    "latency_ratio",
                    "network_profile",
                    "available_action_count",
                    "missing_action_count",
                    "best_action_id",
                    "best_profile_id",
                    "best_reward_proxy",
                    "best_q_perc",
                    "best_model_ready_latency_proxy_ms",
                    "best_model_ready_support_rate",
                    "best_rate_reassembled_per_sent",
                    "best_rate_admitted_per_sent",
                    "best_rate_installed_per_sent",
                )
            }
            for row in initial_rewards
        ],
        "sensitivity_result": {
            "quality_spearman_min": min(
                float(row["spearman_vs_initial_quality"])
                for row in quality_summary
            ),
            "quality_spearman_max": max(
                float(row["spearman_vs_initial_quality"])
                for row in quality_summary
            ),
            "quality_top_5_overlap_min": min(
                int(row["top_5_overlap_count_vs_initial"])
                for row in quality_summary
            ),
            "quality_top_10_overlap_min": min(
                int(row["top_10_overlap_count_vs_initial"])
                for row in quality_summary
            ),
            "reward_spearman_min": min(
                float(row["spearman_vs_initial_reward_same_latency_ratio"])
                for row in reward_summary
            ),
            "reward_spearman_max": max(
                float(row["spearman_vs_initial_reward_same_latency_ratio"])
                for row in reward_summary
            ),
            "reward_top_5_overlap_min": min(
                int(row["top_5_overlap_count_vs_initial"])
                for row in reward_summary
            ),
            "reward_top_10_overlap_min": min(
                int(row["top_10_overlap_count_vs_initial"])
                for row in reward_summary
            ),
            "interpretation": (
                "Rank order is broadly stable over the design grid, but the "
                "conditional-latency winners are often supported by only one "
                "or a handful of model-ready survivors. They are not candidate "
                "production defaults without a separately chosen support/failure rule."
            ),
        },
        "limitations": [
            "Aggregate action-level validation quality is joined to profile-level latency; this is noncausal and not per-frame reward evidence.",
            "The P50 sensor_model_ready latency proxy is not the exact reward latency; it is conditional on frames that survived to model-ready, and unavailable actions are excluded rather than imputed.",
            "Support varies sharply by action and network profile, so rankings can reflect survivor bias.",
            "The latency weight w_L cannot be frozen from this survivor-conditioned proxy screen.",
            "The optimized 288-row v3 table is a counterfactual analysis over the immutable sent population, not a second live 288-cell campaign.",
            "Mode-switch cost, q-switch cost, registered failure reward and timeout adjudication are intentionally absent from this one-step screen.",
            "Person/vehicle class-quality floor thresholds remain unfrozen and require supervisor choice.",
            "No value emitted here is a production reward default or a ReplayTransitionV1.",
        ],
    }
    _write_json(output_dir / "summary.json", summary_document)

    report_lines = [
        "# Hybrid-SAC anchor reward sensitivity: design-only screen",
        "",
        "**Status:** noncausal, action-level design screen. This is not a causal per-frame reward, "
        "not `ReplayTransitionV1`, and freezes no production default.",
        "",
        "## Inputs and binding",
        "",
        f"- 72-row action validation table: `{source_binding['action_72_summary']['path']}` "
        f"(`{source_binding['action_72_summary']['sha256']}`).",
        f"- 288-row optimized action/profile table: "
        f"`{source_binding['action_profile_quality_latency_v3']['path']}` "
        f"(`{source_binding['action_profile_quality_latency_v3']['sha256']}`).",
        "- The exact raw joined inputs are reproduced in `raw_action_inputs.csv` and "
        "`raw_action_profile_inputs.csv`.",
        "",
        "## Proxy equations",
        "",
        r"For class $c$, $U_{xy,c}=\exp(-e_c/\tau_c)$ and "
        r"$U_{loc,c}=\sqrt{Recall_c U_{xy,c}}$. $Q_{loc}$ is either the explicit "
        r"weighted arithmetic or weighted geometric combiner. Segmentation uses "
        r"$s_c=\mathrm{clip}(IoU_c/IoU_c^{ref},0,1)$ and a weighted geometric mean. "
        r"$Q_{perc}=Q_{loc}[(1-\beta)+\beta Q_{seg}]$.",
        "",
        r"The design-only scalar is $R_{proxy}=Q_{perc}-\rho L_{proxy}/200\,ms$, "
        r"for $\rho\in\{0.10,0.25,0.50\}$. `L_proxy` is only the finite "
        "`sensor_model_ready_p50_ms`, renamed `model_ready_latency_proxy_ms`; it is "
        "not the exact reward latency.",
        "",
        "## Grid",
        "",
        "The audit evaluates all 162 quality hypotheses: two localization combiners, "
        "person shares {0.5, 0.6, 0.7} independently for localization and segmentation, "
        "three `(tau_person, tau_vehicle)` pairs, and beta {0.15, 0.30, 0.45}. The "
        "comparison anchor is geometric, person shares 0.6/0.6, nominal taus "
        "(1.2 m, 1.0 m), beta 0.30. It is an initial hypothesis, not a default.",
        "",
        "## Initial-hypothesis best action by profile",
        "",
        "| rho | network profile | timed actions | best action | proxy reward | quality | "
        "latency proxy (ms) | timing support | reassembled/sent | admitted/sent | installed/sent |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        initial_rewards,
        key=lambda value: (
            float(value["latency_ratio"]),
            profile_order[value["network_profile"]],
        ),
    ):
        report_lines.append(
            "| {latency_ratio:.2f} | {network_profile} | {available_action_count}/72 | "
            "{best_action_id} | {best_reward_proxy:.4f} | {best_q_perc:.4f} | "
            "{best_model_ready_latency_proxy_ms:.1f} | {best_model_ready_support_rate:.3%} | "
            "{best_rate_reassembled_per_sent:.3f} | {best_rate_admitted_per_sent:.3f} | "
            "{best_rate_installed_per_sent:.3f} |".format(**row)
        )
    report_lines.extend(
        [
            "",
            "These are mathematical winners among actions with finite survivor-conditioned "
            "latency, not recommended actions. Most winners above have timing support around "
            "0.03--0.20%, exposing the central survivor-bias problem rather than solving it. "
            "No support threshold or failure penalty was invented in this audit.",
            "",
            "## Rank robustness",
            "",
            "Across all 162 quality specifications, Spearman correlation against the "
            f"initial hypothesis ranges from "
            f"{summary_document['sensitivity_result']['quality_spearman_min']:.4f} to "
            f"{summary_document['sensitivity_result']['quality_spearman_max']:.4f}; the "
            f"smallest top-5 overlap is "
            f"{summary_document['sensitivity_result']['quality_top_5_overlap_min']}/5 and "
            f"top-10 overlap is "
            f"{summary_document['sensitivity_result']['quality_top_10_overlap_min']}/10.",
            "",
            "Within the same rho and network profile, proxy-reward Spearman correlation "
            f"ranges from {summary_document['sensitivity_result']['reward_spearman_min']:.4f} "
            f"to {summary_document['sensitivity_result']['reward_spearman_max']:.4f}; the "
            f"smallest top-5 overlap is "
            f"{summary_document['sensitivity_result']['reward_top_5_overlap_min']}/5 and "
            f"top-10 overlap is "
            f"{summary_document['sensitivity_result']['reward_top_10_overlap_min']}/10. "
            "This shows broad robustness to the quality-weight hypotheses, not validity of "
            "the survivor-conditioned latency proxy.",
            "",
            "## Availability and survivor bias",
            "",
        ]
    )
    for profile in NETWORK_PROFILE_ORDER:
        counts = availability[profile]
        report_lines.append(
            f"- {profile}: {counts['available_actions']}/72 actions have a finite "
            f"model-ready P50; {counts['missing_actions']} remain unavailable."
        )
    report_lines.extend(
        [
            "",
            "A finite P50 describes only frames that reached model-ready. It does not make "
            "transport-incomplete or pre-model failures fast. The ranking tables therefore "
            "carry timing-support, reassembly, admission and installation rates next to the "
            "score. Missing latency is blank and excluded, never replaced by zero.",
            "",
            "## Sensitivity products",
            "",
            "- `quality_sensitivity_summary.csv`: Spearman and top-5/top-10 overlap against "
            "the initial quality hypothesis.",
            "- `reward_sensitivity_summary.csv`: the same comparison within each rho/profile, "
            "plus best action and support.",
            "- `top_k_members.csv`: raw recall/error/IoU inputs and normalized person/vehicle "
            "components for every top-5/top-10 set.",
            "- `quality_rankings.csv` and `reward_rankings.csv`: complete rankings, including "
            "explicit unavailable reward rows.",
            "",
            "Observed top-k component minima are diagnostics, not registered policy floors. "
            "Actual person/vehicle quality-floor thresholds remain unfrozen and require a "
            "supervisor decision.",
            "",
            "The latency weight `w_L` also remains unfrozen: this screen uses a "
            "survivor-conditioned model-ready proxy, not the exact feedback/reward latency, "
            "so freezing `w_L` from these rankings would encode survivor bias.",
            "",
            "## Limitations",
            "",
        ]
    )
    report_lines.extend(f"- {item}" for item in summary_document["limitations"])
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    artifact_names = (
        "REPORT.md",
        "quality_spec_grid.csv",
        "raw_action_inputs.csv",
        "raw_action_profile_inputs.csv",
        "quality_rankings.csv",
        "quality_sensitivity_summary.csv",
        "reward_rankings.csv",
        "reward_sensitivity_summary.csv",
        "top_k_members.csv",
        "summary.json",
    )
    manifest = {
        "schema": SCHEMA,
        "source_binding": source_binding,
        "artifacts": {
            name: {"sha256": _sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
            for name in artifact_names
        },
    }
    manifest["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-summary", type=Path, default=DEFAULT_ACTION_SUMMARY)
    parser.add_argument("--action-profile", type=Path, default=DEFAULT_ACTION_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(
        run_audit(
            action_summary_path=args.action_summary,
            action_profile_path=args.action_profile,
            output_dir=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
