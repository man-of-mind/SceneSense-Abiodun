#!/usr/bin/env python3
"""Deterministic Step-5 diagnostics over the frozen 72-action summary.

This module deliberately produces descriptive Pareto labels, not a reduced
action registry.  All 72 technically valid actions remain policy-admissible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SOURCE_SHA256 = (
    "250eb6d9391c0de5a343d692484647bbdb181153ba64f7f4a447b0076f580a88"
)
NETWORK_PROFILES = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
FAMILY_ORDER = ("noAE", "AE128", "AE64", "AE32")
QUANTIZER_ORDER = ("UINT8", "UINT6", "UINT4")
Q_ORDER = (0.0, 0.3, 0.5, 0.7, 0.9, 0.98)


class SurfaceError(RuntimeError):
    """Raised when the frozen source or its action inventory is inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def as_bool(row: dict[str, str], key: str) -> bool:
    value = row.get(key, "")
    if value not in {"True", "False"}:
        raise SurfaceError(f"{key} is not a serialized boolean: {value!r}")
    return value == "True"


def preservation_pass_count(row: dict[str, str]) -> float:
    value = row.get("val_preservation_gates_passed_total", "")
    try:
        passed_text, total_text = value.split("/", 1)
        passed = int(passed_text)
        total = int(total_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SurfaceError(f"invalid preservation count {value!r}") from exc
    if total <= 0 or not 0 <= passed <= total:
        raise SurfaceError(f"unexpected preservation count {value!r}")
    return float(passed)


def quality_maxima(row: dict[str, str]) -> tuple[float, ...]:
    """Registered discrete quality capabilities; every coordinate is maximized."""

    return (
        preservation_pass_count(row),
        as_float(row, "val_absolute_service_gates_passed"),
        as_float(row, "val_localization_gates_passed"),
        float(as_bool(row, "val_segmentation_installable")),
    )


def transport_vectors(
    row: dict[str, str], profiles: Iterable[str] = NETWORK_PROFILES
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return minimized and maximized transport coordinates.

    Conditional install AoI is +infinity when no installation was observed.
    This prevents a zero-delivery action from appearing fast because its
    latency cell is blank.
    """

    minimized = [as_float(row, "live_scientific_inner_bytes_median__across_profile_mean")]
    maximized: list[float] = []
    for profile in profiles:
        maximized.extend(
            (
                as_float(row, f"rate_installed_per_sent__{profile}"),
                as_float(row, f"rate_ack_500ms_per_sent__{profile}"),
                as_float(row, f"rate_reassembled_per_sent__{profile}"),
            )
        )
        aoi = as_float(row, f"install_aoi_ms_median__{profile}", math.inf)
        minimized.append(aoi if math.isfinite(aoi) else math.inf)
    return tuple(minimized), tuple(maximized)


def quality_payload_vectors(
    row: dict[str, str],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        (as_float(row, "live_scientific_inner_bytes_median__across_profile_mean"),),
        quality_maxima(row),
    )


def dominates(
    left: tuple[tuple[float, ...], tuple[float, ...]],
    right: tuple[tuple[float, ...], tuple[float, ...]],
    tolerance: float = 1e-12,
) -> bool:
    left_min, left_max = left
    right_min, right_max = right
    if len(left_min) != len(right_min) or len(left_max) != len(right_max):
        raise SurfaceError("dominance vectors have different dimensions")
    weak = all(a <= b + tolerance for a, b in zip(left_min, right_min)) and all(
        a + tolerance >= b for a, b in zip(left_max, right_max)
    )
    strict = any(a < b - tolerance for a, b in zip(left_min, right_min)) or any(
        a > b + tolerance for a, b in zip(left_max, right_max)
    )
    return weak and strict


def pareto_labels(
    rows: list[dict[str, str]], vector_fn: Any
) -> tuple[set[int], dict[int, list[int]]]:
    front: set[int] = set()
    dominators: dict[int, list[int]] = {}
    for candidate in rows:
        candidate_id = int(candidate["action_id"])
        parents = sorted(
            int(other["action_id"])
            for other in rows
            if other is not candidate
            and dominates(vector_fn(other), vector_fn(candidate))
        )
        dominators[candidate_id] = parents
        if not parents:
            front.add(candidate_id)
    return front, dominators


def joint_vectors(
    row: dict[str, str], profiles: Iterable[str] = NETWORK_PROFILES
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    minimized, maximized = transport_vectors(row, profiles)
    return minimized, maximized + quality_maxima(row)


def require_inventory(rows: list[dict[str, str]]) -> None:
    if len(rows) != 72:
        raise SurfaceError(f"expected 72 action rows, found {len(rows)}")
    action_ids = [int(row["action_id"]) for row in rows]
    if sorted(action_ids) != list(range(72)) or len(set(action_ids)) != 72:
        raise SurfaceError("action IDs are not the exact contiguous range 0..71")
    for row in rows:
        action_id = int(row["action_id"])
        if row.get("profiles_present") != str(len(NETWORK_PROFILES)):
            raise SurfaceError(
                f"action {row['action_id']} does not contain the four frozen profiles"
            )
        if not as_bool(row, "val_join_verified"):
            raise SurfaceError(f"action {row['action_id']} validation join is not verified")
        expected_family = FAMILY_ORDER[action_id // 18]
        expected_quantizer = QUANTIZER_ORDER[(action_id % 18) // 6]
        expected_q = Q_ORDER[action_id % 6]
        if (
            row.get("family") != expected_family
            or row.get("quantizer") != expected_quantizer
            or as_float(row, "q") != expected_q
        ):
            raise SurfaceError(f"action {action_id} factor ordering differs from the catalog")
        required_values = {
            "payload": as_float(
                row, "live_scientific_inner_bytes_median__across_profile_mean"
            ),
            "absolute_service_gates": as_float(
                row, "val_absolute_service_gates_passed"
            ),
            "localization_gates": as_float(row, "val_localization_gates_passed"),
        }
        for profile in NETWORK_PROFILES:
            for prefix in (
                "rate_installed_per_sent",
                "rate_ack_500ms_per_sent",
                "rate_reassembled_per_sent",
                "rate_on_time_100ms_per_sent",
                "sensor_preparation_coverage",
            ):
                required_values[f"{prefix}__{profile}"] = as_float(
                    row, f"{prefix}__{profile}"
                )
        if any(not math.isfinite(value) for value in required_values.values()):
            raise SurfaceError(
                f"action {action_id} has a missing/non-finite required value: "
                f"{required_values}"
            )
        if required_values["payload"] <= 0:
            raise SurfaceError(f"action {action_id} has non-positive payload")
        for profile in NETWORK_PROFILES:
            install_rate = required_values[f"rate_installed_per_sent__{profile}"]
            if install_rate > 0 and not math.isfinite(
                as_float(row, f"install_aoi_ms_median__{profile}")
            ):
                raise SurfaceError(
                    f"action {action_id}/{profile} delivered but has no install AoI"
                )


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path, output: Path) -> dict[str, Any]:
    source_declared = str(source)
    source = source.resolve(strict=True)
    source_sha256 = sha256_file(source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SurfaceError(
            f"source digest mismatch: {source_sha256} != {EXPECTED_SOURCE_SHA256}"
        )
    if output.exists():
        raise SurfaceError(f"create-only output already exists: {output}")

    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require_inventory(rows)
    rows.sort(key=lambda item: int(item["action_id"]))
    quality_front, quality_dominators = pareto_labels(rows, quality_payload_vectors)
    transport_front, transport_dominators = pareto_labels(rows, transport_vectors)
    joint_front, joint_dominators = pareto_labels(rows, joint_vectors)

    profile_fronts: dict[str, set[int]] = {}
    profile_dominators: dict[str, dict[int, list[int]]] = {}
    for profile in NETWORK_PROFILES:
        front, parents = pareto_labels(
            rows, lambda row, selected=profile: joint_vectors(row, (selected,))
        )
        profile_fronts[profile] = front
        profile_dominators[profile] = parents

    action_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    for row in rows:
        action_id = int(row["action_id"])
        q = as_float(row, "q")
        emergency = as_bool(row, "val_emergency_only") or as_bool(
            row, "val_stress_or_emergency_anchor"
        )
        zero_profile_count = int(row["zero_delivery_profile_count"])
        action_rows.append(
            {
                "action_id": action_id,
                "profile_id": row["profile_id"],
                "family": row["family"],
                "quantizer": row["quantizer"],
                "q": q,
                "measured_payload_bytes_mean_of_profile_medians": as_float(
                    row, "live_scientific_inner_bytes_median__across_profile_mean"
                ),
                "preservation_gates_passed": int(preservation_pass_count(row)),
                "absolute_service_gates_passed": int(
                    as_float(row, "val_absolute_service_gates_passed")
                ),
                "localization_gates_passed": int(
                    as_float(row, "val_localization_gates_passed")
                ),
                "segmentation_installable": as_bool(
                    row, "val_segmentation_installable"
                ),
                "emergency_or_stress": emergency,
                "zero_delivery_profile_count": zero_profile_count,
                "positive_delivery_all_four_profiles": zero_profile_count == 0,
                "quality_payload_pareto": action_id in quality_front,
                "transport_robust_pareto": action_id in transport_front,
                "joint_robust_pareto": action_id in joint_front,
                "quality_transport_intersection": (
                    action_id in quality_front and action_id in transport_front
                ),
                "quality_payload_dominators": "|".join(
                    str(value) for value in quality_dominators[action_id]
                ),
                "transport_robust_dominators": "|".join(
                    str(value) for value in transport_dominators[action_id]
                ),
                "joint_robust_dominators": "|".join(
                    str(value) for value in joint_dominators[action_id]
                ),
                "policy_technical_mask_allowed": True,
                "mask_reason": "",
            }
        )
        for profile in NETWORK_PROFILES:
            install_rate = as_float(row, f"rate_installed_per_sent__{profile}")
            conditional_rows.append(
                {
                    "action_id": action_id,
                    "profile_id": row["profile_id"],
                    "network_profile": profile,
                    "q": q,
                    "emergency_or_stress": emergency,
                    "segmentation_installable": as_bool(
                        row, "val_segmentation_installable"
                    ),
                    "sensor_preparation_coverage": as_float(
                        row, f"sensor_preparation_coverage__{profile}"
                    ),
                    "sensor_preparation_coverage_meets_registered_0p95": as_float(
                        row, f"sensor_preparation_coverage__{profile}"
                    )
                    >= 0.95,
                    "rate_reassembled_per_sent": as_float(
                        row, f"rate_reassembled_per_sent__{profile}"
                    ),
                    "rate_installed_per_sent": install_rate,
                    "rate_ack_500ms_per_sent": as_float(
                        row, f"rate_ack_500ms_per_sent__{profile}"
                    ),
                    "rate_on_time_100ms_per_sent": as_float(
                        row, f"rate_on_time_100ms_per_sent__{profile}"
                    ),
                    "install_aoi_ms_median": row.get(
                        f"install_aoi_ms_median__{profile}", ""
                    ),
                    "observed_delivery_label": (
                        "POSITIVE_DELIVERY" if install_rate > 0 else "ZERO_DELIVERY"
                    ),
                    "observed_any_100ms_install": as_float(
                        row, f"rate_on_time_100ms_per_sent__{profile}"
                    )
                    > 0,
                    "profile_joint_pareto": action_id in profile_fronts[profile],
                    "profile_joint_dominators": "|".join(
                        str(value)
                        for value in profile_dominators[profile][action_id]
                    ),
                    "policy_technical_mask_allowed": True,
                    "mask_reason": "",
                }
            )

    output.mkdir(parents=True, exist_ok=False)
    action_fields = list(action_rows[0])
    conditional_fields = list(conditional_rows[0])
    action_path = output / "action_dominance_diagnostics.csv"
    conditional_path = output / "conditional_outcome_labels.csv"
    write_csv(action_path, action_fields, action_rows)
    write_csv(conditional_path, conditional_fields, conditional_rows)

    intersection = sorted(quality_front & transport_front)
    summary = {
        "schema": "scenesense.splitfusion.step5_action_surface.v1",
        "source": source_declared,
        "source_sha256": source_sha256,
        "scientific_scope": "DESCRIPTIVE_DIAGNOSTIC_NOT_ACTION_PRUNING",
        "dominance_warning": (
            "One route realization per action/profile provides no replicate-based "
            "uncertainty test. Exact dominance is descriptive and never a policy mask."
        ),
        "action_count": len(rows),
        "conditional_cell_count": len(conditional_rows),
        "technical_mask_allowed_count": sum(
            int(item["policy_technical_mask_allowed"]) for item in action_rows
        ),
        "quality_payload_pareto_action_ids": sorted(quality_front),
        "transport_robust_pareto_action_ids": sorted(transport_front),
        "joint_robust_pareto_action_ids": sorted(joint_front),
        "quality_transport_intersection_action_ids": intersection,
        "profile_joint_pareto_action_ids": {
            profile: sorted(profile_fronts[profile]) for profile in NETWORK_PROFILES
        },
        "positive_delivery_all_four_profiles_count": sum(
            int(item["positive_delivery_all_four_profiles"]) for item in action_rows
        ),
        "positive_delivery_all_four_non_emergency_count": sum(
            int(item["positive_delivery_all_four_profiles"])
            and not int(item["emergency_or_stress"])
            for item in action_rows
        ),
        "zero_delivery_cells": sum(
            item["observed_delivery_label"] == "ZERO_DELIVERY"
            for item in conditional_rows
        ),
        "observed_100ms_install_cells": sum(
            int(item["observed_any_100ms_install"]) for item in conditional_rows
        ),
        "outputs": {},
    }
    for path in (action_path, conditional_path):
        summary["outputs"][path.name] = {
            "sha256": sha256_file(path),
            "rows": len(action_rows) if path == action_path else len(conditional_rows),
        }
    summary_path = output / "step5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.source, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
