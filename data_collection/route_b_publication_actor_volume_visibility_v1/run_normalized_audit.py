"""Part 2: score the 100 pilot panels with the training-normalised denominator.

The actor-volume extraction is reused verbatim from the pilot — same 0.05 m
containment tolerance, same 0.03 m ground rejection, same back-projection, same
actor-local containment, same deterministic overlapping-actor assignment.  Only
the denominator changes:

    support_density   = retained_actor_volume_pixels / clipped_projected_box_area
    normalized_visibility = clamp(support_density / expected_clear_support_density, 0, 1)

`expected_clear_support_density` comes from the frozen training-only reference
built by `build_training_reference.py`, whose sha256 is verified here before it
is used.  Human labels never touch the reference or the score; they enter only
at the agreement stage.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 -m \
        data_collection.route_b_publication_actor_volume_visibility_v1.run_normalized_audit \
        --reference-run <training reference run id>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import agreement as agreement_mod
from . import core, training_reference as tref
from . import run_audit as pilot

TERMINAL_PASS = (
    "TRAIN_NORMALIZED_ACTOR_VOLUME_VISIBILITY_DEVELOPMENT_PASS"
    "_AWAITING_INDEPENDENT_HUMAN_AUDIT"
)
TERMINAL_FAIL = "TRAIN_NORMALIZED_ACTOR_VOLUME_VISIBILITY_NOT_FEASIBLE_RETAIN_HUMAN_BANDS"
TERMINAL_INVALID = "TRAIN_NORMALIZED_ACTOR_VOLUME_VISIBILITY_IMPLEMENTATION_INVALID"

OUTPUT_PARENT = pilot.OUTPUT_PARENT / "normalized"
REFERENCE_PARENT = pilot.OUTPUT_PARENT / "training_reference"
LARGEST_DISAGREEMENTS = 20


def load_frozen_reference(reference_run: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the training reference and verify it against its recorded hash."""
    directory = REFERENCE_PARENT / reference_run
    reference_path = directory / "training_reference.json"
    recorded = json.loads((directory / "REFERENCE_HASHES.json").read_text())
    actual = pilot.sha256_file(reference_path)
    if actual != recorded["training_reference.json"]:
        raise pilot.AuditError(
            f"training reference hash mismatch: {actual} != {recorded['training_reference.json']}"
        )
    artifact = json.loads(reference_path.read_text())
    return artifact, {
        "reference_run": reference_run,
        "training_reference_sha256": actual,
        "training_support_records_sha256": recorded["training_support_records.csv"],
        "reference_version": artifact["reference"]["reference_version"],
    }


def reference_integrity(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Gates on the frozen reference itself, independent of any score."""
    leakage = artifact["leakage_proof"]
    non_positive = artifact["non_positive_reference_groups"]
    checks = [
        {
            "check": "training_reference_has_no_validation_or_test_leakage",
            "passed": bool(
                leakage["validation_sample_ids_in_population"] == 0
                and leakage["validation_sample_ids_in_records"] == 0
                and leakage["test_rows_read"] == 0
                and leakage["human_annotation_files_read"] == 0
                and artifact["training_provenance"]["dataset_manifest_split_counts"].get("test", 0)
                == 0
            ),
            "detail": leakage,
        },
        {
            "check": "training_reference_built_without_human_annotation",
            "passed": bool(
                artifact["human_annotation_files_read"] == 0
                and artifact["human_pilot_directory_opened"] is False
                and artifact["model_or_prediction_artifacts_read"] == 0
            ),
            "detail": {
                "human_annotation_files_read": artifact["human_annotation_files_read"],
                "human_pilot_directory_opened": artifact["human_pilot_directory_opened"],
            },
        },
        {
            "check": "extraction_constants_unchanged_from_pilot",
            "passed": bool(
                artifact["extraction_constants"]["actor_volume_tolerance_m"]
                == core.ACTOR_VOLUME_TOLERANCE_M
                and artifact["extraction_constants"]["ground_reject_margin_m"]
                == core.GROUND_REJECT_MARGIN_M
                and artifact["extraction_constants"]["algorithm_version"]
                == core.ALGORITHM_VERSION
            ),
            "detail": artifact["extraction_constants"],
        },
        {
            "check": "all_reference_values_finite_and_positive",
            "passed": all(len(v) == 0 for v in non_positive.values()),
            "detail": {tier: len(v) for tier, v in non_positive.items()},
        },
    ]
    return checks


def normalize(scored: pd.DataFrame, data: dict[str, Any], artifact: dict[str, Any]) -> pd.DataFrame:
    """Attach support density, the reference lookup, and the normalised score."""
    reference = artifact["reference"]
    manifest = data["dataset_manifest"].set_index("sample_id")
    boxes = data["object_boxes"].set_index(["sample_id", "gt_actor_id"])

    rows: list[dict[str, Any]] = []
    for record in scored.itertuples():
        meta = manifest.loc[record.sample_id]
        target = boxes.loc[(record.sample_id, record.gt_actor_id)]
        camera_position = np.asarray(
            json.loads(meta.camera_matrix_json), dtype=np.float64
        )[:3, 3]
        angle = tref.folded_view_angle_deg(
            (target.object_world_x, target.object_world_y, target.object_world_z),
            float(target.object_yaw_deg),
            camera_position,
        )
        angle_bin = tref.angle_bin(angle)
        height_bin = tref.height_bin(float(record.clipped_bbox_h))
        density = tref.support_density(
            int(record.retained_actor_point_count),
            float(record.clipped_projected_area_px),
        )
        expected, tier, group_n, group_key = tref.lookup(
            reference, str(target.gt_actor_type_id), angle_bin, height_bin
        )
        value = tref.normalized_visibility(density, expected)
        rows.append(
            {
                "sample_id": record.sample_id,
                "gt_actor_id": record.gt_actor_id,
                "actor_type": str(target.gt_actor_type_id),
                "folded_view_angle_deg": angle,
                "angle_bin": angle_bin,
                "height_bin": height_bin,
                "support_density": density,
                "expected_clear_support_density": expected,
                "reference_tier": tier,
                "reference_group_n": group_n,
                "reference_group_key": group_key,
                "normalized_visibility": value,
                "normalized_band": core.band_for_score(value),
            }
        )
    return scored.merge(pd.DataFrame(rows), on=["sample_id", "gt_actor_id"], validate="one_to_one")


def build_agreement(scored: pd.DataFrame, human: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    # `scored` already carries a `visibility_band` from the pilot metric, so the
    # human column is renamed before the merge rather than relying on suffixes.
    labels = human[["panel_number", "sample_id", "visibility_band", "truncation_label"]].rename(
        columns={"visibility_band": "human_band"}
    )
    merged = scored.merge(labels, on="sample_id", validate="one_to_one")
    scoreable = merged[merged.human_band != "ambiguous"].copy()

    methods = {
        "normalized_actor_volume": ("normalized_visibility", "normalized_band"),
        "unnormalized_actor_volume": ("visibility", "visibility_band"),
        "old_depth_interval": (
            "old_depth_interval_visible_fraction",
            "old_depth_interval_band",
        ),
    }
    report: dict[str, Any] = {
        "annotated_samples": int(len(merged)),
        "ambiguous_excluded": int((merged.human_band == "ambiguous").sum()),
        "scoreable_samples": int(len(scoreable)),
        "human_band_counts": merged.human_band.value_counts().to_dict(),
    }
    for name, (score_column, band_column) in methods.items():
        report[name] = agreement_mod.evaluate(
            scoreable.human_band.tolist(),
            scoreable[score_column].tolist(),
            scoreable[band_column].tolist(),
        )

    by_band: dict[str, Any] = {}
    for band in core.BAND_ORDER:
        subset = scoreable[scoreable.human_band == band]
        entry = {"n": int(len(subset))}
        for name, (score_column, _band) in methods.items():
            values = subset[score_column]
            entry[name] = {
                "median": float(values.median()) if len(subset) else float("nan"),
                "mean": float(values.mean()) if len(subset) else float("nan"),
                "p25": float(values.quantile(0.25)) if len(subset) else float("nan"),
                "p75": float(values.quantile(0.75)) if len(subset) else float("nan"),
                "min": float(values.min()) if len(subset) else float("nan"),
                "max": float(values.max()) if len(subset) else float("nan"),
            }
        by_band[band] = entry
    report["score_distribution_by_human_band"] = by_band

    medians = [by_band[b]["normalized_actor_volume"]["median"] for b in core.BAND_ORDER]
    report["medians_in_band_order"] = medians
    report["median_monotonic_increasing"] = bool(
        all(
            math.isfinite(a) and math.isfinite(b) and b > a
            for a, b in zip(medians, medians[1:])
        )
    )

    by_distance: dict[str, Any] = {}
    for name, low, high in pilot.DISTANCE_BANDS:
        subset = scoreable[(scoreable.distance_m >= low) & (scoreable.distance_m < high)]
        if len(subset) == 0:
            by_distance[name] = {"n": 0}
            continue
        entry = {
            "n": int(len(subset)),
            "median_normalized_visibility": float(subset.normalized_visibility.median()),
        }
        for method, (score_column, band_column) in methods.items():
            entry[method] = agreement_mod.evaluate(
                subset.human_band.tolist(),
                subset[score_column].tolist(),
                subset[band_column].tolist(),
            )
        by_distance[name] = entry
    report["by_distance_band"] = by_distance

    report["fallback_usage_on_pilot"] = {
        "all_samples": merged.reference_tier.value_counts().to_dict(),
        "scoreable_samples": scoreable.reference_tier.value_counts().to_dict(),
        "group_n_min": int(merged.reference_group_n.min()),
        "expected_support_min": float(merged.expected_clear_support_density.min()),
        "expected_support_max": float(merged.expected_clear_support_density.max()),
        "expected_support_all_finite_positive": bool(
            np.all(np.isfinite(merged.expected_clear_support_density))
            and bool((merged.expected_clear_support_density > 0.0).all())
        ),
    }
    return report, merged


def largest_disagreements(merged: pd.DataFrame, count: int = LARGEST_DISAGREEMENTS) -> pd.DataFrame:
    """Rank non-ambiguous samples by ordinal band distance, then by score gap.

    The score gap is how far the normalised score sits outside the human band's
    own interval, so ties on band distance are broken by how badly the score
    misses rather than arbitrarily.
    """
    edges = {name: (low, high) for low, high, name in core.BAND_EDGES}
    scoreable = merged[merged.human_band != "ambiguous"].copy()
    ranks = {name: index for index, name in enumerate(core.BAND_ORDER)}
    scoreable["band_distance"] = [
        abs(ranks[auto] - ranks[human])
        for auto, human in zip(scoreable.normalized_band, scoreable.human_band)
    ]

    def gap(row: Any) -> float:
        low, high = edges[row.human_band]
        value = float(row.normalized_visibility)
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    scoreable["score_gap"] = [gap(row) for row in scoreable.itertuples()]
    ordered = scoreable[scoreable.band_distance > 0].sort_values(
        ["band_distance", "score_gap", "panel_number"], ascending=[False, False, True]
    )
    return ordered.head(count)


def decide(qualification: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    new = report["normalized_actor_volume"]
    unnormalized = report["unnormalized_actor_volume"]
    fallback = report["fallback_usage_on_pilot"]
    gates = {
        "all_qualification_checks_pass": all(check["passed"] for check in qualification),
        "expected_support_all_finite_and_positive": bool(
            fallback["expected_support_all_finite_positive"]
        ),
        "median_visibility_monotonic": bool(report["median_monotonic_increasing"]),
        "weighted_kappa_at_least_0_60": bool(
            math.isfinite(new["linear_weighted_cohen_kappa"])
            and new["linear_weighted_cohen_kappa"] >= pilot.MIN_WEIGHTED_KAPPA
        ),
        "balanced_accuracy_at_least_0_80": bool(
            math.isfinite(new["balanced_accuracy"])
            and new["balanced_accuracy"] >= pilot.MIN_BALANCED_ACCURACY
        ),
        "not_worse_than_unnormalized_on_kappa": bool(
            new["linear_weighted_cohen_kappa"]
            >= unnormalized["linear_weighted_cohen_kappa"]
        ),
        "not_worse_than_unnormalized_on_balanced_accuracy": bool(
            new["balanced_accuracy"] >= unnormalized["balanced_accuracy"]
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "thresholds": {
            "min_weighted_kappa": pilot.MIN_WEIGHTED_KAPPA,
            "min_balanced_accuracy": pilot.MIN_BALANCED_ACCURACY,
        },
        "passed": passed,
        "terminal": TERMINAL_PASS if passed else TERMINAL_FAIL,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise pilot.AuditError('refusing to run without CUDA_VISIBLE_DEVICES="" (CPU-only)')
    if "torch" in sys.modules:
        raise pilot.AuditError("torch is imported; this audit must stay prediction-blind")

    started = time.perf_counter()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_PARENT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # create-only

    decoder = pilot.assert_registered_decoder()
    artifact, reference_provenance = load_frozen_reference(args.reference_run)

    # Every pilot qualification check is re-run unchanged, then the reference
    # integrity gates are appended.
    data = pilot.load_inputs()
    qualification = pilot.qualify_provenance(data)
    identity = pilot.frame_identity_checks(
        data["sample_manifest"], data["dataset_manifest"], data["object_visibility"]
    )
    qualification.append(
        {
            "check": "depth_rgb_metadata_frame_identity",
            "passed": bool(
                identity["sample_id_frame_suffix_matches_frame_id"]
                and identity["rgb_frame_id_equals_frame_id"]
                and identity["depth_frame_id_equals_frame_id"]
                and identity["dataset_manifest_frame_id_matches"]
                and identity["visibility_frame_id_matches"]
                and identity["max_rgb_depth_timestamp_delta_s"] <= pilot.MAX_TIMESTAMP_DELTA_S
                and identity["max_dataset_vs_depth_timestamp_delta_s"] <= pilot.MAX_TIMESTAMP_DELTA_S
                and identity["max_dataset_vs_visibility_timestamp_delta_s"]
                <= pilot.MAX_TIMESTAMP_DELTA_S
                and identity["max_pilot_vs_dataset_depth_timestamp_delta_s"]
                <= pilot.MAX_TIMESTAMP_DELTA_S
            ),
            "detail": identity,
        }
    )
    qualification.extend(reference_integrity(artifact))

    failures = [check["check"] for check in qualification if not check["passed"]]
    if failures:
        payload = {"terminal": TERMINAL_INVALID, "failed_checks": failures,
                   "qualification": qualification}
        (run_dir / "QUALIFICATION_FAILED.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        print(TERMINAL_INVALID)
        return 2

    scored, diagnostics = pilot.score_all(data)
    qualification.extend(pilot.qualify_geometry(diagnostics))
    failures = [check["check"] for check in qualification if not check["passed"]]
    if failures:
        payload = {"terminal": TERMINAL_INVALID, "failed_checks": failures,
                   "qualification": qualification}
        (run_dir / "QUALIFICATION_FAILED.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        print(TERMINAL_INVALID)
        return 2

    scored = normalize(scored, data, artifact)
    report, merged = build_agreement(scored, data["human"])
    decision = decide(qualification, report)

    scores_path = run_dir / "normalized_visibility_scores.csv"
    merged_path = run_dir / "normalized_visibility_with_human_bands.csv"
    scored.to_csv(scores_path, index=False)
    merged.to_csv(merged_path, index=False)

    # Overlays for the twenty largest disagreements.  The panel renderer reads
    # `visibility` / `auto_band`, so the normalised columns are mapped onto them.
    from .contact_sheet import build_disagreement_sheet

    top = largest_disagreements(merged)
    display = top.copy()
    display["visibility"] = display["normalized_visibility"]
    display["auto_band"] = display["normalized_band"]
    sheet = build_disagreement_sheet(
        display, pilot.DATASET_DIR, run_dir / "largest_disagreements.png",
        pilot.VIEW_DIR / "object_boxes_all.csv",
    )
    top_path = run_dir / "largest_disagreements.csv"
    top[
        [
            "panel_number", "sample_id", "gt_actor_id", "distance_m", "distance_band",
            "human_band", "normalized_band", "normalized_visibility", "visibility",
            "old_depth_interval_visible_fraction", "support_density",
            "expected_clear_support_density", "reference_tier", "band_distance",
            "score_gap", "truncation", "truncation_label", "no_support",
        ]
    ].to_csv(top_path, index=False)

    metadata = {
        "schema": "route_b_publication_train_normalized_actor_volume_visibility_v1",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "model_or_prediction_artifacts_read": 0,
        "test_rows_read": 0,
        "carla_started": False,
        "fcos_or_lraspp_rescored": False,
        "depth_images_opened": diagnostics["depth_images_opened"],
        "decoder": decoder,
        "reference_provenance": reference_provenance,
        "training_reference_summary": {
            "total_records": artifact["reference"]["total_records"],
            "group_counts": artifact["group_counts"],
            "extraction_stats": artifact["extraction_stats"],
            "training_provenance": artifact["training_provenance"],
        },
        "constants": {
            "actor_volume_tolerance_m": core.ACTOR_VOLUME_TOLERANCE_M,
            "ground_reject_margin_m": core.GROUND_REJECT_MARGIN_M,
            "reference_percentile": tref.REFERENCE_PERCENTILE,
            "percentile_method": tref.PERCENTILE_METHOD,
            "min_n": {
                tref.TIER_TYPE_ANGLE_HEIGHT: tref.MIN_N_TYPE_ANGLE_HEIGHT,
                tref.TIER_ANGLE_HEIGHT: tref.MIN_N_ANGLE_HEIGHT,
                tref.TIER_HEIGHT: tref.MIN_N_HEIGHT,
            },
            "band_edges": [list(edge) for edge in core.BAND_EDGES],
        },
        "qualification": qualification,
        "frame_identity": identity,
        "geometry_diagnostics": diagnostics,
        "agreement": report,
        "decision": decision,
        "largest_disagreements_sheet": sheet,
        "code_hashes": {
            name: pilot.sha256_file(Path(__file__).parent / name)
            for name in (
                "core.py", "scoring.py", "agreement.py", "run_audit.py",
                "contact_sheet.py", "training_reference.py",
                "build_training_reference.py", "run_normalized_audit.py",
            )
        },
        "wall_seconds": time.perf_counter() - started,
    }
    (run_dir / "RUN_METADATA.json").write_text(json.dumps(metadata, indent=2, default=str))
    (run_dir / "ARTIFACT_HASHES.json").write_text(
        json.dumps(
            {
                path.name: pilot.sha256_file(path)
                for path in sorted(run_dir.iterdir())
                if path.is_file()
            },
            indent=2,
        )
    )
    (run_dir / decision["terminal"]).write_text(
        f"{decision['terminal']}\n{datetime.now(timezone.utc).isoformat()}\n"
    )

    print(json.dumps({"run_dir": str(run_dir), "decision": decision}, indent=2))
    print(decision["terminal"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
