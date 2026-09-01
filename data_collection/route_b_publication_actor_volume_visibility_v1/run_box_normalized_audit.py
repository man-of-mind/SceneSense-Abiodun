"""Final corrected experiment: normalise the *audited box statistic*, not pixels.

The first normalisation attempt replaced both the denominator and the numerator:
it swapped `area(B_visible)` for a retained-pixel count, so pose gaps and
rendering holes read as occlusion.  This driver applies the intended
denominator-only correction.

    raw_box_visibility    = area(B_visible) / area(B_full_clipped)
                            (exactly the statistic the original audit validated;
                             0 for no-support actor-frames)
    corrected_visibility  = clamp(raw_box_visibility
                                  / expected_clear_raw_box_visibility, 0, 1)

`expected_clear_raw_box_visibility` is the 95th percentile (`method="higher"`) of
`raw_box_visibility` over the same training-only groups, with the same
50/100/100/global fallback hierarchy, as the previous attempt — only the
aggregated statistic differs, and the reference build proves that by rebuilding
the earlier pixel reference from the same records bit for bit.

Everything about the extraction is untouched: 0.05 m containment tolerance,
0.03 m ground rejection, back-projection, actor-local containment, deterministic
overlap assignment, and the qualification and human-comparison logic.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 -m \
        data_collection.route_b_publication_actor_volume_visibility_v1.run_box_normalized_audit \
        --box-reference-run <id> --pixel-reference-run <id>
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
from . import run_normalized_audit as pixel_audit

TERMINAL_PASS = (
    "BOX_NORMALIZED_ACTOR_VOLUME_VISIBILITY_DEVELOPMENT_PASS"
    "_AWAITING_INDEPENDENT_HUMAN_AUDIT"
)
TERMINAL_FAIL = "BOX_NORMALIZED_ACTOR_VOLUME_VISIBILITY_NOT_FEASIBLE_FINAL_RETAIN_HUMAN_BANDS"
TERMINAL_INVALID = "BOX_NORMALIZED_ACTOR_VOLUME_VISIBILITY_IMPLEMENTATION_INVALID"

OUTPUT_PARENT = pilot.OUTPUT_PARENT / "box_normalized"

METHODS = {
    "box_normalized_actor_volume": ("corrected_visibility", "corrected_band"),
    "pixel_support_normalized": ("pixel_normalized_visibility", "pixel_normalized_band"),
    "unnormalized_actor_volume": ("visibility", "visibility_band"),
    "old_depth_interval": ("old_depth_interval_visible_fraction", "old_depth_interval_band"),
}


def box_reference_integrity(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Reference gates, plus the proof that only the statistic changed."""
    checks = pixel_audit.reference_integrity(artifact)
    equivalence = artifact.get("frozen_support_density_equivalence", {})
    checks.append(
        {
            "check": "reference_statistic_is_raw_box_visibility",
            "passed": artifact["reference"].get("statistic")
            == tref.STATISTIC_RAW_BOX_VISIBILITY,
            "detail": {"statistic": artifact["reference"].get("statistic")},
        }
    )
    checks.append(
        {
            "check": "extraction_unchanged_reproduces_frozen_pixel_reference",
            "passed": bool(
                equivalence.get("checked")
                and equivalence.get("total_records_match")
                and equivalence.get("tables_identical")
            ),
            "detail": equivalence,
        }
    )
    return checks


def normalize(
    scored: pd.DataFrame,
    data: dict[str, Any],
    box_artifact: dict[str, Any],
    pixel_artifact: dict[str, Any],
) -> pd.DataFrame:
    """Attach the corrected box-normalised score and the pixel-normalised one."""
    box_reference = box_artifact["reference"]
    pixel_reference = pixel_artifact["reference"]
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
        actor_type = str(target.gt_actor_type_id)

        raw_box = tref.raw_box_visibility(
            float(record.visible_box_area_px), float(record.clipped_projected_area_px)
        )
        # The numerator must be the exact statistic the original audit validated.
        if abs(raw_box - float(record.visibility)) > 1e-12:
            raise pilot.AuditError(
                f"raw box visibility diverges from the audited score for {record.sample_id}"
            )
        expected_box, box_tier, box_n, box_key = tref.lookup(
            box_reference, actor_type, angle_bin, height_bin
        )
        corrected = tref.normalized_visibility(raw_box, expected_box)

        density = tref.support_density(
            int(record.retained_actor_point_count),
            float(record.clipped_projected_area_px),
        )
        expected_pixel, pixel_tier, _pixel_n, _pixel_key = tref.lookup(
            pixel_reference, actor_type, angle_bin, height_bin
        )
        pixel_value = tref.normalized_visibility(density, expected_pixel)

        rows.append(
            {
                "sample_id": record.sample_id,
                "gt_actor_id": record.gt_actor_id,
                "actor_type": actor_type,
                "folded_view_angle_deg": angle,
                "angle_bin": angle_bin,
                "height_bin": height_bin,
                "raw_box_visibility": raw_box,
                "expected_clear_raw_box_visibility": expected_box,
                "reference_tier": box_tier,
                "reference_group_n": box_n,
                "reference_group_key": box_key,
                "corrected_visibility": corrected,
                "corrected_band": core.band_for_score(corrected),
                "support_density": density,
                "expected_clear_support_density": expected_pixel,
                "pixel_reference_tier": pixel_tier,
                "pixel_normalized_visibility": pixel_value,
                "pixel_normalized_band": core.band_for_score(pixel_value),
            }
        )
    return scored.merge(
        pd.DataFrame(rows), on=["sample_id", "gt_actor_id"], validate="one_to_one"
    )


def build_agreement(scored: pd.DataFrame, human: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = human[["panel_number", "sample_id", "visibility_band", "truncation_label"]].rename(
        columns={"visibility_band": "human_band"}
    )
    merged = scored.merge(labels, on="sample_id", validate="one_to_one")
    scoreable = merged[merged.human_band != "ambiguous"].copy()

    report: dict[str, Any] = {
        "annotated_samples": int(len(merged)),
        "ambiguous_excluded": int((merged.human_band == "ambiguous").sum()),
        "scoreable_samples": int(len(scoreable)),
        "human_band_counts": merged.human_band.value_counts().to_dict(),
    }
    for name, (score_column, band_column) in METHODS.items():
        report[name] = agreement_mod.evaluate(
            scoreable.human_band.tolist(),
            scoreable[score_column].tolist(),
            scoreable[band_column].tolist(),
        )

    by_band: dict[str, Any] = {}
    for band in core.BAND_ORDER:
        subset = scoreable[scoreable.human_band == band]
        entry = {"n": int(len(subset))}
        for name, (score_column, _band) in METHODS.items():
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

    medians = [by_band[b]["box_normalized_actor_volume"]["median"] for b in core.BAND_ORDER]
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
            "median_corrected_visibility": float(subset.corrected_visibility.median()),
        }
        for method, (score_column, band_column) in METHODS.items():
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
        "expected_clear_min": float(merged.expected_clear_raw_box_visibility.min()),
        "expected_clear_max": float(merged.expected_clear_raw_box_visibility.max()),
        "expected_clear_all_finite_positive": bool(
            np.all(np.isfinite(merged.expected_clear_raw_box_visibility))
            and bool((merged.expected_clear_raw_box_visibility > 0.0).all())
        ),
        "samples_at_clamp_ceiling": int((merged.corrected_visibility >= 1.0).sum()),
    }
    return report, merged


def largest_disagreements(merged: pd.DataFrame, count: int = pixel_audit.LARGEST_DISAGREEMENTS):
    edges = {name: (low, high) for low, high, name in core.BAND_EDGES}
    scoreable = merged[merged.human_band != "ambiguous"].copy()
    ranks = {name: index for index, name in enumerate(core.BAND_ORDER)}
    scoreable["band_distance"] = [
        abs(ranks[auto] - ranks[human])
        for auto, human in zip(scoreable.corrected_band, scoreable.human_band)
    ]

    def gap(row: Any) -> float:
        low, high = edges[row.human_band]
        value = float(row.corrected_visibility)
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    scoreable["score_gap"] = [gap(row) for row in scoreable.itertuples()]
    return scoreable[scoreable.band_distance > 0].sort_values(
        ["band_distance", "score_gap", "panel_number"], ascending=[False, False, True]
    ).head(count)


def decide(qualification: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    new = report["box_normalized_actor_volume"]
    unnormalized = report["unnormalized_actor_volume"]
    fallback = report["fallback_usage_on_pilot"]
    gates = {
        "all_qualification_and_leakage_checks_pass": all(
            check["passed"] for check in qualification
        ),
        "expected_clear_all_finite_and_positive": bool(
            fallback["expected_clear_all_finite_positive"]
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
            new["linear_weighted_cohen_kappa"] >= unnormalized["linear_weighted_cohen_kappa"]
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
    parser.add_argument("--box-reference-run", required=True)
    parser.add_argument("--pixel-reference-run", required=True)
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
    box_artifact, box_provenance = pixel_audit.load_frozen_reference(args.box_reference_run)
    pixel_artifact, pixel_provenance = pixel_audit.load_frozen_reference(args.pixel_reference_run)

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
    qualification.extend(box_reference_integrity(box_artifact))

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

    scored = normalize(scored, data, box_artifact, pixel_artifact)
    report, merged = build_agreement(scored, data["human"])
    decision = decide(qualification, report)

    scores_path = run_dir / "box_normalized_visibility_scores.csv"
    merged_path = run_dir / "box_normalized_visibility_with_human_bands.csv"
    scored.to_csv(scores_path, index=False)
    merged.to_csv(merged_path, index=False)

    from .contact_sheet import build_disagreement_sheet

    top = largest_disagreements(merged)
    display = top.copy()
    display["visibility"] = display["corrected_visibility"]
    display["auto_band"] = display["corrected_band"]
    sheet = build_disagreement_sheet(
        display, pilot.DATASET_DIR, run_dir / "largest_disagreements.png",
        pilot.VIEW_DIR / "object_boxes_all.csv",
    )
    top[
        [
            "panel_number", "sample_id", "gt_actor_id", "distance_m", "distance_band",
            "human_band", "corrected_band", "corrected_visibility", "raw_box_visibility",
            "expected_clear_raw_box_visibility", "reference_tier",
            "pixel_normalized_visibility", "old_depth_interval_visible_fraction",
            "band_distance", "score_gap", "truncation", "truncation_label", "no_support",
        ]
    ].to_csv(run_dir / "largest_disagreements.csv", index=False)

    metadata = {
        "schema": "route_b_publication_box_normalized_actor_volume_visibility_v1",
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
        "box_reference_provenance": box_provenance,
        "pixel_reference_provenance": pixel_provenance,
        "training_reference_summary": {
            "statistic": box_artifact["reference"]["statistic"],
            "total_records": box_artifact["reference"]["total_records"],
            "group_counts": box_artifact["group_counts"],
            "extraction_stats": box_artifact["extraction_stats"],
            "training_provenance": box_artifact["training_provenance"],
            "frozen_support_density_equivalence": box_artifact[
                "frozen_support_density_equivalence"
            ],
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
                "run_box_normalized_audit.py",
            )
        },
        "wall_seconds": time.perf_counter() - started,
    }
    (run_dir / "RUN_METADATA.json").write_text(json.dumps(metadata, indent=2, default=str))
    (run_dir / "ARTIFACT_HASHES.json").write_text(
        json.dumps(
            {p.name: pilot.sha256_file(p) for p in sorted(run_dir.iterdir()) if p.is_file()},
            indent=2, sort_keys=True,
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
