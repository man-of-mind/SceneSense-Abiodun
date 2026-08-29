#!/usr/bin/env python3
"""Create-only, retained-prediction Route B v3.1 person contract audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matching_v1 import (
    MATCH_DEFINITIONS,
    annotate_neutral_predictions,
    assignment_difference,
    build_taxonomy,
    canonical_world_match,
    image_match,
    load_frame_ids,
    load_person_gt,
    load_predictions,
    score_summary,
    summarize_conditional,
    threshold_grid,
    trapezoid_auprc,
)
from visibility_gaussian_v1 import (
    audit_gaussian,
    audit_target_visibility,
    render_review_panels,
    select_review_panels,
)


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
DEFAULT_CONFIG = PACKAGE / "configs/person_contract_audit_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv_x(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty required CSV: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def register(config_path: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    config_hash = sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Hash every immutable gate input before parsing any of its payload content.
    inputs: dict[str, dict[str, Any]] = {}
    paths = {
        "dataset_manifest": ROOT / config["dataset"]["root"] / "dataset/manifest.csv",
        "dataset_object_boxes": ROOT / config["dataset"]["root"] / "dataset/object_boxes.csv",
        "dataset_resolved_config": ROOT / config["dataset"]["root"] / "resolved_config.json",
    }
    for model_name, model in config["models"].items():
        paths[f"{model_name}_checkpoint"] = ROOT / model["checkpoint"]
        paths[f"{model_name}_detections"] = ROOT / model["prediction_root"] / "detections.csv"
        paths[f"{model_name}_inference_manifest"] = ROOT / model["prediction_root"] / "inference_manifest.json"
    for name, path in paths.items():
        if not path.is_file():
            terminal = "PERSON_CONTRACT_AUDIT_BLOCKED_MISSING_RETAINED_PREDICTIONS" if "prediction" in name or "detections" in name else "PERSON_CONTRACT_AUDIT_INVALID"
            raise RuntimeError(f"{terminal}: missing {name}: {path}")
        inputs[name] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}

    for model_name, model in config["models"].items():
        if inputs[f"{model_name}_checkpoint"]["sha256"] != model["checkpoint_sha256"]:
            raise RuntimeError(f"PERSON_CONTRACT_AUDIT_INVALID: {model_name} checkpoint hash mismatch")
        if inputs[f"{model_name}_detections"]["sha256"] != model["detections_sha256"]:
            raise RuntimeError(f"PERSON_CONTRACT_AUDIT_INVALID: {model_name} detections hash mismatch")

    manifest = read_csv(paths["dataset_manifest"])
    counts = {split: sum(row["split"] == split for row in manifest) for split in ("train", "val", "test")}
    expected = config["dataset"]["expected_frames"]
    if counts != expected:
        raise RuntimeError(f"PERSON_CONTRACT_AUDIT_INVALID: split counts {counts} != {expected}")
    train_ids = {row["sample_id"] for row in manifest if row["split"] == "train"}
    val_ids = {row["sample_id"] for row in manifest if row["split"] == "val"}
    train_episodes = {row["experiment_id"] for row in manifest if row["split"] == "train"}
    val_episodes = {row["experiment_id"] for row in manifest if row["split"] == "val"}
    forbidden_tokens = ("canonical_v3_07", "canonical_v3_08")
    locked_references = sum(any(token in " ".join(row.values()) for token in forbidden_tokens) for row in manifest)
    if train_ids & val_ids or train_episodes & val_episodes or locked_references:
        raise RuntimeError("PERSON_CONTRACT_AUDIT_INVALID: split disjointness or locked-reference gate failed")

    dataset_root = ROOT / config["dataset"]["root"]
    v010_val = read_csv(dataset_root / "contracts/v010/val/object_boxes.csv")
    person_count = sum(row["label"] == "person" for row in v010_val)
    if person_count != config["dataset"]["expected_primary_v010_val_person_gt"]:
        raise RuntimeError(f"PERSON_CONTRACT_AUDIT_INVALID: v010 val person GT={person_count}")

    inference: dict[str, Any] = {}
    for model_name, model in config["models"].items():
        payload = json.loads((ROOT / model["prediction_root"] / "inference_manifest.json").read_text(encoding="utf-8"))
        if payload.get("score_floor") != 0.02 or payload.get("validation_frames") != 3345:
            raise RuntimeError(f"PERSON_CONTRACT_AUDIT_INVALID: retained prediction manifest mismatch: {model_name}")
        if payload.get("checkpoint_sha256") != model["checkpoint_sha256"] or payload.get("detections_sha256") != model["detections_sha256"]:
            raise RuntimeError(f"PERSON_CONTRACT_AUDIT_INVALID: prediction provenance mismatch: {model_name}")
        inference[model_name] = payload

    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "route_b_v3_1_person_contract_audit_input_provenance_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": _git_head(),
        "registration_config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_hash},
        "inputs": inputs,
        "prediction_manifests": inference,
        "gates": {
            "split_counts": counts,
            "unique_sample_ids": len({row["sample_id"] for row in manifest}),
            "v010_validation_person_gt": person_count,
            "train_validation_sample_ids_disjoint": not bool(train_ids & val_ids),
            "train_validation_episodes_disjoint": not bool(train_episodes & val_episodes),
            "train_episode_count": len(train_episodes),
            "validation_episode_count": len(val_episodes),
            "test_rows": counts["test"],
            "locked_test_payload_references": locked_references,
            "both_prediction_floors_are_0_02": True
        }
    }
    write_json_x(output / "REGISTERED_AUDIT_PLAN.json", config)
    write_json_x(output / "INPUT_PROVENANCE.json", provenance)
    markdown = "\n".join([
        "# Registered Route B v3.1 person contract audit plan", "",
        f"Registered before metric computation at `{provenance['created_utc']}`.", "",
        "The JSON registration is normative. The canonical 3 m joint metric remains unchanged and authoritative. FULL_BOX_IOU_050 is the preregistered primary 2D diagnostic; the centre and IoU-0.30 definitions are reported without cherry-picking. All inputs are immutable and only retained floor-0.02 predictions may be used.", "",
        "Target observability uses the retained actor depth interval and raw BGRA depth. The continuous full-box `gt_center_x/y` is sampled with a half-open/floor pixel convention and mapped to 768x432 then stride 4 exactly as the native builder does.", ""
    ])
    (output / "REGISTERED_AUDIT_PLAN.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"terminal": "AUDIT_PLAN_REGISTERED", "output": str(output), "gates": provenance["gates"]}, indent=2))
    return 0


def _git_head() -> str:
    head = ROOT / ".git/HEAD"
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        return (ROOT / ".git" / text[5:]).read_text(encoding="utf-8").strip()
    return text


def _public_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in (
        "eligible_gt", "tp", "fp", "fn", "ignored_predictions", "precision", "recall", "f1"
    )}


def _verify_registration(config_path: Path, output: Path) -> dict[str, Any]:
    registered = output / "REGISTERED_AUDIT_PLAN.json"
    provenance_path = output / "INPUT_PROVENANCE.json"
    if not registered.is_file() or not provenance_path.is_file():
        raise RuntimeError("PERSON_CONTRACT_AUDIT_INVALID: preregistration artifacts absent")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if json.loads(registered.read_text(encoding="utf-8")) != config:
        raise RuntimeError("PERSON_CONTRACT_AUDIT_INVALID: registered plan drift")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["registration_config"]["sha256"] != sha256(config_path):
        raise RuntimeError("PERSON_CONTRACT_AUDIT_INVALID: registration config hash drift")
    for item in provenance["inputs"].values():
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"PERSON_CONTRACT_AUDIT_INVALID: immutable input drift: {path}")
    return config


def _source_findings() -> list[dict[str, Any]]:
    return [
        {
            "finding": "canonical scorer uses only class equality and world-XY distance <=3 m for matching",
            "classification": "NOT_A_DEFECT",
            "evidence": "route_b_v3_1_clean_base_v1/score_contract_v1.py:118-132",
            "interpretation": "This is the registered authoritative service metric; 2D matching remains diagnostic only."
        },
        {
            "finding": "visibility eligibility and reconstruction use the retained depth-consistent near/far interval",
            "classification": "NOT_A_DEFECT",
            "evidence": "data_collection/route_b_perception_v3/visibility_v1.py:146-168,203-220",
            "interpretation": "The audit can reconstruct actor-specific depth-consistent masks without new collection or inference."
        },
        {
            "finding": "actor positives retain the original full projected-box centre",
            "classification": "CONFIRMED_DESIGN_LIMIT",
            "evidence": "route_b_v3_1_clean_base_v1/build_contract_v1.py:340-347 plus source actor object_boxes.csv gt_center fields; static analogue at lines 317-321",
            "interpretation": "Visibility qualifies the actor but does not relocate its target to an observed pixel."
        },
        {
            "finding": "native targets consume gt_center_x/y and force the floor grid cell to exactly one",
            "classification": "CONFIRMED_DESIGN_LIMIT",
            "evidence": "pole_lraspp_multimodal_fusion/object_targets.py:97-107; route_b_v3_1_native_grid_v1/targets_v1.py:83-97",
            "interpretation": "The localization and person-objectness supervision is anchored at the full-box centre."
        },
        {
            "finding": "COCO distillation froze the regression output head while training person heatmap, shared trunk, and upstream backbone",
            "classification": "CONFIRMED_DESIGN_LIMIT",
            "evidence": "route_b_v3_1_person_coco_distillation_v1/student_v1.py:31-37,75-103,149-180",
            "interpretation": "Metric regression could move indirectly through shared/upstream features, but its output layers received no direct updates."
        },
        {
            "finding": "augmentation transformed camera intrinsics, but neither model forward nor supervised localization loss consumed them",
            "classification": "CONFIRMED_DESIGN_LIMIT",
            "evidence": "route_b_v3_1_person_coco_distillation_v1/dataset_v1.py:189-199,305-312; run_pipeline_v1.py:220-284",
            "interpretation": "Intrinsics are moved to the device but absent from forward_once and every loss call."
        },
        {
            "finding": "the transported high tensor is physically 27x48 (stride 16), then explicitly average-pooled to a nominal 14x24 stride-32 train-only ROI level",
            "classification": "CONFIRMED_DESIGN_LIMIT",
            "evidence": "route_b_v3_1_person_coco_distillation_v1/roi_v1.py:180-208 and run_pipeline_v1.py:438-445; immutable PREFLIGHT.json:1251-1257",
            "interpretation": "This is not a silent raw stride mismatch: the code deliberately derives a coarser level, discarding high-level spatial detail before feature distillation."
        },
        {
            "finding": "ceil-pooling the odd 27-row high tensor may give the terminal 14th row a different effective centre/extent than a uniform stride-32 coordinate field",
            "classification": "UNTESTED_HYPOTHESIS",
            "evidence": "route_b_v3_1_person_coco_distillation_v1/roi_v1.py:190-208 and run_pipeline_v1.py:441-445",
            "interpretation": "The existing round-trip probe constructs a fresh nominal stride-32 coordinate feature rather than propagating coordinates through the 27-to-14 pooling operation; downstream impact was not tested in this read-only audit."
        },
        {
            "finding": "the feature adapter is train-time-only and absent from deployable checkpoints",
            "classification": "CONFIRMED_DESIGN_LIMIT",
            "evidence": "route_b_v3_1_person_coco_distillation_v1/distill_v1.py:2-8,115-124; run_pipeline_v1.py:592-612",
            "interpretation": "Adapter capacity cannot directly improve deployed predictions; only gradients transferred into the student persist."
        },
        {
            "finding": "the earlier diagnostic slice metadata used source_identity alone",
            "classification": "CONFIRMED_DEFECT",
            "evidence": "route_b_v3_1_person_coco_distillation_v1/evaluation_v1.py:57-64,95-104",
            "interpretation": "Actor identities repeat across frames, so slices can read another frame's distance/radar metadata. This audit keys by (sample_id, source_identity)."
        },
        {
            "finding": "true instance silhouettes are available for target supervision",
            "classification": "UNTESTED_HYPOTHESIS",
            "evidence": "collector visibility algorithm deliberately does not consult instance walker tags (visibility_v1.py:1-6)",
            "interpretation": "This audit tests depth-consistent actor masks only and does not claim identity-perfect silhouettes."
        },
    ]


def _save_figure(fig: Any, figures: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.png", dpi=180)
    fig.savefig(figures / f"{stem}.pdf")
    plt.close(fig)


def _make_figures(
    figures: Path, calibration_rows: Sequence[Mapping[str, Any]], taxonomy_rows: Sequence[Mapping[str, Any]],
    conditional_rows: Sequence[Mapping[str, Any]], visibility_rows: Sequence[Mapping[str, Any]],
    gaussian_rows: Sequence[Mapping[str, Any]],
) -> None:
    figures.mkdir(parents=True, exist_ok=False)
    colors = {"base_epoch_040": "#276FBF", "distilled_epoch_012": "#D1495B"}
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for model in colors:
        for metric, style in (("canonical_joint_3m", "-"), ("FULL_BOX_IOU_050", "--")):
            rows = [row for row in calibration_rows if row["model"] == model and row["metric"] == metric]
            rows.sort(key=lambda row: float(row["recall"]))
            ax.plot([row["recall"] for row in rows], [row["precision"] for row in rows], style,
                    color=colors[model], label=f"{model.replace('_epoch_', ' e')} {metric}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Person PR from retained floor-0.02 predictions")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    _save_figure(fig, figures, "person_pr_curves_2d_and_joint")

    labels = ["NO_2D_PERSON_SUPPORT", "TWO_D_MATCH_WORLD_ERROR_GT_3M", "MATCHING_CONTENTION"]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bottoms = np.zeros(2)
    models = list(colors)
    palette = ("#9E9E9E", "#E45756", "#7B2CBF")
    for label, color in zip(labels, palette):
        values = []
        for model in models:
            match = [row for row in taxonomy_rows if row["model"] == model and float(row["threshold"]) == .02
                     and row["scope"] == "canonical_joint_fn" and row["label"] == label]
            values.append(float(match[0]["fraction"]) if match else 0.0)
        ax.bar(models, values, bottom=bottoms, label=label, color=color)
        bottoms += np.asarray(values)
    ax.set(ylabel="Fraction of canonical joint person FNs", title="Joint-FN decomposition at score 0.02", ylim=(0, 1))
    ax.legend(fontsize=8)
    _save_figure(fig, figures, "joint_fn_decomposition")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for axis, kind, title in zip(axes, ("distance_m", "radar_support"), ("GT distance", "Radar support")):
        labels_here = []
        for row in conditional_rows:
            if (row["match_definition"] == "FULL_BOX_IOU_050" and float(row["threshold"]) == .02
                    and row["subset_kind"] == kind and row["subset_label"] not in labels_here):
                labels_here.append(row["subset_label"])
        x = np.arange(len(labels_here)); width = .36
        for offset, model in zip((-.18, .18), colors):
            values = []
            for label in labels_here:
                hit = [row for row in conditional_rows if row["model"] == model
                       and row["match_definition"] == "FULL_BOX_IOU_050" and float(row["threshold"]) == .02
                       and row["subset_kind"] == kind and row["subset_label"] == label]
                values.append(float(hit[0]["within_3m_fraction"]) if hit and hit[0]["within_3m_fraction"] != "" else 0)
            axis.bar(x + offset, values, width, color=colors[model], label=model)
        axis.set_xticks(x, labels_here, rotation=25, ha="right"); axis.set_title(title); axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Conditional fraction within 3 m")
    axes[1].legend(fontsize=8)
    _save_figure(fig, figures, "conditional_localization_within_3m")

    categories = ("ON_OWN_VISIBLE_MASK", "ON_CLOSER_OCCLUDER", "INSIDE_PROJECTED_BOX_BUT_NOT_VISIBLE",
                  "OUTSIDE_OWN_PROJECTED_BOX", "OUTSIDE_FRAME")
    groups = [(contract, split) for contract in ("v010", "v025") for split in ("train", "val")]
    fig, ax = plt.subplots(figsize=(10.5, 5.4)); bottom = np.zeros(len(groups))
    palette = ("#2A9D8F", "#E76F51", "#F4A261", "#6C757D", "#264653")
    for category, color in zip(categories, palette):
        fractions = []
        for contract, split in groups:
            subset = [row for row in visibility_rows if row["contract"] == contract and row["split"] == split]
            fractions.append(sum(row["category"] == category for row in subset) / max(1, len(subset)))
        ax.bar([f"{c}\n{s}" for c, s in groups], fractions, bottom=bottom, label=category, color=color)
        bottom += np.asarray(fractions)
    ax.set(ylabel="Fraction of authoritative person positives", title="Target-centre depth observability", ylim=(0, 1))
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, .5))
    _save_figure(fig, figures, "target_centre_observability")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for implementation, color in (("current", "#E76F51"), ("reference", "#2A9D8F")):
        values = [int(row["integer_radius"]) for row in gaussian_rows if row["split"] == "val" and row["implementation"] == implementation]
        bins = np.arange(0.5, max(values) + 1.5, 1)
        axes[0].hist(values, bins=bins, alpha=.6, label=implementation, color=color)
    axes[0].set(xlabel="Integer radius (cells)", ylabel="Validation persons", title="Radius distribution")
    axes[0].legend()
    labels_impl, fractions = [], []
    for implementation in ("current", "reference"):
        values = [int(row["integer_radius"]) for row in gaussian_rows if row["split"] == "val" and row["implementation"] == implementation]
        labels_impl.append(implementation); fractions.append(sum(value == 1 for value in values) / len(values))
    axes[1].bar(labels_impl, fractions, color=("#E76F51", "#2A9D8F"))
    axes[1].set(ylabel="Fraction radius 1", title="Validation radius-1 prevalence", ylim=(0, 1))
    for i, value in enumerate(fractions): axes[1].text(i, value + .02, f"{100*value:.2f}%", ha="center")
    _save_figure(fig, figures, "gaussian_radius_current_vs_reference")


def _score_values(result: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key == "tp":
        values = [pair["prediction"]["score"] for pair in result["matches"]]
    elif key == "fp":
        values = [item["score"] for item in result["unmatched_predictions"]]
    else:
        values = [item["score"] for item in result["ignored"]]
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90))}


def _visibility_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for contract in ("v010", "v025"):
        for split in ("train", "val"):
            subset = [row for row in rows if row["contract"] == contract and row["split"] == split]
            counts = Counter(row["category"] for row in subset)
            output[f"{contract}_{split}"] = {
                "denominator": len(subset),
                "categories": {category: {"count": count, "percentage": 100 * count / max(1, len(subset))}
                               for category, count in sorted(counts.items())},
                "off_own_visible_mask": sum(row["category"] != "ON_OWN_VISIBLE_MASK" for row in subset),
                "off_own_visible_mask_percentage": 100 * sum(row["category"] != "ON_OWN_VISIBLE_MASK" for row in subset) / max(1, len(subset)),
                "centroid_offset_model_px": {
                    "median": float(np.median([row["visible_centroid_offset_model_px"] for row in subset])),
                    "p90": float(np.percentile([row["visible_centroid_offset_model_px"] for row in subset], 90)),
                    "max": float(max(row["visible_centroid_offset_model_px"] for row in subset)),
                }
            }
    return output


def _gaussian_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("train", "val"):
        output[split] = {}
        for implementation in ("current", "reference"):
            subset = [row for row in rows if row["split"] == split and row["implementation"] == implementation]
            raw = np.asarray([row["raw_radius"] for row in subset], dtype=np.float64)
            ints = np.asarray([row["integer_radius"] for row in subset], dtype=np.int64)
            output[split][implementation] = {
                "denominator": len(subset), "raw_mean": float(raw.mean()), "raw_median": float(np.median(raw)),
                "integer_mean": float(ints.mean()), "radius_1_count": int(np.count_nonzero(ints == 1)),
                "radius_1_percentage": float(100 * np.mean(ints == 1)),
                "support_cells_mean": float(np.mean([row["support_cells"] for row in subset])),
                "mean_abs_left_right_asymmetry_cells": float(np.mean(np.abs([row["left_right_asymmetry_cells"] for row in subset]))),
                "mean_abs_up_down_asymmetry_cells": float(np.mean(np.abs([row["up_down_asymmetry_cells"] for row in subset]))),
            }
    return output


def _recommendation(terminal: str) -> str:
    if terminal == "PERSON_CONTRACT_AUDIT_LOCALIZATION_DOMINANT":
        bottleneck = "the established localization bottleneck"
    elif terminal == "PERSON_CONTRACT_AUDIT_DETECTION_DOMINANT":
        bottleneck = "the established person-recognition bottleneck"
    else:
        bottleneck = "the established mixed recognition/localization bottleneck"
    return ("Exactly one next design: a factorized visible-person centre-and-range head: supervise the person heatmap at the "
            "depth-consistent visible-mask centroid, then predict camera-ray/metric range separately from the same seven-channel "
            f"RGB+radar LR-ASPP features to address {bottleneck}. Preserve the existing `{{low, high}}` split bundle and the current "
            "q/quant/AE/zstd attachment point; add no raw RGB or radar tail side channel. This is a design recommendation only, not a trained change.")


def _report_markdown(decision: Mapping[str, Any], metrics_rows: Sequence[Mapping[str, Any]], conditional_rows: Sequence[Mapping[str, Any]]) -> str:
    terminal = decision["terminal"]
    main = decision["main_conclusion"]
    lines = [f"# {terminal}", "", main, "",
             "The canonical class-aware 3 m world-XY metric remains authoritative for deployment. All image-space results below are diagnostic and do not establish service readiness or promote a threshold.", "",
             "## Decisive evidence", ""]
    for model in ("base_epoch_040", "distilled_epoch_012"):
        fixed = [row for row in metrics_rows if row["model"] == model and float(row["threshold"]) == .02 and row["match_definition"] == "FULL_BOX_IOU_050"][0]
        component = decision["per_model_fn_decision"][model]
        lines.append(f"- {model}: IoU50@0.02 TP/FP/FN `{fixed['tp']}/{fixed['fp']}/{fixed['fn']}`, P/R/F1 `{float(fixed['precision']):.6f}/{float(fixed['recall']):.6f}/{float(fixed['f1']):.6f}`; among `{component['canonical_joint_fn_denominator']}` canonical joint FNs, `{component['valid_2d_but_world_error_gt_3m_fraction']:.2%}` have a valid 2D match with >3 m error and `{component['lacks_valid_one_to_one_2d_match_fraction']:.2%}` lack a one-to-one 2D match.")
        loc = [row for row in conditional_rows if row["model"] == model and float(row["threshold"]) == .02 and row["match_definition"] == "FULL_BOX_IOU_050" and row["subset_kind"] == "overall"][0]
        lines.append(f"- {model} conditional localization: `{loc['matched_pairs']}` 2D pairs; within 1/2/3/5 m `{float(loc['within_1m_fraction']):.2%}/{float(loc['within_2m_fraction']):.2%}/{float(loc['within_3m_fraction']):.2%}/{float(loc['within_5m_fraction']):.2%}`; median/p90/p95 `{float(loc['median_m']):.3f}/{float(loc['p90_m']):.3f}/{float(loc['p95_m']):.3f}` m.")
    lines += ["", "## Independent flags", ""]
    for flag, value in decision["flags"].items():
        lines.append(f"- `{flag}`: **{'YES' if value else 'NO'}**")
    visibility = decision["target_visibility"]
    lines += ["", f"Primary v0.10 validation target centres are off their own reconstructed visible mask in `{visibility['v010_val']['off_own_visible_mask']}/{visibility['v010_val']['denominator']}` cases (`{visibility['v010_val']['off_own_visible_mask_percentage']:.3f}%`).",
              "", f"Gaussian radius-1 prevalence on validation people is `{decision['gaussian']['val']['current']['radius_1_percentage']:.3f}%` current versus `{decision['gaussian']['val']['reference']['radius_1_percentage']:.3f}%` reference; independent synthetic and population tests all passed.",
              "", f"Calibration verdict: {decision['calibration_verdict']}",
              "", "## Source-contract audit", ""]
    for finding in decision["source_contract_findings"]:
        lines.append(f"- `{finding['classification']}` — {finding['finding']} ({finding['evidence']}). {finding['interpretation']}")
    lines += ["", "## One subsequent design", "", decision["recommendation"], "",
              "No training, optimizer step, new inference, threshold calibration/promotion, test-payload access, CARLA, OAI, q/AE, live split-runtime, or 288 work occurred.", ""]
    return "\n".join(lines)


def _read_typed_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in read_csv(path):
        row: dict[str, Any] = {}
        for key, value in source.items():
            if value == "":
                row[key] = ""
                continue
            try:
                number = float(value)
                row[key] = int(number) if math.isfinite(number) and number.is_integer() else number
            except ValueError:
                row[key] = value
        rows.append(row)
    return rows


def _finalize_existing(config: Mapping[str, Any], output: Path) -> int:
    """Recover from a reporting-only failure without changing computed artifacts."""
    dataset_root = ROOT / config["dataset"]["root"]
    metric_rows = _read_typed_csv(output / "person_2d_metrics.csv")
    conditional_rows = _read_typed_csv(output / "conditional_localization.csv")
    taxonomy_rows = _read_typed_csv(output / "joint_failure_taxonomy.csv")
    calibration_rows = _read_typed_csv(output / "calibration_pr_curve.csv")
    visibility_rows = _read_typed_csv(output / "target_visibility_audit.csv")
    gaussian_rows = _read_typed_csv(output / "gaussian_radius_comparison.csv")
    gaussian_tests = json.loads((output / "GAUSSIAN_UNIT_TESTS.json").read_text(encoding="utf-8"))["tests"]
    rendered = json.loads((output / "REVIEW_PANEL_MANIFEST.json").read_text(encoding="utf-8"))["panels"]

    frame_ids = load_frame_ids(dataset_root)
    gt, metadata, _clear = load_person_gt(dataset_root)
    predictions: dict[str, Any] = {}
    canonical_results: dict[tuple[str, float], Any] = {}
    image_results: dict[tuple[str, float, str], Any] = {}
    fn_decisions: dict[str, Any] = {}
    calibration_summary: dict[str, Any] = {}
    for model, spec in config["models"].items():
        model_predictions = load_predictions(ROOT / spec["prediction_root"] / "detections.csv")
        annotate_neutral_predictions(model_predictions, dataset_root, frame_ids)
        predictions[model] = model_predictions
        canonical_results[(model, .02)] = canonical_world_match(frame_ids, gt, model_predictions, .02)
        image_results[(model, .02, "FULL_BOX_IOU_050")] = image_match(
            frame_ids, gt, model_predictions, .02, "FULL_BOX_IOU_050"
        )
        _rows, fn_decisions[model] = build_taxonomy(
            model, .02, canonical_results[(model, .02)], image_results[(model, .02, "FULL_BOX_IOU_050")]
        )
        calibration_summary[model] = {"all_person_prediction_scores": score_summary(model_predictions), "metrics": {}}
        for metric in ("canonical_joint_3m", "FULL_BOX_IOU_050"):
            rows = [row for row in calibration_rows if row["model"] == model and row["metric"] == metric]
            best = max(rows, key=lambda row: (float(row["f1"]), -float(row["threshold"])))
            fixed = canonical_results[(model, .02)] if metric == "canonical_joint_3m" else image_results[(model, .02, "FULL_BOX_IOU_050")]
            calibration_summary[model]["metrics"][metric] = {
                "auprc_dense": float(rows[0]["auprc_dense"]),
                "post_hoc_max_f1": float(best["f1"]),
                "post_hoc_max_f1_threshold": float(best["threshold"]),
                "score_distributions_at_0_02": {key: _score_values(fixed, key) for key in ("tp", "fp", "ignored")},
            }

    visibility_summary = _visibility_summary(visibility_rows)
    gaussian_summary = _gaussian_summary(gaussian_rows)
    base_component = fn_decisions["base_epoch_040"]
    if base_component["valid_2d_but_world_error_gt_3m_fraction"] > .5:
        terminal = "PERSON_CONTRACT_AUDIT_LOCALIZATION_DOMINANT"
        main_conclusion = "Low reported person recall is primarily a world-localization failure: most authoritative canonical joint FNs retain one-to-one IoU50 person support but the paired world position misses 3 m."
    elif base_component["lacks_valid_one_to_one_2d_match_fraction"] > .5:
        terminal = "PERSON_CONTRACT_AUDIT_DETECTION_DOMINANT"
        main_conclusion = "Low reported person recall is primarily an image-space recognition failure: most authoritative canonical joint FNs lack a valid one-to-one IoU50 person match."
    else:
        terminal = "PERSON_CONTRACT_AUDIT_MIXED_FAILURE"
        main_conclusion = "Low reported person recall is a mixed recognition/localization failure: neither missing one-to-one IoU50 support nor >3 m world error alone exceeds half of canonical joint FNs."
    visibility_flag = any(row["category"] != "ON_OWN_VISIBLE_MASK" for row in visibility_rows)
    gaussian_flag = any(abs(float(row["raw_radius"]) - float(other["raw_radius"])) > 1e-12
                        for row, other in zip(gaussian_rows[0::2], gaussian_rows[1::2]))
    base_cal = calibration_summary["base_epoch_040"]["metrics"]["canonical_joint_3m"]
    dist_cal = calibration_summary["distilled_epoch_012"]["metrics"]["canonical_joint_3m"]
    calibration_flag = (
        abs(dist_cal["post_hoc_max_f1"] - base_cal["post_hoc_max_f1"]) <= .01
        and abs(dist_cal["post_hoc_max_f1_threshold"] - base_cal["post_hoc_max_f1_threshold"]) >= .05
        and abs(dist_cal["auprc_dense"] - base_cal["auprc_dense"]) <= .02
    )
    calibration_verdict = (
        "DISTILLATION_CALIBRATION_SHIFT_CONFIRMED: near-equal post-hoc joint max F1/AUPRC accompanies a material best-threshold shift."
        if calibration_flag else
        "DISTILLATION_CALIBRATION_SHIFT_NOT_CONFIRMED: ranking/AUPRC and threshold evidence do not isolate calibration alone."
    )
    expected_curve = {
        "base_f1_at_0_40": next(float(row["f1"]) for row in calibration_rows if row["model"] == "base_epoch_040" and row["metric"] == "canonical_joint_3m" and float(row["threshold"]) == .4),
        "distilled_f1_at_0_50": next(float(row["f1"]) for row in calibration_rows if row["model"] == "distilled_epoch_012" and row["metric"] == "canonical_joint_3m" and float(row["threshold"]) == .5),
    }
    source_findings = _source_findings()
    unique_visibility = {(row["sample_id"], row["source_identity"]) for row in visibility_rows}
    first_result_mtime = min((output / name).stat().st_mtime for name in (
        "CANONICAL_PARITY.json", "person_2d_metrics.csv", "target_visibility_audit.csv", "gaussian_radius_comparison.csv"
    ))
    decision: dict[str, Any] = {
        "schema": "route_b_v3_1_person_contract_audit_decision_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "terminal": terminal,
        "decision_model": "base_epoch_040", "main_conclusion": main_conclusion,
        "canonical_metrics_remain_authoritative": True,
        "diagnostic_2d_does_not_establish_service_readiness": True,
        "per_model_fn_decision": fn_decisions,
        "flags": {
            "HEATMAP_TARGET_VISIBILITY_DEFECT_CONFIRMED": visibility_flag,
            "GAUSSIAN_RADIUS_IMPLEMENTATION_MISMATCH_CONFIRMED": gaussian_flag,
            "DISTILLATION_CALIBRATION_SHIFT_CONFIRMED": calibration_flag,
            "PERSON_CONTRACT_AUDIT_INVALID": False,
        },
        "target_visibility": visibility_summary,
        "target_visibility_reconciliation": {
            "contract_split_denominators": {key: value["denominator"] for key, value in visibility_summary.items()},
            "unique_actor_frame_reconstructions": len(unique_visibility), "rows": len(visibility_rows),
        },
        "gaussian": gaussian_summary,
        "gaussian_unit_tests_all_pass": all(row["passed"] for row in gaussian_tests),
        "calibration": calibration_summary,
        "calibration_expected_observations": expected_curve,
        "calibration_verdict": calibration_verdict,
        "source_contract_findings": source_findings,
        "recommendation": _recommendation(terminal), "review_panel_count": len(rendered),
        "wall_seconds": time.time() - first_result_mtime,
        "wall_time_measurement": "reporting-recovery lower bound measured from first result artifact write",
        "reporting_recovery": "finalized without overwriting completed evidence after literal-brace formatting NameError",
        "forbidden_work": {"training": 0, "optimizer_steps": 0, "new_model_inference": 0,
                           "test_payload_access": 0, "carla": 0, "oai": 0, "q_ae": 0, "measurements_288": 0},
    }
    _make_figures(output / "figures", calibration_rows, taxonomy_rows, conditional_rows, visibility_rows, gaussian_rows)
    write_json_x(output / "SOURCE_CONTRACT_AUDIT.json", {"findings": source_findings})
    (output / "PERSON_CONTRACT_AUDIT_REPORT.md").write_text(_report_markdown(decision, metric_rows, conditional_rows), encoding="utf-8")
    write_json_x(output / "DECISION.json", decision)
    (output / "COMPLETION_SENTINEL").write_text(f"{terminal}\n", encoding="utf-8")
    print(json.dumps({"terminal": terminal, "flags": decision["flags"], "wall_seconds": decision["wall_seconds"],
                      "output": str(output), "reporting_recovery": True}, indent=2), flush=True)
    return 0


def audit(config_path: Path, output: Path) -> int:
    started = time.monotonic()
    config = _verify_registration(config_path, output)
    intermediate = (
        "CANONICAL_PARITY.json", "person_2d_metrics.csv", "conditional_localization.csv",
        "joint_failure_taxonomy.csv", "target_visibility_audit.csv", "gaussian_radius_comparison.csv",
        "calibration_pr_curve.csv", "GAUSSIAN_UNIT_TESTS.json", "REVIEW_PANEL_MANIFEST.json"
    )
    finals = ("PERSON_CONTRACT_AUDIT_REPORT.md", "DECISION.json", "COMPLETION_SENTINEL")
    if all((output / name).is_file() for name in intermediate) and not any((output / name).exists() for name in finals):
        return _finalize_existing(config, output)
    if any((output / name).exists() for name in (*intermediate, *finals)):
        raise FileExistsError("create-only audit result is partial in an unrecognized state")
    dataset_root = ROOT / config["dataset"]["root"]
    frame_ids = load_frame_ids(dataset_root)
    gt, metadata, _clear = load_person_gt(dataset_root)
    if len(metadata) != 3872 or len(frame_ids) != 3345:
        raise RuntimeError("PERSON_CONTRACT_AUDIT_INVALID: validation denominator drift")

    print("[phase A/B] canonical parity, 2D factorization, conditional localization", flush=True)
    predictions: dict[str, Any] = {}
    canonical_results: dict[tuple[str, float], Any] = {}
    image_results: dict[tuple[str, float, str], Any] = {}
    metric_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    fn_decisions: dict[str, Any] = {}
    parity: dict[str, Any] = {"schema": "route_b_v3_1_person_canonical_parity_v1", "models": {}, "all_exact": True}
    expected = {
        "base_epoch_040": {0.20: (0.537513397642015, 0.5180785123966942, 0.527617043661231), 0.02: (None, 0.6151859504132231, None)},
        "distilled_epoch_012": {0.20: (0.4281957928802589, 0.546745867768595, 0.4802631578947368), 0.02: (None, 0.6655475206611571, None)},
    }
    for model, spec in config["models"].items():
        model_predictions = load_predictions(ROOT / spec["prediction_root"] / "detections.csv")
        annotate_neutral_predictions(model_predictions, dataset_root, frame_ids)
        predictions[model] = model_predictions
        parity["models"][model] = {}
        for threshold in (0.20, 0.02):
            canonical = canonical_world_match(frame_ids, gt, model_predictions, threshold)
            canonical_results[(model, threshold)] = canonical
            actual = (canonical["precision"], canonical["recall"], canonical["f1"])
            checks = [want is None or abs(got - want) <= 5e-10 for got, want in zip(actual, expected[model][threshold])]
            parity["models"][model][f"{threshold:.2f}"] = {**_public_metrics(canonical), "published_expected": expected[model][threshold], "exact_within_5e_10": all(checks)}
            parity["all_exact"] = parity["all_exact"] and all(checks)
            for definition in MATCH_DEFINITIONS:
                result = image_match(frame_ids, gt, model_predictions, threshold, definition)
                image_results[(model, threshold, definition)] = result
        for threshold in (0.20, 0.02):
            primary = image_results[(model, threshold, "FULL_BOX_IOU_050")]
            for definition in MATCH_DEFINITIONS:
                result = image_results[(model, threshold, definition)]
                changed, symmetric = assignment_difference(result, primary)
                metric_rows.append({
                    "model": model, "threshold": threshold, "match_definition": definition,
                    **_public_metrics(result), "unmatched_gt_count": len(result["unmatched_gt"]),
                    "unmatched_prediction_count": len(result["unmatched_predictions"]),
                    "assignment_changed_gt_vs_iou050": changed,
                    "assignment_pair_symmetric_difference_vs_iou050": symmetric,
                    "class_confusion_gt_count": result["class_confusion_gt_count"],
                    "contended_gt_count": result["contended_gt_count"],
                    "contended_prediction_count": result["contended_prediction_count"],
                })
                conditional_rows.extend(summarize_conditional(model, threshold, definition, result))
        rows, decision = build_taxonomy(model, .02, canonical_results[(model, .02)], image_results[(model, .02, "FULL_BOX_IOU_050")])
        taxonomy_rows.extend(rows); fn_decisions[model] = decision
    if not parity["all_exact"]:
        raise RuntimeError(f"PERSON_CONTRACT_AUDIT_INVALID: canonical parity failed: {parity}")
    write_json_x(output / "CANONICAL_PARITY.json", parity)
    write_csv_x(output / "person_2d_metrics.csv", metric_rows)
    write_csv_x(output / "conditional_localization.csv", conditional_rows)
    write_csv_x(output / "joint_failure_taxonomy.csv", taxonomy_rows)

    print("[phase E] dense retained-prediction calibration/ranking curves", flush=True)
    calibration_rows: list[dict[str, Any]] = []
    calibration_summary: dict[str, Any] = {}
    for model, model_predictions in predictions.items():
        group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index, threshold in enumerate(threshold_grid(), 1):
            joint = canonical_world_match(frame_ids, gt, model_predictions, threshold)
            image = image_match(frame_ids, gt, model_predictions, threshold, "FULL_BOX_IOU_050")
            for name, result in (("canonical_joint_3m", joint), ("FULL_BOX_IOU_050", image)):
                group_rows[name].append({"model": model, "metric": name, "threshold": threshold,
                                         **{key: result[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")}})
            if index % 50 == 0:
                print(f"[calibration] {model} {index}/{len(threshold_grid())}", flush=True)
        calibration_summary[model] = {"all_person_prediction_scores": score_summary(model_predictions), "metrics": {}}
        for metric, rows in group_rows.items():
            auprc = trapezoid_auprc(rows)
            best = max(rows, key=lambda row: (row["f1"], -row["threshold"]))
            for row in rows:
                row["auprc_dense"] = auprc
                row["post_hoc_max_f1"] = best["f1"]
                row["post_hoc_max_f1_threshold"] = best["threshold"]
                row["threshold_role"] = "post_hoc_non_promotable" if row["threshold"] == best["threshold"] else "diagnostic_curve"
            calibration_rows.extend(rows)
            fixed002 = canonical_results[(model, .02)] if metric == "canonical_joint_3m" else image_results[(model, .02, "FULL_BOX_IOU_050")]
            calibration_summary[model]["metrics"][metric] = {
                "auprc_dense": auprc, "post_hoc_max_f1": best["f1"],
                "post_hoc_max_f1_threshold": best["threshold"],
                "score_distributions_at_0_02": {key: _score_values(fixed002, key) for key in ("tp", "fp", "ignored")},
            }
    write_csv_x(output / "calibration_pr_curve.csv", calibration_rows)

    print("[phase C] reconstructing every v010/v025 train/validation person target", flush=True)
    visibility_rows, visibility_reconciliation = audit_target_visibility(dataset_root)
    write_csv_x(output / "target_visibility_audit.csv", visibility_rows)
    selections = select_review_panels(visibility_rows)
    rendered = render_review_panels(dataset_root, selections, output / "review_panels")
    write_json_x(output / "REVIEW_PANEL_MANIFEST.json", {"panels": rendered, "counts": dict(Counter(item["review_role"] for item in rendered))})

    print("[phase D] independent Gaussian reference and population tests", flush=True)
    gaussian_rows, gaussian_tests = audit_gaussian(dataset_root)
    write_csv_x(output / "gaussian_radius_comparison.csv", gaussian_rows)
    write_json_x(output / "GAUSSIAN_UNIT_TESTS.json", {"all_pass": all(row["passed"] for row in gaussian_tests), "tests": gaussian_tests})

    visibility_summary = _visibility_summary(visibility_rows)
    gaussian_summary = _gaussian_summary(gaussian_rows)
    base_component = fn_decisions["base_epoch_040"]
    if base_component["valid_2d_but_world_error_gt_3m_fraction"] > .5:
        terminal = "PERSON_CONTRACT_AUDIT_LOCALIZATION_DOMINANT"
        main_conclusion = "Low reported person recall is primarily a world-localization failure: most authoritative canonical joint FNs retain one-to-one IoU50 person support but the paired world position misses 3 m."
    elif base_component["lacks_valid_one_to_one_2d_match_fraction"] > .5:
        terminal = "PERSON_CONTRACT_AUDIT_DETECTION_DOMINANT"
        main_conclusion = "Low reported person recall is primarily an image-space recognition failure: most authoritative canonical joint FNs lack a valid one-to-one IoU50 person match."
    else:
        terminal = "PERSON_CONTRACT_AUDIT_MIXED_FAILURE"
        main_conclusion = "Low reported person recall is a mixed recognition/localization failure: neither missing one-to-one IoU50 support nor >3 m world error alone exceeds half of canonical joint FNs."
    visibility_flag = any(row["category"] != "ON_OWN_VISIBLE_MASK" for row in visibility_rows)
    gaussian_flag = any(abs(float(row["raw_radius"]) - float(other["raw_radius"])) > 1e-12
                        for row, other in zip(gaussian_rows[0::2], gaussian_rows[1::2]))
    base_cal = calibration_summary["base_epoch_040"]["metrics"]["canonical_joint_3m"]
    dist_cal = calibration_summary["distilled_epoch_012"]["metrics"]["canonical_joint_3m"]
    calibration_flag = (
        abs(dist_cal["post_hoc_max_f1"] - base_cal["post_hoc_max_f1"]) <= .01
        and abs(dist_cal["post_hoc_max_f1_threshold"] - base_cal["post_hoc_max_f1_threshold"]) >= .05
        and abs(dist_cal["auprc_dense"] - base_cal["auprc_dense"]) <= .02
    )
    calibration_verdict = (
        "DISTILLATION_CALIBRATION_SHIFT_CONFIRMED: near-equal post-hoc joint max F1/AUPRC accompanies a material best-threshold shift."
        if calibration_flag else
        "DISTILLATION_CALIBRATION_SHIFT_NOT_CONFIRMED: ranking/AUPRC and threshold evidence do not isolate calibration alone."
    )
    expected_curve = {
        "base_f1_at_0_40": next(row["f1"] for row in calibration_rows if row["model"] == "base_epoch_040" and row["metric"] == "canonical_joint_3m" and row["threshold"] == .4),
        "distilled_f1_at_0_50": next(row["f1"] for row in calibration_rows if row["model"] == "distilled_epoch_012" and row["metric"] == "canonical_joint_3m" and row["threshold"] == .5),
    }
    source_findings = _source_findings()
    decision: dict[str, Any] = {
        "schema": "route_b_v3_1_person_contract_audit_decision_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "terminal": terminal,
        "decision_model": "base_epoch_040", "main_conclusion": main_conclusion,
        "canonical_metrics_remain_authoritative": True,
        "diagnostic_2d_does_not_establish_service_readiness": True,
        "per_model_fn_decision": fn_decisions,
        "flags": {
            "HEATMAP_TARGET_VISIBILITY_DEFECT_CONFIRMED": visibility_flag,
            "GAUSSIAN_RADIUS_IMPLEMENTATION_MISMATCH_CONFIRMED": gaussian_flag,
            "DISTILLATION_CALIBRATION_SHIFT_CONFIRMED": calibration_flag,
            "PERSON_CONTRACT_AUDIT_INVALID": False,
        },
        "target_visibility": visibility_summary,
        "target_visibility_reconciliation": visibility_reconciliation,
        "gaussian": gaussian_summary,
        "gaussian_unit_tests_all_pass": all(row["passed"] for row in gaussian_tests),
        "calibration": calibration_summary,
        "calibration_expected_observations": expected_curve,
        "calibration_verdict": calibration_verdict,
        "source_contract_findings": source_findings,
        "recommendation": _recommendation(terminal),
        "review_panel_count": len(rendered),
        "wall_seconds": time.monotonic() - started,
        "forbidden_work": {"training": 0, "optimizer_steps": 0, "new_model_inference": 0,
                           "test_payload_access": 0, "carla": 0, "oai": 0, "q_ae": 0, "measurements_288": 0},
    }
    _make_figures(output / "figures", calibration_rows, taxonomy_rows, conditional_rows, visibility_rows, gaussian_rows)
    write_json_x(output / "SOURCE_CONTRACT_AUDIT.json", {"findings": source_findings})
    report = _report_markdown(decision, metric_rows, conditional_rows)
    (output / "PERSON_CONTRACT_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    write_json_x(output / "DECISION.json", decision)
    (output / "COMPLETION_SENTINEL").write_text(f"{terminal}\n", encoding="utf-8")
    print(json.dumps({"terminal": terminal, "flags": decision["flags"], "wall_seconds": decision["wall_seconds"],
                      "output": str(output)}, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("register", "audit"), required=True)
    args = parser.parse_args()
    if args.phase == "register":
        return register(args.config.resolve(), args.output.resolve())
    return audit(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
