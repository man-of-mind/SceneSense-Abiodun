from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_collection.route_b_publication_actor_volume_observability_model_comparison_v1 import (
    run_comparison as avo,
)

from .provenance import load_candidate_contract, sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "experiments/splitfusion_fcos_person_p025_calibration_v1"
TRAIN_RESULT = OUTPUT_DIR / "train_holdout_qualification.json"
AVO_TABLE = (
    REPO_ROOT
    / "experiments/actor_volume_observability_model_comparison_v1"
    / "20260901_repaired_tolerance_cpu_once/actor_volume_observability_table.csv"
)
AVO_HASHES = AVO_TABLE.parent / "ARTIFACT_HASHES.json"
PREDICTION_ROOT = REPO_ROOT / "experiments/splitfusion_fcos_service_candidate_v1/predictions"
DETECTIONS = PREDICTION_ROOT / "detections.csv"
INFERENCE_MANIFEST = PREDICTION_ROOT / "inference_manifest.json"
CANONICAL_EXPERIMENT = (
    REPO_ROOT / "experiments/route_b_v3_1_expanded_train_camera_plane_v1/20260828_094151"
)
NATIVE_EVALUATOR = (
    REPO_ROOT
    / "pole_lraspp_multimodal_fusion/object_head_pilot_v1"
    / "route_b_v3_1_native_grid_v1/evaluate_v1.py"
)
DETECTION_THRESHOLD = 0.25
AVO_THRESHOLD = 0.65
TERMINAL = "PERSON_P025_TRAIN_HOLDOUT_QUALIFIED_VALIDATION_CONFIRMED"


class ValidationError(RuntimeError):
    """Fail-closed frozen-validation provenance or scoring error."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_native_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "person_p025_frozen_native_evaluator", NATIVE_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise ImportError(NATIVE_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_frozen_inputs() -> dict[str, str]:
    manifest = read_json(INFERENCE_MANIFEST)
    detection_hash = sha256(DETECTIONS)
    registered_hashes = read_json(AVO_HASHES)
    table_hash = sha256(AVO_TABLE)
    if not (
        detection_hash == "a682a1fc5eabb2e59e07449a8c6b5fc604077b40ef094b57dc30c5a18d7ec260"
        and manifest.get("detections_sha256") == detection_hash
        and manifest.get("schema") == "splitfusion_fcos_service_candidate_inference_v1"
        and manifest.get("inference_pass_count") == 1
        and manifest.get("validation_frames") == 3345
        and manifest.get("derived_threshold") == 0.20
        and manifest.get("person_configuration")
        == {
            "grid_index": 27,
            "group_box_iou_threshold": 0.20,
            "semantic_support_threshold": 0.10,
        }
        and table_hash
        == registered_hashes.get("outputs", {}).get("actor_volume_observability_table.csv")
        == "abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386"
    ):
        raise ValidationError("frozen validation input provenance drift")
    return {
        "detections.csv": detection_hash,
        "inference_manifest.json": sha256(INFERENCE_MANIFEST),
        "actor_volume_observability_table.csv": table_hash,
        "actor_volume_ARTIFACT_HASHES.json": sha256(AVO_HASHES),
    }


def score_avo() -> tuple[dict[str, Any], dict[str, str]]:
    raw = avo.load_raw_sources()
    table = avo.read_csv_pandas(AVO_TABLE, dtype={"gt_actor_id": str})
    table_keys = {(str(row["sample_id"]), str(row["gt_actor_id"])) for row in table}
    raw_keys = {
        (str(row["sample_id"]), str(row["gt_actor_id"])) for row in raw["qualified"]
    }
    if len(table_keys) != len(table) or table_keys != raw_keys:
        raise ValidationError("frozen AVO table does not exactly cover qualified validation GT")
    predictions, total_prediction_rows = avo.load_person_predictions(DETECTIONS)
    person_rows = [row for values in predictions.values() for row in values]
    identities = [
        (sample_id, int(row["prediction_index"]))
        for sample_id, values in predictions.items()
        for row in values
    ]
    if not (
        total_prediction_rows == 116744
        and len(person_rows) == 3577
        and len(identities) == len(set(identities))
        and all(float(row["score"]) >= 0.20 for row in person_rows)
    ):
        raise ValidationError("frozen p020 person prediction-set contract drift")
    qualified_gt = avo.gt_from_table(table)
    structural = avo.structural_gt(raw)
    episode_by_sample = {
        sample_id: str(meta["experiment_id"])
        for sample_id, meta in raw["manifest_by_sample"].items()
    }
    result = avo.score_person_view(
        frame_ids=raw["frame_ids"],
        predictions=predictions,
        qualified_gt=qualified_gt,
        structural_ignored_gt=structural,
        episode_by_sample=episode_by_sample,
        avo_threshold=AVO_THRESHOLD,
        detection_threshold=DETECTION_THRESHOLD,
    )
    result["frozen_prediction_rows_all_classes"] = total_prediction_rows
    result["retained_p020_person_rows"] = len(person_rows)
    result["retained_p025_person_rows"] = sum(
        float(row["score"]) >= DETECTION_THRESHOLD
        for values in predictions.values()
        for row in values
    )
    result["p025_exact_score_filtered_subset_of_p020"] = True
    return result, raw["input_hashes"]


def person_only(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    return {
        sample_id: [dict(row) for row in rows if str(row["class_name"]) == "person"]
        for sample_id, rows in grouped.items()
    }


def score_canonical() -> tuple[dict[str, Any], dict[str, str]]:
    native = load_native_evaluator()
    manifest_path = CANONICAL_EXPERIMENT / "dataset/manifest.csv"
    manifest = native.read_csv(manifest_path)
    validation = [row for row in manifest if row["split"] == "val"]
    frame_ids = [str(row["sample_id"]) for row in validation]
    if len(frame_ids) != 3345 or len(frame_ids) != len(set(frame_ids)):
        raise ValidationError("canonical validation frame identity drift")
    if set(str(row["experiment_id"]) for row in validation) != set(avo.EPISODES):
        raise ValidationError("canonical validation episode identity drift")
    predictions, missing = native.load_predictions(DETECTIONS)
    if missing:
        raise ValidationError(f"invalid frozen prediction fields: {missing[:3]}")
    gt, _states = native.load_gt(CANONICAL_EXPERIMENT, "v010")
    person_predictions = person_only(predictions)
    person_gt = person_only(gt)
    ignore_cache: dict[str, Any] = {}

    def score(ids: Sequence[str]) -> dict[str, Any]:
        scored = native.score_arm(
            experiment=CANONICAL_EXPERIMENT,
            contract="v010",
            frame_ids=ids,
            predictions=person_predictions,
            gt=person_gt,
            threshold=DETECTION_THRESHOLD,
            ignore_cache=ignore_cache,
        )
        return dict(scored["classes"]["person"])

    by_episode = {
        episode: score(
            [str(row["sample_id"]) for row in validation if row["experiment_id"] == episode]
        )
        for episode in avo.EPISODES
    }
    result = {
        "contract": "v010",
        "detection_score_threshold": DETECTION_THRESHOLD,
        "overall": score(frame_ids),
        "episodes": by_episode,
        "person_only_threshold_change": True,
        "vehicle_behavior_evaluated_or_changed": False,
    }
    hashes = {
        str(manifest_path.relative_to(REPO_ROOT)): sha256(manifest_path),
        str(
            (CANONICAL_EXPERIMENT / "contracts/v010/val/object_boxes.csv").relative_to(
                REPO_ROOT
            )
        ): sha256(CANONICAL_EXPERIMENT / "contracts/v010/val/object_boxes.csv"),
        str(NATIVE_EVALUATOR.relative_to(REPO_ROOT)): sha256(NATIVE_EVALUATOR),
    }
    return result, hashes


def report(
    train: Mapping[str, Any], validation_avo: Mapping[str, Any], canonical: Mapping[str, Any],
    runtime_seconds: float,
) -> str:
    lines = [
        "# Frozen person p025 service-candidate confirmation",
        "",
        "The fixed post-consolidation person threshold 0.25 qualified on the two registered "
        "train-only holdout episodes and was then confirmed against the existing frozen "
        "validation predictions. No inference, training, cache rebuild, CUDA, CARLA, or test "
        "access occurred.",
        "",
        "## Train-only holdout at AVO >= 0.65",
        "",
        "| view | precision | recall | F1 | XY MAE m |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("p020", "p025"):
        metric = train["train_holdout"][name]["overall"]
        lines.append(
            f"| {name} | {metric['precision']:.6f} | {metric['recall']:.6f} | "
            f"{metric['f1']:.6f} | {metric['xy_mae_m']:.6f} |"
        )
        for episode, episode_metric in train["train_holdout"][name]["episodes"].items():
            lines.append(
                f"| {name} / {episode} | {episode_metric['precision']:.6f} | "
                f"{episode_metric['recall']:.6f} | {episode_metric['f1']:.6f} | "
                f"{episode_metric['xy_mae_m']:.6f} |"
            )
    lines.extend((
        "",
        "## Frozen validation at person threshold 0.25",
        "",
        "| view | episode | precision | recall | F1 | XY MAE m |",
        "|---|---|---:|---:|---:|---:|",
    ))
    for view_name, view in (("AVO>=0.65", validation_avo), ("canonical v0.10", canonical)):
        overall = view["overall"]
        lines.append(
            f"| {view_name} | aggregate | {overall['precision']:.6f} | "
            f"{overall['recall']:.6f} | {overall['f1']:.6f} | {overall['xy_mae_m']:.6f} |"
        )
        for episode, metric in view["episodes"].items():
            lines.append(
                f"| {view_name} | {episode} | {metric['precision']:.6f} | "
                f"{metric['recall']:.6f} | {metric['f1']:.6f} | "
                f"{metric['xy_mae_m']:.6f} |"
            )
    lines.extend((
        "",
        "## Exact frozen input hashes",
        "",
        f"- Cache manifest: `{train['input_hashes']['cache']['cache_manifest_sha256']}`",
        f"- Cache shard hash map: `{train['input_hashes']['cache']['shard_hash_map_sha256']}`",
        f"- Training support records: `{train['input_hashes']['reference']['training_support_records_sha256']}`",
        f"- Training reference JSON: `{train['input_hashes']['reference']['training_reference_json_sha256']}`",
        "",
        "Validation threshold behavior was previously explored. The untouched test set remains "
        "necessary for independent publication confirmation. The supervisor-approved p020 "
        "service is unchanged; p025 is a proposed deployment candidate awaiting final acceptance.",
        "",
        f"Runtime: {train['runtime_seconds']:.3f} seconds train qualification + "
        f"{runtime_seconds:.3f} seconds validation confirmation = "
        f"{train['runtime_seconds'] + runtime_seconds:.3f} seconds total.",
        "",
        TERMINAL,
        "",
    ))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValidationError('refusing to run without CUDA_VISIBLE_DEVICES=""')
    output = args.output.resolve(strict=True)
    if output != OUTPUT_DIR.resolve():
        raise ValidationError("output must be the registered calibration directory")
    if (output / "validation_confirmation.json").exists():
        raise FileExistsError("validation confirmation is create-only")
    started = time.perf_counter()

    contract = load_candidate_contract()
    train = read_json(TRAIN_RESULT)
    frozen_hashes = validate_frozen_inputs()
    validation_avo, raw_hashes = score_avo()
    canonical, canonical_hashes = score_canonical()
    elapsed = time.perf_counter() - started
    result = {
        "schema": "splitfusion_fcos_person_p025_validation_confirmation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": TERMINAL,
        "train_qualification_sha256": sha256(TRAIN_RESULT),
        "train_qualified": True,
        "frozen_candidate": contract,
        "validation": {"avo_gte_0_65": validation_avo, "canonical_v010": canonical},
        "input_hashes": {
            "frozen_outputs": frozen_hashes,
            "raw_validation_metadata": raw_hashes,
            "canonical": canonical_hashes,
        },
        "runtime_seconds": elapsed,
        "total_runtime_seconds_including_train_qualification": train["runtime_seconds"] + elapsed,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_inference_run": False,
        "training_run": False,
        "cache_rebuilt": False,
        "carla_started": False,
        "test_accessed": False,
        "validation_threshold_behavior_previously_explored": True,
        "untouched_test_required_for_independent_publication_confirmation": True,
        "approved_p020_service_automatically_replaced": False,
        "p025_status": "proposed_deployment_candidate_awaiting_final_acceptance",
    }
    write_json_x(output / "validation_confirmation.json", result)
    (output / "FINAL_REPORT.md").write_text(
        report(train, validation_avo, canonical, elapsed), encoding="utf-8"
    )
    (output / TERMINAL).write_text(TERMINAL + "\n", encoding="utf-8")
    print(json.dumps({
        "terminal": TERMINAL,
        "runtime_seconds": elapsed,
        "validation": result["validation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
