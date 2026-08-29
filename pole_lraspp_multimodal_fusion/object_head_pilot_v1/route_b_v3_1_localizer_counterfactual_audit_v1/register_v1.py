#!/usr/bin/env python3
"""Freeze the revised audit contract before any counterfactual is evaluated."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from common_v1 import git_head, read_csv, sha256, utc_now, write_json_x, write_text_x  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    experiment = args.experiment.resolve()
    if experiment.exists():
        raise FileExistsError(f"create-only experiment already exists: {experiment}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    head = git_head(ROOT)
    if head != config["required_head"]:
        raise RuntimeError(f"HEAD drift: {head} != {config['required_head']}")
    paths = {
        key: (ROOT / config[key]).resolve(strict=True)
        for key in (
            "dataset_root", "base_checkpoint", "candidate_checkpoint", "base_predictions",
            "candidate_predictions", "published_evaluation", "published_baseline",
        )
    }
    hashes = {
        "base_checkpoint": sha256(paths["base_checkpoint"]),
        "candidate_checkpoint": sha256(paths["candidate_checkpoint"]),
        "base_detections": sha256(paths["base_predictions"] / "detections.csv"),
        "candidate_detections": sha256(paths["candidate_predictions"] / "detections.csv"),
    }
    expected_hashes = {
        "base_checkpoint": config["base_checkpoint_sha256"],
        "candidate_checkpoint": config["candidate_checkpoint_sha256"],
        "base_detections": config["base_detections_sha256"],
        "candidate_detections": config["candidate_detections_sha256"],
    }
    if hashes != expected_hashes:
        raise RuntimeError(f"input hash drift: {hashes} != {expected_hashes}")
    manifest = read_csv(paths["dataset_root"] / "dataset/manifest.csv")
    split_counts = {split: sum(row["split"] == split for row in manifest)
                    for split in ("train", "val", "test")}
    person_gt = [row for row in read_csv(
        paths["dataset_root"] / "contracts/v010/val/object_boxes.csv"
    ) if row["label"] == "person"]
    if (split_counts != {"train": 16827, "val": 3345, "test": 0}
            or len(person_gt) != config["expected"]["primary_person_gt"]):
        raise RuntimeError(f"dataset population drift: {split_counts}, person={len(person_gt)}")
    experiment.mkdir(parents=True)
    resolved = dict(config)
    resolved["registration"] = {
        "created_utc": utc_now(), "repository_head": head,
        "config_source": str(config_path), "config_source_sha256": sha256(config_path),
        "experiment": str(experiment), "input_hashes": hashes,
        "split_counts": split_counts, "primary_person_gt": len(person_gt),
        "optimizer_steps": 0, "model_training_runs": 0,
        "candidate_inference_runs": 0, "base_inference_runs": 0,
    }
    write_json_x(experiment / "RESOLVED_CONFIG.json", resolved)
    provenance = {
        "schema": "route_b_v3_1_localizer_counterfactual_input_provenance_v1",
        "created_utc": utc_now(), "repository_head": head,
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": hashes, "split_counts": split_counts,
        "primary_person_gt": len(person_gt),
        "retained_candidate_local_xyz_supports_exact_ray_reconstruction": True,
        "retained_dense_epoch40_maps_found": False,
        "allowed_new_candidate_traversals": 0,
        "allowed_new_base_traversals": 1,
        "forbidden_scope_access_counts": {
            "locked_test": 0, "carla": 0, "oai": 0, "q_ae": 0,
            "live_runtime": 0, "measurements_288": 0, "training": 0,
        },
    }
    write_json_x(experiment / "INPUT_PROVENANCE.json", provenance)
    registration = {
        "schema": "route_b_v3_1_localizer_counterfactual_registered_plan_v1",
        "created_utc": utc_now(), "config_sha256": sha256(experiment / "RESOLVED_CONFIG.json"),
        "repository_head": head, "counterfactual_contract": config["counterfactual_contract"],
        "inference_allowance": config["inference_allowance"],
        "strata": config["strata"], "attribution": config["attribution"],
        "composition_success": config["composition_success"], "decision": config["decision"],
        "required_outputs": config["required_outputs"], "optimizer_steps_before_registration": 0,
        "counterfactual_results_examined_before_registration": False,
    }
    write_json_x(experiment / "REGISTERED_AUDIT_PLAN.json", registration)
    markdown = "# Registered LR-ASPP localizer counterfactual audit\n\n"
    markdown += f"Registered before counterfactual evaluation at `{registration['created_utc']}`.\n\n"
    markdown += "- Candidate traversal allowance: 0.\n- Epoch-40 traversal allowance: at most 1.\n"
    markdown += "- Sampler: `floor(predicted_full_box_center / 4)` hard native-cell read; no interpolation or clamping.\n"
    markdown += "- Pairwise oracles freeze the candidate IoU50 association before substitution.\n"
    markdown += "- Primary composition and secondary depth/ray attribution are separate decisions.\n"
    markdown += "- A GT-centre-cell inherited-field arm is diagnostic only and distinguishes field quality from sampling error.\n"
    markdown += "- No training, threshold/NMS sweep, v0.25 inference, test, CARLA, OAI, q/AE, live runtime, or measurements.\n"
    write_text_x(experiment / "REGISTERED_AUDIT_PLAN.md", markdown)
    write_text_x(experiment / "REGISTRATION_COMPLETE", "REGISTERED\n")
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
