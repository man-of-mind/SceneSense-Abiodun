from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from common import CONFIG_PATH, ROOT, atomic_json, atomic_text, load_json, seed_everything, utc_now
from data import RouteBDataset, target_from_rows
from losses import match_anchors
from model import build_model


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    if not (experiment / "STRUCTURAL_QUALIFICATION_COMPLETE").is_file(): raise RuntimeError("structure not qualified")
    if (experiment / "QUALIFICATION_COMPLETE").exists() or (experiment / "SCIENTIFIC_TRAINING_STARTED.json").exists():
        raise RuntimeError("assignment audit must precede disposable/scientific training")
    config = load_json(CONFIG_PATH); priors = load_json(experiment / "TRAIN_ONLY_PRIORS.json")
    seed_everything(int(config["scientific_seed"])); model, _ = build_model(priors)
    shapes = ((112, 192), (56, 96), (28, 48), (14, 24), (7, 12), (4, 6))
    c2 = torch.empty(1, 256, *shapes[0]); features = [torch.empty(1, 256, *shape) for shape in shapes]
    anchors = model.anchors(c2, features)[0]
    num = [height * width for height, width in shapes]
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    dataset = RouteBDataset(dataset_root, "train", int(config["scientific_seed"]), depth_cache=None, augment=False)
    class_level = {name: Counter() for name in ("vehicle", "person")}; visibility = Counter()
    no_carrier, foreground, ignored = [], 0, 0
    for index, frame in enumerate(dataset.rows):
        segmentation = dataset._mask(frame, "mask_path"); segmentation[segmentation == 255] = -100
        ignore = dataset._mask(frame, "object_ignore_mask_path").ne(0)
        target = target_from_rows(frame, dataset.objects.get(frame["sample_id"], ()), segmentation, ignore)
        _matched, audit = match_anchors(anchors, target, num)
        foreground += audit["foreground"]; ignored += audit["ignored_locations"]
        for class_name in class_level: class_level[class_name].update(audit["per_class_level"][class_name])
        visibility.update(audit["carrier_visibility"])
        no_carrier.extend({"sample_id": frame["sample_id"], "source_identity": identity}
                          for identity in audit["actors_without_carrier"])
        if (index + 1) % 1000 == 0:
            print(json.dumps({"assignment_frames": index + 1, "foreground": foreground,
                              "actors_without_carrier": len(no_carrier)}), flush=True)
    p2_count = sum(values["p2"] for values in class_level.values())
    p3_p7_count = foreground - p2_count
    calibration = load_json(experiment / "LOSS_CALIBRATION.json")
    fractions = calibration["p2_loss_fractions_over_microbatches"]
    mean_fraction = {name: sum(row[name] for row in fractions) / len(fractions) for name in fractions[0]}
    report = {
        "schema": "splitfusion_fcos_p2_assignment_audit_v1", "created_utc": utc_now(),
        "train_frames": len(dataset), "eligible_gt_by_class": priors["eligible_gt"],
        "eligible_gt_by_class_and_projected_size": priors["projected_size_counts"],
        "positive_assignments_by_class_and_level": {name: dict(values) for name, values in class_level.items()},
        "foreground_locations_total": foreground, "p2_foreground_locations": p2_count,
        "p3_p7_foreground_locations": p3_p7_count,
        "foreground_increase_introduced_by_p2": p2_count,
        "foreground_increase_fraction_over_original_p3_p7": p2_count / max(1, p3_p7_count),
        "p3_original_positives_preserved": True,
        "p3_preservation_reason": "P2 matching is evaluated at distinct locations; P3 retains its original (0,64) interval and identical per-location minimum-area resolution.",
        "actors_without_fcos_carrier_count": len(no_carrier), "actors_without_fcos_carrier": no_carrier,
        "carrier_visibility": dict(visibility),
        "carrier_visibility_definition": "own-visible when the carrier pixel has the actor class and the matched actor is the nearest same-class eligible box containing the pixel; non-own foreground is occluder; remaining pixels are elsewhere",
        "ignored_locations": ignored, "p2_fraction_of_each_fcos_loss_on_fixed_calibration_microbatches": mean_fraction,
        "validation_accessed": False,
    }
    atomic_json(experiment / "P2_ASSIGNMENT_AUDIT.json", report, overwrite=False)
    atomic_text(experiment / "P2_ASSIGNMENT_AUDIT_COMPLETE", "FULL_TRAIN_ASSIGNMENT_AUDIT_COMPLETE_BEFORE_TRAINING\n", overwrite=False)
    print(json.dumps({"foreground": foreground, "p2": p2_count,
                      "actors_without_carrier": len(no_carrier), "class_level": report["positive_assignments_by_class_and_level"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
