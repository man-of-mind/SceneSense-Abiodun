from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import CONFIG_PATH, ROOT, load_json, read_csv, sha256, utc_now, write_json_x, write_text_x
from data import DepthCache, InferenceDataset
from model import build_model, freeze_bn_running_state, split_report

PACKAGE = Path(__file__).resolve().parent
SCORING = PACKAGE.parent / "route_b_v3_1_native_grid_expanded_training_v2/scoring_v2.py"
VAL_CACHE_SOURCE = ROOT / "experiments/route_b_v3_1_depth_aware_lraspp_v1/20260829_060656/depth_cache/val"


def load_scoring() -> Any:
    spec = importlib.util.spec_from_file_location("two_stage_frozen_scoring", SCORING)
    if spec is None or spec.loader is None: raise ImportError(SCORING)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def macro_depth(frame_values: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    result = {}
    for band in ("overall", "20_30", "30_40"):
        episodes = {episode: float(np.mean(values[band])) for episode, values in frame_values.items() if values[band]}
        result[band] = {"episode_macro_log_mae": float(np.mean(list(episodes.values()))),
                        "episodes": episodes, "episode_count": len(episodes),
                        "frames": sum(len(values[band]) for values in frame_values.values())}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True); started = time.monotonic()
    if not (experiment / "stage1/TRAINING_COMPLETE").is_file():
        raise RuntimeError("Stage-1 evaluation begins only after all 20 epochs complete")
    config = load_json(CONFIG_PATH); root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(root / "dataset/manifest.csv") if row["split"] == "val"]
    if len(rows) != 3345: raise RuntimeError("validation population drift")
    cache_target = experiment / "depth_cache/val"
    if not cache_target.exists():
        os.symlink(os.path.relpath(VAL_CACHE_SOURCE, cache_target.parent), cache_target, target_is_directory=True)
    cache_report = load_json(cache_target / "CACHE_REPORT.json")
    # Validation depth artifacts are opened and hashed only here, after Stage-1
    # optimization, and only for the preregistered depth evaluation.
    actual_cache = {name: sha256(cache_target / name) for name in
                    ("CACHE_REPORT.json", "index.csv", "depth_forward_f16.bin", "valid_u8.bin")}
    cache = DepthCache(cache_target, rows); dataset = InferenceDataset(root, rows)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda"); scoring = load_scoring(); native = scoring.native_evaluator()
    evaluation = experiment / "stage1/evaluation"; evaluation.mkdir(parents=True, exist_ok=True)
    records = []
    for epoch in (10, 20):
        checkpoint_path = experiment / f"stage1/checkpoints/epoch_{epoch:03d}.pt"
        checkpoint_hash = sha256(checkpoint_path); payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != epoch: raise RuntimeError("Stage-1 checkpoint epoch drift")
        model, _ = build_model(Path(config["pretrained"]["path"]), device)
        model.load_state_dict(payload["model"], strict=True); model.eval(); freeze_bn_running_state(model)
        prediction_root = experiment / f"stage1/predictions/epoch_{epoch:03d}"
        prediction_root.mkdir(parents=True, exist_ok=False); (prediction_root / "segmentation").mkdir()
        segmentation_rows = []; frame_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {name: [] for name in ("overall", "20_30", "30_40")})
        finite = True; torch.cuda.reset_peak_memory_stats(device); epoch_start = time.monotonic()
        with torch.inference_mode():
            for index in range(len(dataset)):
                value, row = dataset[index]; depth, valid, _radar = cache.get(row["sample_id"])
                value_gpu = value.unsqueeze(0).to(device); output = model(value_gpu, dense=True)
                finite = finite and bool(torch.isfinite(output["out"]).all()) and bool(torch.isfinite(output["dense_depth_log1p"]).all())
                source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                labels = F.interpolate(output["out"], size=source_hw, mode="bilinear", align_corners=False)
                labels = labels.argmax(1)[0].cpu().numpy().astype(np.uint8)
                relative = Path("segmentation") / f"{row['sample_id']}.png"; path = prediction_root / relative
                if not cv2.imwrite(str(path), labels): raise RuntimeError(f"cannot write {path}")
                segmentation_rows.append({"sample_id": row["sample_id"], "prediction_path": str(relative),
                                          "width": labels.shape[1], "height": labels.shape[0], "sha256": sha256(path)})
                prediction = output["dense_depth_log1p"][0, 0].double().cpu()
                target = depth.double(); base = valid & torch.isfinite(target) & target.gt(0) & target.le(40)
                episode = row.get("experiment_id", row.get("source_experiment", ""))
                if not episode: episode = row["sample_id"].rsplit("_", 2)[0]
                for band, mask in (("overall", base), ("20_30", base & target.ge(20) & target.lt(30)),
                                   ("30_40", base & target.ge(30) & target.le(40))):
                    if mask.any(): frame_values[episode][band].append(float((prediction[mask] - torch.log1p(target[mask])).abs().mean()))
                if (index + 1) % 500 == 0: print(f"[stage1 eval {epoch}] {index+1}/{len(dataset)}", flush=True)
        manifest_path = prediction_root / "segmentation_manifest.csv"
        with manifest_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"))
            writer.writeheader(); writer.writerows(segmentation_rows)
        frame_ids = [row["sample_id"] for row in rows]
        segmentation = native.score_segmentation(root, "v010", frame_ids, prediction_root, manifest_path)
        depth_metric = macro_depth(frame_values); parity = split_report(model, dataset[0][0].unsqueeze(0).to(device))
        g = config["stage1_gates"]; baseline = g["constant_train_episode_macro_log_mae"]
        gates = {
            "vehicle_iou": segmentation["vehicle_iou"] >= g["vehicle_iou_min"],
            "person_box_mask_iou": segmentation["person_box_mask_iou"] >= g["person_box_mask_iou_min"],
            "foreground_miou": segmentation["foreground_miou"] >= g["foreground_miou_min"],
            "segmentation_dense_finite": finite, "split_monolithic_parity": parity["all_raw_equal"],
            "depth_overall": depth_metric["overall"]["episode_macro_log_mae"] <= .90 * baseline["overall"],
            "depth_20_30": depth_metric["20_30"]["episode_macro_log_mae"] <= .95 * baseline["20_30"],
            "depth_30_40": depth_metric["30_40"]["episode_macro_log_mae"] <= .95 * baseline["30_40"],
            "both_validation_episodes_finite": all(item["episode_count"] == 2 and
                all(math.isfinite(value) for value in item["episodes"].values()) for item in depth_metric.values()),
            "no_deployable_inference_depth": True,
        }
        record = {"schema": "two_stage_lraspp_stage1_evaluation_v1", "created_utc": utc_now(),
            "epoch": epoch, "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
            "validation_frames": len(rows), "segmentation": segmentation, "dense_depth": depth_metric,
            "constant_train_baseline": baseline, "gates": gates, "pass": all(gates.values()),
            "all_outputs_finite": finite, "split_report": parity,
            "validation_depth_access": {"authorized_stage1_evaluation_only": True,
                "cache_source": str(VAL_CACHE_SOURCE.relative_to(ROOT)), "cache_report": cache_report,
                "actual_file_hashes": actual_cache},
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "wall_seconds": time.monotonic() - epoch_start}
        write_json_x(evaluation / f"epoch_{epoch:03d}.json", record); records.append(record)
        print(json.dumps({"epoch": epoch, "segmentation": segmentation, "dense_depth": depth_metric,
                          "pass": record["pass"]}), flush=True)
        del model; torch.cuda.empty_cache()
    selected = next((record for record in records if record["pass"]), None)
    decision = {"schema": "two_stage_lraspp_stage1_selection_v1", "created_utc": utc_now(),
        "evaluated_epochs": [10, 20], "passing_epochs": [record["epoch"] for record in records if record["pass"]],
        "selection_rule": "earliest passing epoch", "selected_epoch": selected["epoch"] if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "stage2_authorized": selected is not None, "wall_seconds": time.monotonic() - started}
    write_json_x(experiment / "STAGE1_SELECTION.json", decision)
    if selected is None:
        write_text_x(experiment / "TERMINAL_VERDICT.txt", "TWO_STAGE_LRASPP_STAGE1_REPRESENTATION_FAILED\n")
        write_text_x(experiment / "EVALUATION_COMPLETE", "TWO_STAGE_LRASPP_STAGE1_REPRESENTATION_FAILED\n")
    print(json.dumps(decision, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
