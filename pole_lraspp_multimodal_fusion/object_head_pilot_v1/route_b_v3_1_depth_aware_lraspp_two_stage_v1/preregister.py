from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from common import CONFIG_PATH, ROOT, load_json, read_csv, sha256, utc_now, write_json_x, write_text_x
from model import build_model
from two_stage import parameter_allowlist, parameter_counts

PACKAGE = Path(__file__).resolve().parent
CACHE_SOURCE = ROOT / "experiments/route_b_v3_1_depth_aware_lraspp_v1/20260829_042423/depth_cache/train"
CACHE_HASHES = {
    "CACHE_REPORT.json": "6413abfce4f9600b579e9f30c621db16da63c6b8f24dd0196eab6d4ad9d5ddbb",
    "index.csv": "d978b1c93a2bcd6292d0e320f5ceb4b04a73d7c82c29537038ddab96b57db13b",
    "depth_forward_f16.bin": "ec75d0a776097f6fb8a582e98e1fe907a7d0032267c1fcae27eb5a8937bf00ed",
    "valid_u8.bin": "5ec480cb3d2eefa9cba3d35368484ae22d09e253dbc337d7ba10230b67304ee8",
    "radar_consistency_f32.bin": "89f630ddeb32fbf41c83eed42b7ae7dd78ea10c492984fe6ae2764483ebbbdcf",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve()
    experiment.mkdir(parents=True, exist_ok=False)
    config = load_json(CONFIG_PATH)
    head = git("rev-parse", "HEAD")
    if not git("merge-base", "--is-ancestor", "8e556a680516638a720d9d37656493721b2bea87", "HEAD") == "":
        raise RuntimeError("required lineage absent")
    if git("branch", "--show-current") != "master":
        raise RuntimeError("not local master")
    if config["source_commit"] == "PREREGISTRATION_COMMIT_PENDING":
        raise RuntimeError("source implementation must be committed before preregistration")
    weight = Path(config["pretrained"]["path"])
    if sha256(weight) != config["pretrained"]["sha256"]:
        raise RuntimeError("official MobileNet hash drift")
    for name, expected in CACHE_HASHES.items():
        if sha256(CACHE_SOURCE / name) != expected:
            raise RuntimeError(f"train cache hash drift: {name}")
    cache_parent = experiment / "depth_cache"; cache_parent.mkdir()
    os.symlink(os.path.relpath(CACHE_SOURCE, cache_parent), cache_parent / "train",
               target_is_directory=True)
    dataset_root = ROOT / config["dataset_root"]
    manifest = read_csv(dataset_root / "dataset/manifest.csv")
    train = [row for row in manifest if row["split"] == "train"]
    val = [row for row in manifest if row["split"] == "val"]
    if len(train) != 16827 or len(val) != 3345 or len({row["sample_id"] for row in train} &
                                                       {row["sample_id"] for row in val}):
        raise RuntimeError("dataset split drift")
    torch.manual_seed(int(config["stage1_seed"]))
    model, loading = build_model(weight, torch.device("cpu"))
    allowlists = {stage: parameter_allowlist(model, stage) for stage in ("stage1", "stage2")}
    counts = {stage: parameter_counts(model, stage) for stage in ("stage1", "stage2")}
    sources = sorted(path for path in PACKAGE.glob("*.py")) + [CONFIG_PATH]
    source_hashes = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                     for path in sources}
    resolved = {
        "schema": "two_stage_lraspp_resolved_config_v1", "created_utc": utc_now(),
        "experiment": str(experiment.relative_to(ROOT)), "config": config,
        "config_path": str(CONFIG_PATH.relative_to(ROOT)), "config_sha256": sha256(CONFIG_PATH),
        "source_commit": config["source_commit"], "preregistration_head": head,
        "branch": "master", "source_hashes": source_hashes,
    }
    write_json_x(experiment / "RESOLVED_CONFIG.json", resolved)
    design = {
        "schema": "two_stage_lraspp_registered_design_v1", "created_utc": utc_now(),
        "hypothesis": "A segmentation/depth-pretrained representation frozen against object gradients preserves semantic/range quality while private heads learn detection/localization.",
        "final_lraspp_experiment": True, "stage3_allowed": False, "joint_finetuning_allowed": False,
        "lineage": {"required_ancestor": "8e556a680516638a720d9d37656493721b2bea87",
                    "source_commit": config["source_commit"], "registration_head": head,
                    "official_weight": {"path": str(weight), "sha256": sha256(weight),
                                        "enum": config["pretrained"]["enum"]}},
        "data": {"train_frames": 16827, "train_episodes": 10, "validation_frames": 3345,
                 "validation_episodes": 2, "test": "absent_unopened", "primary": "v010",
                 "selected_only_sensitivity": "v025", "manifest_sha256": sha256(dataset_root / "dataset/manifest.csv")},
        "train_depth_cache": {"mode": "verified_read_only_symlink", "source": str(CACHE_SOURCE.relative_to(ROOT)),
                              "hashes": CACHE_HASHES},
        "parameter_allowlists": allowlists, "parameter_counts": counts,
        "seeds": {"stage1": int(config["stage1_seed"]), "stage2": int(config["stage2_seed"]),
                  "stage2_initialization": int(config["stage2_initialization_seed"]),
                  "derivation": "python=numpy=torch=cuda=sampler base; epoch sampler=base+epoch"},
        "stage1": config["training"]["stage1"], "stage2": config["training"]["stage2"],
        "loss_weights": config["loss_weights"], "stage1_gates": config["stage1_gates"],
        "stage2_gates": config["stage2_gates"], "inference": config["inference"],
        "selection": {"stage1": "earliest of epochs 10,20 passing every gate",
                      "stage2": config["stage2_gates"]["ranking"]},
        "expected_counts": {"stage1_optimizer_epochs": 20, "stage1_checkpoints_including_epoch000": 21,
                            "stage1_evaluations": 2, "stage2_optimizer_epochs_if_authorized": 30,
                            "stage2_checkpoints_including_epoch000_if_authorized": 31,
                            "stage2_evaluations_if_authorized": 3, "v025_if_selected_eligible": 1},
        "terminals": config["terminals"], "exactly_one_terminal": True,
        "validation_policy": "none during optimization; Stage1 only after epoch20; Stage2 only after epoch30",
        "depth_baseline_interpretation": "The scalar and per-band train episode-macro log-MAE values are frozen before validation is opened; candidate validation aggregation is compared directly to these preregistered train constants.",
        "official_loading": loading, "source_hashes": source_hashes,
        "prohibited": ["test", "CARLA", "OAI contents", "q/AE training", "live split runtime", "288 measurements", "push"],
    }
    write_json_x(experiment / "REGISTERED_TWO_STAGE_DESIGN.json", design)
    md = f"""# Registered Route B v3.1 two-stage LR-ASPP design

Registered: {design['created_utc']}
Source commit: `{config['source_commit']}`
Config SHA-256: `{resolved['config_sha256']}`

This is the final LR-ASPP experiment. Stage 1 trains only the representation allowlist for exactly 20 epochs with segmentation, dense-depth and radar-consistency losses. Stage 2 is forbidden unless the earliest passing Stage-1 epoch (10 then 20) satisfies all frozen semantic and train-baseline-relative depth gates. If authorized, both private object branches are deterministically reset and trained alone for exactly 30 epochs. There is no joint fine-tuning, Stage 3, threshold sweep or LR-ASPP follow-up.

The exact parameter allowlists, seeds, optimizer schedules, baseline values, gates, evaluation checkpoints, selection order, expected artifact counts, source hashes and six exclusive terminals are recorded in `REGISTERED_TWO_STAGE_DESIGN.json`. Validation is not opened during optimization. Deployable inference accepts only RGB-radar input and never a depth label.

Constant depth is {config['stage1_gates']['constant_train_median_depth_m']} m (`log1p` {config['stage1_gates']['constant_train_median_log1p']}). Frozen train episode-macro log-MAE values are {json.dumps(config['stage1_gates']['constant_train_episode_macro_log_mae'], sort_keys=True)}.
"""
    write_text_x(experiment / "REGISTERED_TWO_STAGE_DESIGN.md", md)
    write_json_x(experiment / "CACHE_REFERENCE.json", {
        "schema": "two_stage_lraspp_train_cache_reference_v1", "created_utc": utc_now(),
        "source": str(CACHE_SOURCE.relative_to(ROOT)), "mode": "verified_read_only_symlink", "hashes": CACHE_HASHES,
    })
    print(json.dumps({"experiment": str(experiment), "config_sha256": resolved["config_sha256"],
                      "source_files": len(source_hashes), "stage1_parameters": counts["stage1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
