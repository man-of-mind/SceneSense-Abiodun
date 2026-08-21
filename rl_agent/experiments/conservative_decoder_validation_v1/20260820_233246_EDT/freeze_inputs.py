#!/usr/bin/env python3
"""Freeze source identity and prove original split integrity before inference."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
AB = HERE.parents[3]
DATASET = AB / "fusion_training_data" / "moving_ego_pps200000_merged_8loops_stride2"
EXPECTED_COUNTS = {"train": 10911, "val": 2110, "test": 2162}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def run_text(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=AB, text=True, capture_output=True, check=False)
    return result.returncode, (result.stdout + result.stderr).strip()


def main() -> int:
    if (HERE / "PREINFERENCE_FREEZE.json").exists():
        raise RuntimeError("PREINFERENCE_FREEZE.json already exists; refusing to refreeze")

    config = json.loads((HERE / "resolved_config.json").read_text(encoding="utf-8"))
    if config["dataset"] != str(DATASET.relative_to(AB)):
        raise AssertionError("Resolved dataset does not match freeze target")

    by_split: dict[str, list[str]] = {name: [] for name in EXPECTED_COUNTS}
    asset_columns = ("rgb_path", "mask_path", "radar_tensor_path", "radar_points_path")
    missing_assets: list[dict[str, str]] = []
    manifest_ids: set[str] = set()
    with (DATASET / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = str(row.get("split", ""))
            sample_id = str(row.get("sample_id", ""))
            if split not in by_split or not sample_id:
                raise AssertionError(f"Unexpected manifest split/sample: {split!r}/{sample_id!r}")
            if sample_id in manifest_ids:
                raise AssertionError(f"Duplicate manifest sample_id: {sample_id}")
            manifest_ids.add(sample_id)
            by_split[split].append(sample_id)
            for column in asset_columns:
                relative = str(row.get(column, ""))
                if not relative or not (DATASET / relative).is_file():
                    missing_assets.append({"sample_id": sample_id, "column": column, "path": relative})

    counts = {name: len(values) for name, values in by_split.items()}
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"Split counts differ: {counts} != {EXPECTED_COUNTS}")
    if missing_assets:
        raise AssertionError(f"Missing dataset assets: {missing_assets[:10]}")

    split_sets = {name: set(values) for name, values in by_split.items()}
    overlaps = {
        "train_val": sorted(split_sets["train"] & split_sets["val"]),
        "train_test": sorted(split_sets["train"] & split_sets["test"]),
        "val_test": sorted(split_sets["val"] & split_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError("Original train/val/test identifiers overlap")

    split_dir = HERE / "split_identifiers"
    split_dir.mkdir(exist_ok=True)
    split_records: dict[str, dict[str, object]] = {}
    historical_dirs = {
        "noae": AB / "experiments/ae_integrated_20260710/noae_baseline/splits",
        "ae32": AB / "experiments/ae_integrated_20260710/ae32/splits",
        "ae64": AB / "experiments/ae_integrated_20260710/ae64/splits",
        "ae128": AB / "experiments/ae_integrated_20260710/ae128/splits",
    }
    agreement: dict[str, dict[str, bool]] = {}
    for split, values in by_split.items():
        frozen_path = split_dir / f"{split}.txt"
        frozen_path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
        split_records[split] = {
            "count": len(values),
            "sha256": sha256(frozen_path),
            "first_identifier": values[0],
            "last_identifier": values[-1],
        }
    for family, directory in historical_dirs.items():
        agreement[family] = {}
        for split, values in by_split.items():
            historical = (directory / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            exact = historical == values
            agreement[family][split] = exact
            if not exact:
                raise AssertionError(f"Manifest/{family} {split} order or identity mismatch")

    integrity = {
        "dataset": str(DATASET),
        "manifest_unique_identifier_count": len(manifest_ids),
        "split_counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "pairwise_overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "pairwise_overlap_identifiers": overlaps,
        "all_checkpoint_split_files_exact_order_match": agreement,
        "frozen_identifier_files": split_records,
        "asset_columns_checked": list(asset_columns),
        "asset_references_checked": len(manifest_ids) * len(asset_columns),
        "missing_asset_references": 0,
        "validation_identifier_count_proven": len(by_split["val"]) == 2110,
        "disjointness_proven": not any(overlaps.values()),
    }
    atomic_json(HERE / "split_integrity.json", integrity)

    source_paths = [
        ("dataset_manifest", DATASET / "manifest.csv"),
        ("dataset_object_manifest", DATASET / "object_boxes.csv"),
        ("dataset_metadata", DATASET / "metadata.json"),
        ("dataset_merge_summary", DATASET / "merge_summary.json"),
        ("fusion_config", AB / "pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"),
        ("evaluator_source", AB / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/evaluate_fusion.py"),
        ("decoder_source", AB / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/object_targets.py"),
        ("split_runtime_source", AB / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/split_runtime.py"),
        ("model_source", AB / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/model.py"),
        ("common_source", AB / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/common.py"),
        ("training_source", AB / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/train_fusion.py"),
        ("density_reference_evaluator", AB / "rl_agent/density_knob/density_knob_eval.py"),
        ("prior_audit_report", AB / "rl_agent/experiments/model_precision_decoder_audit_v1/20260819_210004/REPORT.md"),
        ("prior_audit_results", AB / "rl_agent/experiments/model_precision_decoder_audit_v1/20260819_210004/RESULTS_SUMMARY.json"),
        ("prior_audit_driver", AB / "rl_agent/experiments/model_precision_decoder_audit_v1/20260819_210004/run_decoder_audit.py"),
        ("catalog_report", AB / "rl_agent/experiments/ue_split_catalog_proposal_v1/20260820_042414_candidate/REPORT.md"),
        ("catalog_rows", AB / "rl_agent/experiments/ue_split_catalog_proposal_v1/20260820_042414_candidate/ue_split_candidate_catalog.csv"),
        ("preregistered_plan", HERE / "PLAN.md"),
        ("preregistered_config", HERE / "resolved_config.json"),
        ("freeze_driver", HERE / "freeze_inputs.py"),
        ("inference_driver", HERE / "offline_inference.py"),
        ("analysis_driver", HERE / "analyze_study.py"),
    ]
    checkpoints = {
        "noae": AB / "experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt",
        "ae32": AB / "experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt",
        "ae64": AB / "experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt",
        "ae128": AB / "experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt",
    }
    trial_summaries = {
        "noae": AB / "experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/trial_summary.json",
        "ae32": AB / "experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/trial_summary.json",
        "ae64": AB / "experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/trial_summary.json",
        "ae128": AB / "experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/trial_summary.json",
    }
    for family, path in checkpoints.items():
        source_paths.append((f"checkpoint_{family}", path))
    for family, path in trial_summaries.items():
        source_paths.append((f"checkpoint_config_{family}", path))
    for family, directory in historical_dirs.items():
        for split in EXPECTED_COUNTS:
            source_paths.append((f"historical_split_{family}_{split}", directory / f"{split}.txt"))

    files: list[dict[str, object]] = []
    for role, path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required {role} is missing: {path}")
        stat = path.stat()
        files.append({
            "role": role,
            "path": str(path.relative_to(AB)) if path.is_relative_to(AB) else str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path),
        })
    atomic_json(HERE / "input_hash_manifest.json", {
        "audit_id": config["audit_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    })

    git_code, git_head = run_text("git", "rev-parse", "HEAD")
    status_code, git_status = run_text("git", "status", "--short", "--untracked-files=no")
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available_at_freeze": bool(torch.cuda.is_available()),
            "cudnn_version": torch.backends.cudnn.version(),
        }
    except Exception as exc:  # pragma: no cover - recorded as evidence
        torch_info = {"import_error": repr(exc), "cuda_available_at_freeze": False}
    environment = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "git_head": git_head if git_code == 0 else None,
        "git_head_command_exit": git_code,
        "git_tracked_dirty_status": git_status if status_code == 0 else None,
        "git_status_command_exit": status_code,
        "torch": torch_info,
    }
    atomic_json(HERE / "environment_provenance.json", environment)

    freeze_files = [
        HERE / "PLAN.md",
        HERE / "resolved_config.json",
        HERE / "split_integrity.json",
        HERE / "input_hash_manifest.json",
        HERE / "environment_provenance.json",
        *(split_dir / f"{split}.txt" for split in EXPECTED_COUNTS),
    ]
    atomic_json(HERE / "PREINFERENCE_FREEZE.json", {
        "status": "FROZEN_BEFORE_INFERENCE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_identifiers": 2110,
        "disjointness_proven": True,
        "frozen_files": [
            {"path": str(path.relative_to(HERE)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in freeze_files
        ],
    })
    print(json.dumps({
        "status": "FROZEN_BEFORE_INFERENCE",
        "counts": counts,
        "validation_sha256": split_records["val"]["sha256"],
        "test_sha256": split_records["test"]["sha256"],
        "files_hashed": len(files),
        "cuda_available_at_freeze": torch_info.get("cuda_available_at_freeze"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
