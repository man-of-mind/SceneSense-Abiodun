#!/usr/bin/env python3
"""Compare frozen noAE q=0 perception with 200 ms versus current 100 ms radar.

This is deliberately a small, paired sensitivity check.  It does not train a
model and it does not redefine the production sensor contract.  RGB, frame
identity, model weights, UINT8/zstd transport, p025 policy, and scoring remain
fixed; only the radar points rasterized into the four radar channels change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.radar_fusion import (
    rasterize_radar_channels_fast,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    phase8b_uint8_validation as phase8b,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_common import (
    load_frozen_scorers,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase6_validation import (
    _person_only,
    score_validation_pass,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
    ZstdWireCodec,
)


EXECUTE_TOKEN = "SPLITFUSION_QUICK_RADAR_SUPPORT_ABLATION_V1"
SCHEMA = "splitfusion.quick_radar_temporal_support_ablation.v1"
TERMINAL = "SPLITFUSION_QUICK_RADAR_SUPPORT_ABLATION_COMPLETE"
DEFAULT_SAMPLES = 512
CONTENT_H = 432
CONTENT_W = 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    with staging.open("w", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def evenly_spaced(indices: Sequence[int], count: int) -> list[int]:
    """Choose deterministic route-spanning indices without duplicates."""
    values = list(indices)
    if count < 0 or count > len(values):
        raise ValueError("invalid evenly-spaced sample count")
    if count == 0:
        return []
    if count == len(values):
        return values
    raw = np.linspace(0, len(values) - 1, num=count)
    selected = [values[int(round(position))] for position in raw]
    if len(set(selected)) != count:
        # This should be unreachable for count <= population, but fail closed.
        raise RuntimeError("evenly-spaced selection produced duplicate indices")
    return selected


def balanced_subset(
    frame_ids: Sequence[str], episode_by_sample: Mapping[str, str], count: int
) -> list[int]:
    """Balance the deterministic subset across the registered episodes."""
    if count <= 0 or count > len(frame_ids):
        raise ValueError("sample count must be within the validation population")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(frame_ids):
        groups[str(episode_by_sample[sample_id])].append(index)
    episodes = sorted(groups)
    if not episodes:
        raise RuntimeError("no validation episodes")
    base, remainder = divmod(count, len(episodes))
    selected: list[int] = []
    for rank, episode in enumerate(episodes):
        quota = base + int(rank < remainder)
        selected.extend(evenly_spaced(groups[episode], quota))
    # Preserve the frozen scorer's global frame ordering.
    selected.sort()
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("balanced subset coverage failure")
    return selected


def raster_from_points(payload: Mapping[str, np.ndarray], current_only: bool) -> np.ndarray:
    required = {
        "u", "v", "camera_depth_m", "velocity_mps", "stationary_age_s",
        "valid_projection", "sweep_offset",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"radar-point schema missing {sorted(missing)}")
    offsets = np.asarray(payload["sweep_offset"])
    if offsets.ndim != 1 or not set(np.unique(offsets).tolist()).issubset({0, 1}):
        raise RuntimeError("unexpected radar sweep offsets")
    mask = offsets == 0 if current_only else np.ones(offsets.shape, dtype=bool)
    if not np.any(mask):
        raise RuntimeError("selected radar support is empty")
    return rasterize_radar_channels_fast(
        width=CONTENT_W,
        height=CONTENT_H,
        u=np.asarray(payload["u"])[mask],
        v=np.asarray(payload["v"])[mask],
        depth_m=np.asarray(payload["camera_depth_m"])[mask],
        velocity_mps=np.asarray(payload["velocity_mps"])[mask],
        stationary_age_s=np.asarray(payload["stationary_age_s"])[mask],
        valid_mask=np.asarray(payload["valid_projection"])[mask].astype(bool),
        max_range_m=120.0,
        max_abs_velocity_mps=20.0,
        parked_threshold_s=5.0,
        point_radius_px=4,
    )


class CurrentSweepDataset:
    """Replace only radar channels in an existing frozen inference dataset."""

    def __init__(self, source: Any, points_by_sample: Mapping[str, str]) -> None:
        self.source = source
        self.rows = source.rows
        self.dataset = source.dataset
        self.points_by_sample = dict(points_by_sample)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str], dict[str, torch.Tensor]]:
        fused, row, calibration = self.source[index]
        relative = self.points_by_sample.get(str(row["sample_id"]))
        if relative is None:
            raise RuntimeError(f"missing radar points for {row['sample_id']}")
        with np.load(self.dataset / relative, allow_pickle=False) as payload:
            radar = raster_from_points(payload, current_only=True)
        result = fused.clone()
        result[3:7, :CONTENT_H, :CONTENT_W] = torch.from_numpy(
            np.ascontiguousarray(radar)
        )
        return result, row, calibration


def load_points_index(dataset_root: Path) -> tuple[dict[str, str], Path]:
    manifest = dataset_root / "dataset" / "manifest.csv"
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    index = {
        str(row["sample_id"]): str(row["radar_points_path"])
        for row in rows if str(row.get("split")) == "val"
    }
    if len(index) != contract.VALIDATION_FRAMES:
        raise RuntimeError(
            f"validation radar-point index has {len(index)} rows, expected "
            f"{contract.VALIDATION_FRAMES}"
        )
    return index, manifest


def qualify_reconstruction(
    *, source: Any, positions: Sequence[int], points_by_sample: Mapping[str, str],
    maximum_checks: int = 32,
) -> dict[str, Any]:
    checks = evenly_spaced(list(range(len(positions))), min(maximum_checks, len(positions)))
    maximum_error = 0.0
    current_counts: list[int] = []
    previous_counts: list[int] = []
    for selected_index in checks:
        position = positions[selected_index]
        row = source.rows[position]
        persisted = np.load(source.dataset / row["radar_tensor_path"], allow_pickle=False)
        with np.load(
            source.dataset / points_by_sample[str(row["sample_id"])], allow_pickle=False
        ) as payload:
            reconstructed = raster_from_points(payload, current_only=False)
            offsets = np.asarray(payload["sweep_offset"])
            current_counts.append(int(np.count_nonzero(offsets == 0)))
            previous_counts.append(int(np.count_nonzero(offsets == 1)))
        error = float(np.max(np.abs(persisted - reconstructed)))
        maximum_error = max(maximum_error, error)
        if not np.array_equal(persisted, reconstructed):
            raise RuntimeError(
                f"200-ms reconstruction is not bit-exact for {row['sample_id']}"
            )
    return {
        "checked_frames": len(checks),
        "bit_exact": True,
        "maximum_absolute_error": maximum_error,
        "current_returns": {
            "minimum": min(current_counts),
            "median": float(np.median(current_counts)),
            "maximum": max(current_counts),
        },
        "previous_returns": {
            "minimum": min(previous_counts),
            "median": float(np.median(previous_counts)),
            "maximum": max(previous_counts),
        },
    }


def run_support(
    *, runtime: dict[str, Any], inference: Any, positions: Sequence[int],
    frame_ids: Sequence[str], output: Path, workers: int,
) -> dict[str, Any]:
    local_runtime = dict(runtime)
    local_runtime["inference"] = inference
    local_runtime["positions"] = list(positions)
    local_runtime["frame_ids"] = list(frame_ids)
    original_count = contract.VALIDATION_FRAMES
    try:
        # The frozen runner's coverage assertions use this constant. This
        # process-local diagnostic override changes no model/scoring behavior.
        contract.VALIDATION_FRAMES = len(frame_ids)
        return phase8b.run_validation_pass(
            runtime=local_runtime,
            q=0.0,
            output=output,
            workers=workers,
            wire=ZstdWireCodec(),
        )
    finally:
        contract.VALIDATION_FRAMES = original_count


def score_support(
    *, raw: Mapping[str, Any], runtime: Mapping[str, Any], scorers: Any,
    frame_ids: Sequence[str], gt: Mapping[str, Any], ignore_cache: dict[str, Any],
) -> dict[str, Any]:
    selected_gt = {sample_id: gt.get(sample_id, []) for sample_id in frame_ids}
    # The frozen AVO scorer checks that its eligibility partition covers every
    # row in ``qualified_gt``.  For this deliberately bounded subset, give it
    # the matching truth subset rather than the full 3,345-frame mapping.
    selected_truth = dict(runtime["truth"])
    for key in ("qualified_gt", "structural_gt"):
        source = runtime["truth"][key]
        selected_truth[key] = {
            sample_id: source.get(sample_id, []) for sample_id in frame_ids
        }
    selected_truth["episode_by_sample"] = {
        sample_id: runtime["truth"]["episode_by_sample"][sample_id]
        for sample_id in frame_ids
    }
    return score_validation_pass(
        result=raw,
        scorers=scorers,
        truth=selected_truth,
        experiment=runtime["dataset_root"],
        frame_ids=frame_ids,
        gt=selected_gt,
        person_gt=_person_only(selected_gt),
        ignore_cache=ignore_cache,
    )


def finite_metrics(scored: Mapping[str, Any]) -> dict[str, float]:
    values = {str(key): float(value) for key, value in scored["metrics"].items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("non-finite metric in ablation result")
    return values


def interpretation(baseline: Mapping[str, float], current: Mapping[str, float]) -> str:
    lower_is_better = {"vehicle_xy_mae_m", "person_avo_xy_mae_m"}
    directions: list[int] = []
    for name, old in baseline.items():
        new = current[name]
        delta = (old - new) if name in lower_is_better else (new - old)
        directions.append(0 if abs(delta) < 1e-12 else (1 if delta > 0 else -1))
    nonzero = [value for value in directions if value]
    if nonzero and all(value > 0 for value in nonzero):
        return "CURRENT_100MS_DIRECTIONALLY_BETTER_ON_ALL_CHANGED_METRICS"
    if nonzero and all(value < 0 for value in nonzero):
        return "ROLLING_200MS_DIRECTIONALLY_BETTER_ON_ALL_CHANGED_METRICS"
    return "MIXED_OR_EFFECTIVELY_EQUAL_RETAIN_EXISTING_200MS_CONTRACT"


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen perception comparison")
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    work = args.output / "working_predictions"
    work.mkdir()
    started = time.time()

    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)
    torch.set_num_threads(1)
    binding, _references = phase8b.bind_inputs()
    runtime = phase8b._load_runtime(device)
    points_by_sample, manifest = load_points_index(runtime["dataset_root"])

    subset_indices = balanced_subset(
        runtime["frame_ids"], runtime["truth"]["episode_by_sample"], args.samples
    )
    positions = [runtime["positions"][index] for index in subset_indices]
    frame_ids = [runtime["frame_ids"][index] for index in subset_indices]
    episodes: dict[str, int] = defaultdict(int)
    for sample_id in frame_ids:
        episodes[str(runtime["truth"]["episode_by_sample"][sample_id])] += 1

    qualification = qualify_reconstruction(
        source=runtime["inference"],
        positions=positions,
        points_by_sample=points_by_sample,
        maximum_checks=args.reconstruction_checks,
    )
    scorers = load_frozen_scorers()
    gt, _states = scorers.load_gt(runtime["dataset_root"], contract.PRIMARY_CONTRACT)
    ignore_cache: dict[str, Any] = {}
    supports = {
        "rolling_200ms": runtime["inference"],
        "current_100ms": CurrentSweepDataset(runtime["inference"], points_by_sample),
    }
    compact: dict[str, Any] = {}
    for name, inference in supports.items():
        prediction_root = work / name
        raw = run_support(
            runtime=runtime,
            inference=inference,
            positions=positions,
            frame_ids=frame_ids,
            output=prediction_root,
            workers=args.workers,
        )
        phase8b._require_state_unchanged(runtime)
        scored = score_support(
            raw=raw,
            runtime=runtime,
            scorers=scorers,
            frame_ids=frame_ids,
            gt=gt,
            ignore_cache=ignore_cache,
        )
        compact[name] = {
            "frames": int(scored["frames"]),
            "metrics": finite_metrics(scored),
            "canonical_person_metrics": {
                key: float(value)
                for key, value in scored["canonical_person_metrics"].items()
            },
            "compressed_zstd_bytes": dict(scored["compressed_zstd_bytes"]),
            "wall_seconds": float(scored["wall_seconds"]),
        }
        shutil.rmtree(prediction_root)

    baseline = compact["rolling_200ms"]["metrics"]
    current = compact["current_100ms"]["metrics"]
    deltas = {name: current[name] - baseline[name] for name in baseline}
    document = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "scope": {
            "purpose": "quick frozen-model point-support sensitivity check",
            "family": "noAE",
            "q": 0.0,
            "quantizer": "UINT8",
            "samples": len(frame_ids),
            "episode_counts": dict(sorted(episodes.items())),
            "selection": "deterministic evenly spaced within each validation episode",
            "training": False,
            "carla_or_oai_launched": False,
        },
        "bindings": {
            "phase8b": binding,
            "dataset_manifest": {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
            },
            "sample_ids_sha256": hashlib.sha256(
                ("\n".join(frame_ids) + "\n").encode()
            ).hexdigest(),
        },
        "qualification": qualification,
        "supports": compact,
        "delta_current_100ms_minus_rolling_200ms": deltas,
        "interpretation": interpretation(baseline, current),
        "decision_rule": (
            "This bounded check does not change production. Retain 200 ms unless "
            "100 ms is consistently favorable and then confirm with native tracker "
            "replay/fine-tuning and live validation."
        ),
        "limitations": [
            "512-frame deterministic validation subset, not the full 3345-frame split",
            "frozen model was trained with 200-ms radar support",
            "current-only raster retains stationary_age_s produced by the historical persistent tracker",
            "aggregate sensitivity result has no run-to-run training variance",
        ],
        "working_predictions_removed": True,
        "wall_seconds": time.time() - started,
    }
    atomic_json(args.output / "quick_radar_support_ablation.json", document)
    atomic_text(
        args.output / TERMINAL,
        f"{TERMINAL} {sha256_file(args.output / 'quick_radar_support_ablation.json')}\n",
    )
    shutil.rmtree(work)
    print(json.dumps(document, indent=2, sort_keys=True), flush=True)
    print(TERMINAL, flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--reconstruction-checks", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
