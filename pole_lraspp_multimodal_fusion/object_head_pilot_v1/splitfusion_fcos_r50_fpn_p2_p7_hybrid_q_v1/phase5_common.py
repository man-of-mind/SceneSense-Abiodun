"""Shared Phase-5 plumbing: input binding, teacher store and train-holdout scoring.

Everything here is either a verification of a frozen Phase-1..4 artifact or a thin
adapter that lets the *existing* frozen scorers run over the reserved train-holdout
episodes. No matching rule, threshold, postprocessing step or metric definition is
introduced or changed by this module.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from . import contract, guards
from .gpu_qualification import package_source_hashes, sha256_file
from .teacher_cache import SplitPartition, load_shard


# ---------------------------------------------------------------------------
# Frozen scorers, loaded by path under private module names
# ---------------------------------------------------------------------------

_SCORER_PATHS = {
    "hybrid_q_frozen_score_contract_v1": (
        "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
        "route_b_v3_1_clean_base_v1/score_contract_v1.py"
    ),
    "hybrid_q_frozen_audit_v1": (
        "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
        "route_b_v3_1_targeted_refinement_v1/audit_v1.py"
    ),
}


def _load_by_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load frozen scorer {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FrozenScorers:
    """The registered v3.1 detection and segmentation scorers, loaded verbatim.

    `score_arm` and `score_segmentation` are the same functions the frozen
    validation evaluator calls; only the split whose contract directory they read
    differs, and that is handled by a path alias rather than by editing them.
    """

    segmentation: Any
    detection: Any
    sha256: dict[str, str]

    @property
    def score_segmentation(self) -> Any:
        return self.segmentation.score_segmentation

    @property
    def score_arm(self) -> Any:
        return self.detection.score_arm

    @property
    def load_gt(self) -> Any:
        return self.detection.load_gt

    @property
    def load_predictions(self) -> Any:
        return self.detection.load_predictions


def load_frozen_scorers() -> FrozenScorers:
    root = contract.repository_root()
    modules: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name, relative in _SCORER_PATHS.items():
        path = (root / relative).resolve(strict=True)
        hashes[relative] = sha256_file(path)
        modules[name] = _load_by_path(name, path)
    detection = modules["hybrid_q_frozen_audit_v1"]
    segmentation = modules["hybrid_q_frozen_score_contract_v1"]
    if detection.MATCH_RADIUS_M != 3.0 or detection.CLASSES != ("vehicle", "person"):
        raise guards.HybridQConfigError("frozen detection scorer semantics drift")
    if detection.MODEL_SIZE != (768, 432) or detection.SOURCE_SIZE != (1280, 720):
        raise guards.HybridQConfigError("frozen detection scorer geometry drift")
    return FrozenScorers(segmentation=segmentation, detection=detection, sha256=hashes)


# ---------------------------------------------------------------------------
# Train-split contract alias
# ---------------------------------------------------------------------------


def build_contract_alias(dataset_root: Path, alias_root: Path) -> Path:
    """Expose the *train* contract directory where the frozen scorers read `val`.

    The frozen scorers hardcode `contracts/<contract>/val/...` because they were
    written for the fixed validation pass. Phase 5 must score the reserved
    train-holdout episodes with byte-identical scoring code, so the split is
    redirected by a read-only symlink instead of by editing a frozen scorer. The
    validation contract directory is never linked and never read.
    """
    alias_root = Path(alias_root)
    for name in ("v010", "v025"):
        source = dataset_root / f"contracts/{name}/train"
        if not source.is_dir():
            raise guards.HybridQConfigError(f"missing train contract directory {source}")
        target = alias_root / f"contracts/{name}"
        target.mkdir(parents=True, exist_ok=True)
        link = target / "val"
        if link.is_symlink() or link.exists():
            raise guards.HybridQConfigError(f"contract alias already exists: {link}")
        link.symlink_to(source.resolve(strict=True), target_is_directory=True)
    resolved = (alias_root / "contracts/v010/val").resolve(strict=True)
    if resolved != (dataset_root / "contracts/v010/train").resolve(strict=True):
        raise guards.HybridQConfigError("contract alias does not resolve to the train split")
    return alias_root


# ---------------------------------------------------------------------------
# Input binding
# ---------------------------------------------------------------------------


def teacher_cache_root() -> Path:
    return contract.repository_root() / contract.TEACHER_CACHE_RELPATH


def bind_inputs() -> dict[str, Any]:
    """Verify every Phase-5 input by exact hash and fail closed on any drift."""
    root = contract.repository_root()
    cache_root = teacher_cache_root()

    lock_path = contract.perception_lock_path()
    lock_hash = sha256_file(lock_path)
    if lock_hash != contract.PERCEPTION_LOCK_SHA256:
        raise guards.HybridQConfigError("perception forward lock sha256 drift")
    lock = contract.load_perception_lock()

    checkpoint_path = (root / lock["base_checkpoint"]["path"]).resolve(strict=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != contract.FROZEN_CHECKPOINT_SHA256:
        raise guards.HybridQConfigError("frozen checkpoint sha256 drift")

    locked = contract.load_locked_config()
    locked_hash = sha256_file(contract.locked_config_path())
    if locked_hash != contract.LOCKED_CONFIG_SHA256:
        raise guards.HybridQConfigError("hybrid-q locked configuration sha256 drift")

    manifest_path = cache_root / "teacher_cache_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != contract.TEACHER_CACHE_MANIFEST_SHA256:
        raise guards.HybridQConfigError("Phase-4 teacher-cache manifest sha256 drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["terminal"] != "HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE":
        raise guards.HybridQConfigError("teacher-cache manifest is not a complete cache")
    if manifest["locked_config_sha256"] != locked_hash:
        raise guards.HybridQConfigError(
            "the teacher cache was generated under a different locked configuration"
        )
    if manifest["perception_binding"]["checkpoint_sha256"] != checkpoint_hash:
        raise guards.HybridQConfigError("teacher cache checkpoint binding drift")
    if int(manifest["split"]["validation_or_test_frames"]) != 0:
        raise guards.HybridQConfigError("teacher cache reports validation/test frames")

    entries = list(manifest["shards"]["entries"])
    if len(entries) != contract.TEACHER_CACHE_SHARD_COUNT:
        raise guards.HybridQConfigError(
            f"{len(entries)} teacher shards != {contract.TEACHER_CACHE_SHARD_COUNT}"
        )
    shard_hashes: dict[str, str] = {}
    for entry in entries:
        path = cache_root / entry["path"]
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise guards.HybridQPayloadError(f"{entry['path']} sha256 drift")
        if path.stat().st_size != int(entry["bytes"]):
            raise guards.HybridQPayloadError(f"{entry['path']} byte size drift")
        shard_hashes[entry["path"]] = observed

    medians_path = cache_root / "fit_reference_medians.json"
    medians_hash = sha256_file(medians_path)
    if medians_hash != contract.FIT_REFERENCE_MEDIANS_SHA256:
        raise guards.HybridQConfigError("Phase-4 fit-reference medians sha256 drift")
    medians_doc = json.loads(medians_path.read_text(encoding="utf-8"))
    if medians_doc["source"] != contract.REFERENCE_MEDIAN_SOURCE:
        raise guards.HybridQConfigError("reference medians are not fit-train derived")
    if medians_doc["holdout_contribution"] != "none":
        raise guards.HybridQConfigError("reference medians received a holdout contribution")
    if int(medians_doc["fit_batches"]) != 847:
        raise guards.HybridQConfigError("reference medians fit-batch count drift")
    medians = {name: float(value) for name, value in medians_doc["medians"].items()}
    if set(medians) != set(contract.TEACHER_GROUPS):
        raise guards.HybridQConfigError("reference medians are incomplete")
    for name, value in contract.FROZEN_FIT_REFERENCE_MEDIANS.items():
        if medians[name] != float(value):
            raise guards.HybridQConfigError(
                f"reference median for {name} is {medians[name]!r}, contract requires {value!r}"
            )

    return {
        "perception_forward_lock": {
            "path": contract.PERCEPTION_LOCK_RELPATH,
            "sha256": lock_hash,
        },
        "frozen_checkpoint": {
            "path": str(checkpoint_path.relative_to(root)),
            "sha256": checkpoint_hash,
            "epoch": int(lock["base_checkpoint"]["epoch"]),
        },
        "hybrid_q_locked_config": {
            "path": str(contract.locked_config_path().relative_to(root)),
            "sha256": locked_hash,
            "schema": locked["schema"],
        },
        "teacher_cache_manifest": {
            "path": str(manifest_path.relative_to(root)),
            "sha256": manifest_hash,
        },
        "teacher_cache_shards": {
            "count": len(entries),
            "verified": len(shard_hashes),
            "total_bytes": int(manifest["shards"]["total_bytes"]),
            "sha256": shard_hashes,
        },
        "fit_reference_medians": {
            "path": str(medians_path.relative_to(root)),
            "sha256": medians_hash,
            "medians": medians,
        },
        "hybrid_q_source_sha256": package_source_hashes(),
        "phase4_recorded_source_sha256": dict(manifest["hybrid_q_source_sha256"]),
    }


def source_delta(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Which hybrid-q source files changed since the teacher cache was written.

    Phase 5 adds runner modules and appends Phase-5 constants to `contract.py`; every
    module that defines the cached teacher semantics must be bit-identical.
    """
    before = dict(binding["phase4_recorded_source_sha256"])
    after = dict(binding["hybrid_q_source_sha256"])
    frozen_semantics = (
        "ranker.py", "selection.py", "codec.py", "guards.py", "training.py",
        "teacher_cache.py", "gpu_qualification.py", "locked_config.json", "__init__.py",
    )
    changed = sorted(name for name in before if after.get(name) != before[name])
    added = sorted(set(after) - set(before))
    violated = [name for name in frozen_semantics if after.get(name) != before.get(name)]
    if violated:
        raise guards.HybridQConfigError(
            f"frozen hybrid-q semantics module(s) changed since Phase 4: {violated}"
        )
    return {
        "changed_since_phase4": changed,
        "added_since_phase4": added,
        "frozen_semantics_modules_unchanged": frozen_semantics,
    }


# ---------------------------------------------------------------------------
# Preloaded teacher maps
# ---------------------------------------------------------------------------


class TeacherStore:
    """All 66 cached teacher maps held once in host memory, keyed by sample ID.

    Holdout maps are loaded so the cache is verified end to end, but `fit_map`
    refuses to return one: no holdout supervision may reach an optimizer batch.
    """

    def __init__(self, maps: torch.Tensor, index: Mapping[str, int],
                 splits: Mapping[str, str]) -> None:
        self.maps = maps
        self.index = dict(index)
        self.splits = dict(splits)
        self.fit_ids = frozenset(k for k, v in self.splits.items() if v == "fit")
        self.holdout_ids = frozenset(k for k, v in self.splits.items() if v == "holdout")

    @property
    def bytes(self) -> int:
        return int(self.maps.numel()) * self.maps.element_size()

    def fit_map(self, sample_id: str) -> torch.Tensor:
        label = self.splits.get(sample_id)
        if label is None:
            raise guards.HybridQConfigError(f"{sample_id} is not in the teacher cache")
        if label != "fit":
            raise guards.HybridQOwnershipError(
                f"{sample_id} is a reserved holdout frame and must not enter training"
            )
        return self.maps[self.index[sample_id]]

    def fit_batch(self, sample_ids: Sequence[str], device: torch.device) -> torch.Tensor:
        rows = torch.stack([self.fit_map(str(name)) for name in sample_ids])
        return rows.to(device, non_blocking=True)


def load_teacher_store(binding: Mapping[str, Any], partition: SplitPartition) -> TeacherStore:
    """Load every verified shard once, then reconcile identity against the split."""
    cache_root = teacher_cache_root()
    manifest = json.loads(
        (cache_root / "teacher_cache_manifest.json").read_text(encoding="utf-8")
    )
    blocks: list[torch.Tensor] = []
    index: dict[str, int] = {}
    splits: dict[str, str] = {}
    cursor = 0
    for entry in manifest["shards"]["entries"]:
        payload = load_shard(cache_root / entry["path"])
        maps = payload["importance"]
        if maps.dtype != torch.float32:
            raise guards.HybridQPayloadError(f"{entry['path']} maps are not FP32")
        if tuple(maps.shape[1:]) != contract.SPLIT_SPATIAL_SHAPE:
            raise guards.HybridQPayloadError(f"{entry['path']} map shape drift")
        for offset, (sample_id, label) in enumerate(
            zip(payload["sample_ids"], payload["splits"])
        ):
            key = str(sample_id)
            if key in index:
                raise guards.HybridQConfigError(f"duplicate cached frame {key}")
            index[key] = cursor + offset
            splits[key] = str(label)
        blocks.append(maps.contiguous())
        cursor += int(maps.shape[0])

    if cursor != contract.TRAIN_TOTAL_FRAMES or len(index) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError(
            f"teacher store holds {cursor} frames / {len(index)} unique ids"
        )
    fit_ids = set(partition.fit_sample_ids)
    holdout_ids = set(partition.holdout_sample_ids)
    cached_fit = {k for k, v in splits.items() if v == "fit"}
    cached_holdout = {k for k, v in splits.items() if v == "holdout"}
    if cached_fit != fit_ids or cached_holdout != holdout_ids:
        raise guards.HybridQConfigError("teacher store split labels disagree with the partition")
    if len(cached_fit) != contract.TRAIN_FIT_FRAMES:
        raise guards.HybridQConfigError("teacher store fit count drift")
    if len(cached_holdout) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError("teacher store holdout count drift")
    if cached_fit & cached_holdout:
        raise guards.HybridQConfigError("teacher store fit/holdout overlap")

    maps = torch.cat(blocks, dim=0)
    del blocks
    if not torch.isfinite(maps).all():
        raise guards.HybridQNumericalError("a cached teacher map is non-finite")
    return TeacherStore(maps, index, splits)


def require_no_validation_or_test(dataset: Any, cached_ids: Iterable[str]) -> dict[str, Any]:
    """Registered-metadata check that the cache touches no validation or test frame."""
    root = contract.repository_root()
    config = json.loads(
        (root / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
         "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json").read_text(encoding="utf-8")
    )
    dataset_root = (root / config["dataset_root"]).resolve(strict=True)
    with (dataset_root / "dataset/manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_split[row["split"]].add(row["sample_id"])
    cached = set(str(value) for value in cached_ids)
    reserved = {name: ids for name, ids in by_split.items() if name != "train"}
    intersections = {name: len(cached & ids) for name, ids in reserved.items()}
    if any(intersections.values()):
        raise guards.HybridQConfigError(
            f"the teacher cache intersects a reserved split: {intersections}"
        )
    if cached != by_split["train"]:
        raise guards.HybridQConfigError("the cache is not exactly the registered train split")
    if len(cached) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError("cached train frame count drift")
    return {
        "manifest_rows": len(rows),
        "split_frame_counts": {name: len(ids) for name, ids in sorted(by_split.items())},
        "reserved_split_intersections": intersections,
        "locked_test_split_present": "test" in by_split,
        "cached_equals_registered_train_set": True,
        "duplicate_cached_frame_ids": 0,
        "missing_cached_frame_ids": 0,
    }


# ---------------------------------------------------------------------------
# Reserved train-holdout person ground truth (AVO)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutPersonTruth:
    """Frozen p025 train-holdout person GT: AVO-qualified rows plus structural rows."""

    qualified: dict[str, list[dict[str, Any]]]
    structural: dict[str, list[dict[str, Any]]]
    diagnostics: dict[str, Any]


def load_holdout_person_truth(frame_ids: Sequence[str]) -> HoldoutPersonTruth:
    """Reuse the frozen p025 AVO table; derive structural rows as its raw complement.

    The p025 qualification partitioned every raw holdout person actor-frame into
    canonically qualified rows (the published AVO table) and structurally ignored
    rows. The complement is therefore exact, and its cardinality is checked against
    the published p025 counts before use.
    """
    root = contract.repository_root()
    table_path = root / contract.HOLDOUT_AVO_TABLE_RELPATH
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    if len(table) != contract.HOLDOUT_AVO_QUALIFIED_ACTOR_FRAMES:
        raise guards.HybridQConfigError(
            f"AVO table holds {len(table)} rows != "
            f"{contract.HOLDOUT_AVO_QUALIFIED_ACTOR_FRAMES}"
        )
    frames = set(str(value) for value in frame_ids)
    qualified: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualified_keys: set[tuple[str, str, str]] = set()
    observable = 0
    for row in table:
        sample_id = str(row["sample_id"])
        if sample_id not in frames:
            raise guards.HybridQConfigError(f"AVO row outside the holdout frames: {sample_id}")
        key = (str(row["episode_id"]), sample_id, str(row["gt_actor_id"]))
        if key in qualified_keys:
            raise guards.HybridQConfigError(f"duplicate AVO row: {key}")
        qualified_keys.add(key)
        avo = float(row["actor_volume_observability"])
        if not (math.isfinite(avo) and 0.0 <= avo <= 1.0):
            raise guards.HybridQConfigError(f"invalid AVO value: {key}/{avo}")
        if avo >= contract.PERSON_AVO_THRESHOLD:
            observable += 1
        qualified[sample_id].append({
            "episode_id": key[0],
            "gt_actor_id": key[2],
            "world_x": float(row["world_x"]),
            "world_y": float(row["world_y"]),
            "distance_m": float(row["distance_m"]),
            "distance_bin": str(row["distance_bin"]),
            "actor_volume_observability": avo,
        })
    if observable != contract.HOLDOUT_AVO_OBSERVABLE_ACTOR_FRAMES:
        raise guards.HybridQConfigError(
            f"{observable} AVO-observable actor-frames != "
            f"{contract.HOLDOUT_AVO_OBSERVABLE_ACTOR_FRAMES}"
        )

    raw_root = root / "data_collection/experiments/route_b_perception_v3"
    structural: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_person_rows = 0
    raw_hashes: dict[str, str] = {}
    for episode in contract.TRAIN_HOLDOUT_EPISODES:
        path = raw_root / episode / "object_boxes.csv"
        raw_hashes[str(path.relative_to(root))] = sha256_file(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = str(row["sample_id"])
                if sample_id not in frames or str(row["label"]) != "person":
                    continue
                raw_person_rows += 1
                key = (str(row["experiment_id"]), sample_id, str(row["gt_actor_id"]))
                if key in qualified_keys:
                    continue
                structural[sample_id].append({
                    "world_x": float(row["object_world_x"]),
                    "world_y": float(row["object_world_y"]),
                })
    structural_rows = sum(len(values) for values in structural.values())
    if raw_person_rows != contract.HOLDOUT_RAW_PERSON_ACTOR_FRAMES:
        raise guards.HybridQConfigError(
            f"{raw_person_rows} raw holdout person actor-frames != "
            f"{contract.HOLDOUT_RAW_PERSON_ACTOR_FRAMES}"
        )
    if structural_rows != contract.HOLDOUT_STRUCTURAL_ACTOR_FRAMES:
        raise guards.HybridQConfigError(
            f"{structural_rows} structural actor-frames != "
            f"{contract.HOLDOUT_STRUCTURAL_ACTOR_FRAMES}"
        )
    if structural_rows + len(table) != raw_person_rows:
        raise guards.HybridQConfigError("AVO table and structural rows do not partition raw GT")

    return HoldoutPersonTruth(
        qualified=dict(qualified),
        structural=dict(structural),
        diagnostics={
            "avo_table_path": contract.HOLDOUT_AVO_TABLE_RELPATH,
            "avo_table_sha256": sha256_file(table_path),
            "raw_object_boxes_sha256": raw_hashes,
            "qualified_actor_frames": len(table),
            "observable_actor_frames": observable,
            "avo_ignored_actor_frames": len(table) - observable,
            "structural_ignored_actor_frames": structural_rows,
            "raw_person_actor_frames": raw_person_rows,
            "distance_bins_observable": dict(sorted(Counter(
                row["distance_bin"] for rows in qualified.values() for row in rows
                if row["actor_volume_observability"] >= contract.PERSON_AVO_THRESHOLD
            ).items())),
            "derivation": (
                "qualified rows are the published p025 AVO table verbatim; structural "
                "rows are its exact complement within the raw holdout person GT"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Person AVO scoring: the frozen p025 view plus a distance-bin breakdown
# ---------------------------------------------------------------------------


P025_PACKAGE = (
    "pole_lraspp_multimodal_fusion.object_head_pilot_v1."
    "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1"
)


def load_p025_qualification() -> Any:
    """The frozen p025 train-holdout scorer module, imported as its own package."""
    qualification = importlib.import_module(f"{P025_PACKAGE}.qualification")
    policy = importlib.import_module(f"{P025_PACKAGE}.policy")
    if qualification.AVO_THRESHOLD != contract.PERSON_AVO_THRESHOLD:
        raise guards.HybridQConfigError("frozen p025 AVO threshold drift")
    if qualification.MATCH_RADIUS_M != 3.0 or qualification.MAX_DISTANCE_M != 40.0:
        raise guards.HybridQConfigError("frozen p025 matching contract drift")
    if policy.PERSON_SCORE_THRESHOLD != contract.PERSON_SERVICE_SCORE_THRESHOLD:
        raise guards.HybridQConfigError("frozen p025 person output threshold drift")
    return qualification


def score_person_avo(
    *,
    frame_ids: Sequence[str],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    truth: HoldoutPersonTruth,
    qualification: Any,
) -> dict[str, Any]:
    """The frozen p025 AVO>=0.65 person view, with per-distance-bin recall added.

    Matching is `qualification.greedy_match` verbatim, in the registered cascade
    order: observable GT, then AVO-ignored GT, then structurally ignored GT. The
    only addition is that the identity of each matched observable GT row is kept,
    so the 20-40 m recall is a strict partition of the same TP/FN accounting and
    cannot disagree with the aggregate.
    """
    greedy_match = qualification.greedy_match
    threshold = contract.PERSON_SERVICE_SCORE_THRESHOLD
    totals = {
        "observable_gt": 0, "avo_ignored_gt": 0, "structural_ignored_gt": 0,
        "tp": 0, "fp": 0, "fn": 0,
        "avo_ignored_predictions": 0, "structural_ignored_predictions": 0,
    }
    errors: list[float] = []
    by_bin: dict[str, dict[str, int]] = defaultdict(lambda: {"observable_gt": 0, "tp": 0, "fn": 0})

    for sample_id in frame_ids:
        qualified = list(truth.qualified.get(sample_id, []))
        eligible = [
            row for row in qualified
            if float(row["actor_volume_observability"]) >= contract.PERSON_AVO_THRESHOLD
        ]
        avo_ignored = [
            row for row in qualified
            if float(row["actor_volume_observability"]) < contract.PERSON_AVO_THRESHOLD
        ]
        structural = list(truth.structural.get(sample_id, []))
        frame_predictions = [
            row for row in predictions.get(sample_id, [])
            if float(row["score"]) >= threshold
        ]

        matched, used_eligible = greedy_match(frame_predictions, eligible)
        used_predictions = set(matched)
        for pred_index, gt_index in matched.items():
            target = eligible[gt_index]
            errors.append(math.hypot(
                float(frame_predictions[pred_index]["world_x"]) - float(target["world_x"]),
                float(frame_predictions[pred_index]["world_y"]) - float(target["world_y"]),
            ))
        remaining = set(range(len(frame_predictions))) - used_predictions
        matched_avo, _ = greedy_match(frame_predictions, avo_ignored, remaining)
        remaining -= set(matched_avo)
        matched_structural, _ = greedy_match(frame_predictions, structural, remaining)
        remaining -= set(matched_structural)

        totals["observable_gt"] += len(eligible)
        totals["avo_ignored_gt"] += len(avo_ignored)
        totals["structural_ignored_gt"] += len(structural)
        totals["tp"] += len(used_eligible)
        totals["fn"] += len(eligible) - len(used_eligible)
        totals["fp"] += len(remaining)
        totals["avo_ignored_predictions"] += len(matched_avo)
        totals["structural_ignored_predictions"] += len(matched_structural)
        for gt_index, target in enumerate(eligible):
            bucket = by_bin[str(target["distance_bin"])]
            bucket["observable_gt"] += 1
            if gt_index in used_eligible:
                bucket["tp"] += 1
            else:
                bucket["fn"] += 1

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    observable = totals["observable_gt"]
    if tp + fn != observable:
        raise guards.HybridQConfigError("person AVO TP+FN denominator failure")
    if len(errors) != tp:
        raise guards.HybridQConfigError("person AVO localization sample count drift")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / observable if observable else 0.0
    long_range = {"observable_gt": 0, "tp": 0, "fn": 0}
    for name in contract.PERSON_LONG_RANGE_BINS:
        for key in long_range:
            long_range[key] += by_bin[name][key]
    if long_range["tp"] + long_range["fn"] != long_range["observable_gt"]:
        raise guards.HybridQConfigError("person 20-40 m denominator failure")
    if sum(bucket["observable_gt"] for bucket in by_bin.values()) != observable:
        raise guards.HybridQConfigError("distance-bin partition does not cover observable GT")

    return {
        "avo_threshold": contract.PERSON_AVO_THRESHOLD,
        "detection_score_threshold": threshold,
        "matching_order": "observable_gt_then_avo_ignored_gt_then_structural_ignored_gt",
        **totals,
        "ignored_predictions": (
            totals["avo_ignored_predictions"] + totals["structural_ignored_predictions"]
        ),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "xy_mae_m": (sum(errors) / len(errors)) if errors else None,
        "distance_bins": {
            name: {
                **by_bin[name],
                "recall": (by_bin[name]["tp"] / by_bin[name]["observable_gt"])
                if by_bin[name]["observable_gt"] else 0.0,
            }
            for name in sorted(by_bin)
        },
        "recall_20_40m": (
            long_range["tp"] / long_range["observable_gt"]
            if long_range["observable_gt"] else 0.0
        ),
        "observable_gt_20_40m": long_range["observable_gt"],
        "tp_20_40m": long_range["tp"],
    }


def cross_check_person_avo(
    *,
    frame_ids: Sequence[str],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    truth: HoldoutPersonTruth,
    qualification: Any,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the frozen p025 `score_view` to agree exactly on every shared field."""
    episode_by_sample: dict[str, str] = {}
    for sample_id in frame_ids:
        rows = truth.qualified.get(sample_id, [])
        episode_by_sample[sample_id] = (
            str(rows[0]["episode_id"]) if rows else contract.TRAIN_HOLDOUT_EPISODES[0]
        )
    for sample_id in frame_ids:
        for row in truth.qualified.get(sample_id, []):
            episode_by_sample[sample_id] = str(row["episode_id"])
    frozen = qualification.score_view(
        frame_ids=list(frame_ids),
        episodes=list(contract.TRAIN_HOLDOUT_EPISODES),
        predictions=predictions,
        qualified_gt=truth.qualified,
        structural_gt=truth.structural,
        episode_by_sample=episode_by_sample,
        detection_threshold=contract.PERSON_SERVICE_SCORE_THRESHOLD,
    )
    reference = frozen["overall"]
    shared = (
        "observable_gt", "avo_ignored_gt", "structural_ignored_gt", "tp", "fp", "fn",
        "avo_ignored_predictions", "structural_ignored_predictions",
        "ignored_predictions", "precision", "recall", "f1", "xy_mae_m",
    )
    mismatched = [name for name in shared if reference[name] != observed[name]]
    if mismatched:
        raise guards.HybridQConfigError(
            f"person AVO scorer disagrees with the frozen p025 view on {mismatched}"
        )
    return {
        "frozen_p025_score_view_agrees": True,
        "compared_fields": list(shared),
        "per_episode": frozen["episodes"],
    }
