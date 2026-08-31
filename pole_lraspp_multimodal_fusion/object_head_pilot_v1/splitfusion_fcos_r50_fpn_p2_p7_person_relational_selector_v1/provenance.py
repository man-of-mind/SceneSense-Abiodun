from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
LOCKED_CONFIG_PATH = PACKAGE / "locked_config.json"

FROZEN_CHECKPOINT_SHA256 = "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f"
ROI_MANIFEST_SHA256 = "f3263d702fac50010975f6e0616c5debd0832e9642f3ce0b8a41a0752b5d307c"
CONSOLIDATION_MANIFEST_SHA256 = "6e9386a6ee1d87cb19685ae0afb1c54cc6b9406bfae4ccf01d9e804578ddcc4c"
CONSOLIDATION_EVIDENCE_SHA256 = "a1bb8b2b7062abc2d0ef4c5cbc715154c5a4e9f1da64e050547de14c56bdddde"
ROI_CACHE_RELATIVE = Path("experiments/person_roi_verifier_v1/train_cache")
CONSOLIDATION_CACHE_RELATIVE = Path("experiments/person_instance_consolidation_v1/train_cache")
CONSOLIDATION_EVIDENCE_RELATIVE = Path(
    "experiments/person_instance_consolidation_v1/feasibility_result.json"
)
EXPECTED_FRAMES = 16_827
EXPECTED_CANDIDATES = 1_148_929
MAX_CANDIDATES_PER_FRAME = 97
HOLDOUT_EXPERIMENT_IDS = (
    "canonical_v3_03_train_30_30_s503_tm1503",
    "canonical_v3_04_train_50_50_s504_tm1504",
)
LOCKED_PERSON_RULE = {
    "grid_index": 27,
    "semantic_support_threshold": 0.10,
    "group_box_iou_threshold": 0.20,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_config() -> dict[str, Any]:
    config = json.loads(LOCKED_CONFIG_PATH.read_text(encoding="utf-8"))
    architecture = config.get("architecture", {})
    cache = config.get("cache_contract", {})
    consolidation = config.get("locked_consolidation", {})
    training = config.get("training", {})
    runtime = config.get("runtime", {})
    if (config.get("schema") != "splitfusion_fcos_person_relational_selector_locked_config_v1"
            or config.get("base") != {
                "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256, "epoch": 26, "frozen": True,
            }
            or architecture != {
                "dropout": 0.0,
                "feedforward_dimension": 256,
                "hidden_dimension": 128,
                "input_dimension": 1044,
                "layers": 2,
                "normalization": "LayerNorm(1044)",
                "output": "one residual logit per candidate",
                "projection": "Linear(1044,128)",
                "self_attention_heads": 4,
                "zero_initialized_output": True,
            }
            or cache != {
                "candidates": EXPECTED_CANDIDATES,
                "consolidation_cache": str(CONSOLIDATION_CACHE_RELATIVE),
                "consolidation_manifest_sha256": CONSOLIDATION_MANIFEST_SHA256,
                "frames": EXPECTED_FRAMES,
                "maximum_candidates_per_frame": MAX_CANDIDATES_PER_FRAME,
                "roi_cache": str(ROI_CACHE_RELATIVE),
                "roi_manifest_sha256": ROI_MANIFEST_SHA256,
            }
            or consolidation != {
                "evidence": str(CONSOLIDATION_EVIDENCE_RELATIVE),
                "evidence_sha256": CONSOLIDATION_EVIDENCE_SHA256,
                **LOCKED_PERSON_RULE,
            }
            or training.get("epochs") != 5
            or training.get("batch_frames") != 16
            or training.get("negative_per_positive") != 3
            or training.get("learning_rate") != 0.001
            or training.get("seed") != 20260831
            or tuple(training.get("holdout_episodes", ())) != HOLDOUT_EXPERIMENT_IDS
            or runtime.get("canonical_person_threshold") != 0.20
            or runtime.get("consolidation_is_feature_only") is not True
            or runtime.get("vehicle_policy") != "locked_service_candidate_v1"):
        raise RuntimeError("relational-selector locked configuration drift")
    return config


def _candidate_total(manifest: dict[str, Any]) -> int:
    if "person_candidates" in manifest:
        return int(manifest["person_candidates"])
    counts = manifest.get("partition_counts", {})
    return sum(int(value.get("person_candidates", -1)) for value in counts.values())


def _validate_roi_manifest(manifest: dict[str, Any]) -> None:
    split = manifest.get("episode_split", {})
    if (manifest.get("schema") != "splitfusion_fcos_person_roi_cache_v1"
            or manifest.get("split") != "train"
            or int(manifest.get("pass_count", -1)) != 1
            or int(manifest.get("frames", -1)) != EXPECTED_FRAMES
            or _candidate_total(manifest) != EXPECTED_CANDIDATES
            or int(manifest.get("roi_descriptor_dim", -1)) != 1024
            or manifest.get("roi_descriptor_dtype") != "float16"
            or int(manifest.get("scalar_feature_dim", -1)) != 10
            or int(manifest.get("feature_dim", -1)) != 1034
            or manifest.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or tuple(split.get("holdout", ())) != HOLDOUT_EXPERIMENT_IDS
            or len(split.get("fit", ())) != 8
            or manifest.get("validation_or_test_accessed") is not False
            or sum(int(shard.get("person_candidates", -1)) for shard in manifest.get("shards", ()))
            != EXPECTED_CANDIDATES):
        raise RuntimeError("locked ROI-cache manifest contract drift")


def _validate_consolidation_manifest(manifest: dict[str, Any]) -> None:
    split = manifest.get("episode_split", {})
    shards = manifest.get("shards", ())
    if (manifest.get("schema") != "splitfusion_fcos_person_instance_consolidation_cache_v1"
            or manifest.get("split") != "train"
            or int(manifest.get("pass_count", -1)) != 1
            or int(manifest.get("frames", -1)) != EXPECTED_FRAMES
            or _candidate_total(manifest) != EXPECTED_CANDIDATES
            or manifest.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or tuple(split.get("holdout", ())) != HOLDOUT_EXPERIMENT_IDS
            or len(split.get("fit", ())) != 8
            or manifest.get("validation_or_test_accessed") is not False
            or sum(int(shard.get("frames", -1)) for shard in shards) != EXPECTED_FRAMES
            or sum(int(shard.get("person_candidates", -1)) for shard in shards) != EXPECTED_CANDIDATES):
        raise RuntimeError("locked consolidation-cache manifest contract drift")


def _validate_evidence(evidence: dict[str, Any]) -> None:
    selected = evidence.get("selected_fit", {})
    holdout = evidence.get("holdout", {})
    if (evidence.get("schema") != "splitfusion_fcos_person_instance_consolidation_result_v1"
            or evidence.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or evidence.get("status") != "holdout_feasible"
            or evidence.get("validation_or_test_accessed") is not False
            or any(selected.get(name) != value for name, value in LOCKED_PERSON_RULE.items())
            or any(holdout.get(name) != value for name, value in LOCKED_PERSON_RULE.items())):
        raise RuntimeError("locked consolidation evidence contract drift")


@dataclass(frozen=True)
class LockedCaches:
    roi_cache: Path
    consolidation_cache: Path
    roi_manifest: dict[str, Any]
    consolidation_manifest: dict[str, Any]
    consolidation_evidence: dict[str, Any]


def load_locked_caches(
    roi_cache: Path | None = None,
    consolidation_cache: Path | None = None,
    evidence_path: Path | None = None,
) -> LockedCaches:
    """Validate locked metadata and hashes without opening a cache shard."""
    load_locked_config()
    roi = (ROOT / ROI_CACHE_RELATIVE if roi_cache is None else Path(roi_cache)).resolve(strict=True)
    consolidation = (
        ROOT / CONSOLIDATION_CACHE_RELATIVE if consolidation_cache is None
        else Path(consolidation_cache)
    ).resolve(strict=True)
    evidence = (
        ROOT / CONSOLIDATION_EVIDENCE_RELATIVE if evidence_path is None else Path(evidence_path)
    ).resolve(strict=True)
    roi_manifest_path = roi / "cache_manifest.json"
    consolidation_manifest_path = consolidation / "cache_manifest.json"
    if sha256(roi_manifest_path) != ROI_MANIFEST_SHA256:
        raise RuntimeError("ROI-cache manifest SHA-256 mismatch")
    if sha256(consolidation_manifest_path) != CONSOLIDATION_MANIFEST_SHA256:
        raise RuntimeError("consolidation-cache manifest SHA-256 mismatch")
    if sha256(evidence) != CONSOLIDATION_EVIDENCE_SHA256:
        raise RuntimeError("consolidation-evidence SHA-256 mismatch")
    roi_manifest = json.loads(roi_manifest_path.read_text(encoding="utf-8"))
    consolidation_manifest = json.loads(consolidation_manifest_path.read_text(encoding="utf-8"))
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    _validate_roi_manifest(roi_manifest)
    _validate_consolidation_manifest(consolidation_manifest)
    _validate_evidence(evidence_payload)
    if (roi_manifest["episode_split"] != consolidation_manifest["episode_split"]
            or len(roi_manifest["shards"]) != len(consolidation_manifest["shards"])):
        raise RuntimeError("locked caches disagree on episode split or shard count")
    for roi_shard, consolidation_shard in zip(
        roi_manifest["shards"], consolidation_manifest["shards"], strict=True,
    ):
        if int(roi_shard["person_candidates"]) != int(consolidation_shard["person_candidates"]):
            raise RuntimeError("locked cache shard candidate counts disagree")
    return LockedCaches(roi, consolidation, roi_manifest, consolidation_manifest, evidence_payload)
