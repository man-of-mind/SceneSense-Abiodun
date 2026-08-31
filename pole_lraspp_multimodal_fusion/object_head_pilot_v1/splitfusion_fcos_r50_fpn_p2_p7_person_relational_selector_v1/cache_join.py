from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    consolidate_person_candidates,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.verifier import (
    ROI_DESCRIPTOR_DIM,
    SCALAR_FEATURE_NAMES,
    fp16_round_trip_roi_descriptors,
)

from .provenance import (
    EXPECTED_CANDIDATES,
    EXPECTED_FRAMES,
    LOCKED_PERSON_RULE,
    LockedCaches,
)
from .selector import CACHED_FEATURE_DIM, INPUT_DIM, MAX_CANDIDATES_PER_FRAME

CONTENT_HEIGHT = 432.0
CONTENT_WIDTH = 768.0
WORLD_NORMALIZATION_METRES = 40.0
PERSON_INTERNAL_CLASS = 1


@dataclass(frozen=True)
class JoinedFrame:
    sample_id: str
    experiment_id: str
    partition: int
    features: torch.Tensor
    base_scores: torch.Tensor
    labels: torch.Tensor
    candidate_identities: torch.Tensor
    original_indices: torch.Tensor
    eligible_positive_count: int

    @property
    def candidate_count(self) -> int:
        return int(self.base_scores.numel())


def _require_vector(value: Any, count: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (count,):
        raise RuntimeError(f"{name} is not an aligned candidate vector")
    return value


def build_relational_features(
    cached_features: torch.Tensor,
    boxes: torch.Tensor,
    world_xy: torch.Tensor,
    component_ids: torch.Tensor,
    semantic_support: torch.Tensor,
    base_scores: torch.Tensor,
    original_indices: torch.Tensor,
) -> torch.Tensor:
    """Build only deployment-available features for one complete person set."""
    count = int(base_scores.numel())
    if (cached_features.shape != (count, CACHED_FEATURE_DIM)
            or boxes.shape != (count, 4)
            or world_xy.shape != (count, 2)
            or component_ids.shape != (count,)
            or semantic_support.shape != (count,)
            or original_indices.shape != (count,)):
        raise RuntimeError("relational feature input alignment drift")
    if count > MAX_CANDIDATES_PER_FRAME:
        raise RuntimeError("frame exceeds locked 97-candidate maximum; truncation is prohibited")
    device = cached_features.device
    cached = cached_features.float()
    boxes_fp32 = boxes.to(device=device, dtype=torch.float32)
    world_fp32 = world_xy.to(device=device, dtype=torch.float32)
    components = component_ids.to(device=device, dtype=torch.long)
    support = semantic_support.to(device=device, dtype=torch.float32)
    scores = base_scores.to(device=device, dtype=torch.float32)
    original = original_indices.detach().long().cpu()
    if not bool(torch.isfinite(cached).all() and torch.isfinite(boxes_fp32).all()
                and torch.isfinite(world_fp32).all() and torch.isfinite(support).all()
                and torch.isfinite(scores).all()):
        raise FloatingPointError("non-finite relational feature input")

    box_scale = torch.tensor(
        [CONTENT_WIDTH, CONTENT_HEIGHT, CONTENT_WIDTH, CONTENT_HEIGHT],
        dtype=torch.float32,
        device=device,
    )
    normalized_boxes = boxes_fp32 / box_scale
    centered_world = (world_fp32 - world_fp32.mean(dim=0, keepdim=True)) / WORLD_NORMALIZATION_METRES
    valid_component = components.ge(0)
    occupancy = torch.zeros(count, dtype=torch.float32, device=device)
    if count:
        for component in torch.unique(components[valid_component]).tolist():
            member = components.eq(int(component))
            occupancy[member] = float(member.sum()) / float(count)

    retained = consolidate_person_candidates(
        scores=scores.detach().cpu(),
        boxes=boxes_fp32.detach().cpu(),
        world_xy=world_fp32.detach().cpu(),
        component_ids=components.detach().cpu(),
        semantic_support=support.detach().cpu(),
        original_indices=original,
        semantic_support_threshold=LOCKED_PERSON_RULE["semantic_support_threshold"],
        group_box_iou_threshold=LOCKED_PERSON_RULE["group_box_iou_threshold"],
    )
    retained_feature = torch.zeros(count, dtype=torch.float32, device=device)
    if retained.numel():
        retained_feature[retained.to(device)] = 1.0
    relational = torch.cat((
        normalized_boxes,
        centered_world,
        support[:, None],
        valid_component.float()[:, None],
        occupancy[:, None],
        retained_feature[:, None],
    ), dim=1)
    features = torch.cat((cached, relational), dim=1)
    if features.shape != (count, INPUT_DIM) or not bool(torch.isfinite(features).all()):
        raise RuntimeError("constructed relational feature shape or finiteness drift")
    return features


def join_shard_payloads(
    roi_payload: Mapping[str, Any],
    consolidation_payload: Mapping[str, Any],
    expected_partitions: Mapping[str, int],
    *,
    shard_name: str,
) -> list[JoinedFrame]:
    """Join one synthetic or real shard pair, failing on any alignment mismatch."""
    frames = consolidation_payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"{shard_name}: consolidation shard has no frame list")
    labels = roi_payload.get("labels")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise RuntimeError(f"{shard_name}: ROI labels are missing or malformed")
    total = int(labels.numel())
    roi = roi_payload.get("roi_descriptors")
    scalars = roi_payload.get("scalar_features")
    base_scores = roi_payload.get("base_scores")
    identities = roi_payload.get("candidate_identities")
    partitions = roi_payload.get("partitions")
    sample_ids = roi_payload.get("sample_ids")
    experiment_ids = roi_payload.get("experiment_ids")
    if (not isinstance(roi, torch.Tensor) or roi.shape != (total, ROI_DESCRIPTOR_DIM)
            or roi.dtype != torch.float16
            or not isinstance(scalars, torch.Tensor)
            or scalars.shape != (total, len(SCALAR_FEATURE_NAMES))
            or scalars.dtype != torch.float32
            or not isinstance(base_scores, torch.Tensor) or base_scores.shape != (total,)
            or base_scores.dtype != torch.float32
            or not isinstance(identities, torch.Tensor) or identities.shape != (total, 4)
            or not isinstance(partitions, torch.Tensor) or partitions.shape != (total,)
            or not isinstance(sample_ids, list) or len(sample_ids) != total
            or not isinstance(experiment_ids, list) or len(experiment_ids) != total):
        raise RuntimeError(f"{shard_name}: ROI shard field contract drift")

    joined: list[JoinedFrame] = []
    offset = 0
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise RuntimeError(f"{shard_name}: malformed consolidation frame {frame_index}")
        frame_scores = frame.get("scores")
        if not isinstance(frame_scores, torch.Tensor) or frame_scores.ndim != 1:
            raise RuntimeError(f"{shard_name}: malformed frame score vector")
        count = int(frame_scores.numel())
        stop = offset + count
        if count > MAX_CANDIDATES_PER_FRAME or stop > total:
            raise RuntimeError(f"{shard_name}: frame candidate count exceeds its ROI slice")
        sample_id = str(frame.get("sample_id"))
        experiment_id = str(frame.get("experiment_id"))
        if experiment_id not in expected_partitions:
            raise RuntimeError(f"{shard_name}: unregistered experiment identity")
        expected_partition = int(expected_partitions[experiment_id])
        if (sample_ids[offset:stop] != [sample_id] * count
                or experiment_ids[offset:stop] != [experiment_id] * count):
            raise RuntimeError(f"{shard_name}: frame sample/experiment alignment mismatch")
        frame_partition = partitions[offset:stop].long()
        if not torch.equal(frame_partition, torch.full((count,), expected_partition, dtype=torch.long)):
            raise RuntimeError(f"{shard_name}: frame partition alignment mismatch")
        sliced_scores = base_scores[offset:stop]
        if (frame_scores.dtype != torch.float32 or not torch.equal(sliced_scores, frame_scores)
                or not torch.equal(scalars[offset:stop, 0], sliced_scores)):
            raise RuntimeError(f"{shard_name}: candidate count/order/base-score mismatch")

        frame_identities = identities[offset:stop].long()
        original_indices = frame.get("original_indices")
        boxes = frame.get("boxes")
        world_xy = frame.get("world_xy")
        component_ids = frame.get("component_ids")
        semantic_support = frame.get("semantic_support")
        _require_vector(component_ids, count, "component_ids")
        _require_vector(semantic_support, count, "semantic_support")
        if (not isinstance(original_indices, torch.Tensor) or original_indices.shape != (count,)
                or not isinstance(boxes, torch.Tensor) or boxes.shape != (count, 4)
                or not isinstance(world_xy, torch.Tensor) or world_xy.shape != (count, 2)
                or (count > 1 and not bool((original_indices[1:] > original_indices[:-1]).all()))
                or bool((frame_identities[:, 3] != PERSON_INTERNAL_CLASS).any())
                or len({tuple(value) for value in frame_identities.tolist()}) != count):
            raise RuntimeError(f"{shard_name}: candidate identity/order or geometry mismatch")

        rounded_roi = fp16_round_trip_roi_descriptors(roi[offset:stop])
        cached_features = torch.cat((rounded_roi, scalars[offset:stop].float()), dim=1)
        features = build_relational_features(
            cached_features, boxes, world_xy, component_ids, semantic_support,
            sliced_scores, original_indices,
        )
        gt_world_xy = frame.get("gt_world_xy")
        if not isinstance(gt_world_xy, torch.Tensor) or gt_world_xy.ndim != 2 or gt_world_xy.shape[1] != 2:
            raise RuntimeError(f"{shard_name}: eligible-person count source is malformed")
        joined.append(JoinedFrame(
            sample_id=sample_id,
            experiment_id=experiment_id,
            partition=expected_partition,
            features=features,
            base_scores=sliced_scores.float(),
            labels=labels[offset:stop].long(),
            candidate_identities=frame_identities,
            original_indices=original_indices.long(),
            eligible_positive_count=int(gt_world_xy.shape[0]),
        ))
        offset = stop
    if offset != total:
        raise RuntimeError(f"{shard_name}: unconsumed ROI candidates after frame join")
    return joined


def iter_joined_frames(caches: LockedCaches) -> Iterator[JoinedFrame]:
    """Stream every verified frame from paired cache shards without model execution."""
    split = caches.roi_manifest["episode_split"]
    expected_partitions = {
        **{str(experiment_id): 0 for experiment_id in split["fit"]},
        **{str(experiment_id): 1 for experiment_id in split["holdout"]},
    }
    frame_total = candidate_total = 0
    for shard_index, (roi_shard, consolidation_shard) in enumerate(zip(
        caches.roi_manifest["shards"], caches.consolidation_manifest["shards"], strict=True,
    )):
        roi_path = caches.roi_cache / str(roi_shard["path"])
        consolidation_path = caches.consolidation_cache / str(consolidation_shard["path"])
        if roi_path.name != consolidation_path.name:
            raise RuntimeError("paired cache shard names differ")
        roi_payload = torch.load(roi_path, map_location="cpu", weights_only=True)
        consolidation_payload = torch.load(consolidation_path, map_location="cpu", weights_only=True)
        joined = join_shard_payloads(
            roi_payload, consolidation_payload, expected_partitions,
            shard_name=f"shard_{shard_index:05d}",
        )
        shard_candidates = sum(frame.candidate_count for frame in joined)
        if (len(joined) != int(consolidation_shard["frames"])
                or shard_candidates != int(roi_shard["person_candidates"])
                or shard_candidates != int(consolidation_shard["person_candidates"])):
            raise RuntimeError("joined shard counts disagree with locked manifests")
        for frame in joined:
            frame_total += 1
            candidate_total += frame.candidate_count
            yield frame
    if frame_total != EXPECTED_FRAMES or candidate_total != EXPECTED_CANDIDATES:
        raise RuntimeError("joined cache totals disagree with the locked corpus")


def pad_frames(
    frames: Sequence[JoinedFrame], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not frames:
        raise ValueError("cannot pad an empty frame batch")
    maximum = max(frame.candidate_count for frame in frames)
    if maximum == 0:
        maximum = 1
    if maximum > MAX_CANDIDATES_PER_FRAME:
        raise RuntimeError("frame batch would require prohibited candidate truncation")
    features = torch.zeros((len(frames), maximum, INPUT_DIM), dtype=torch.float32, device=device)
    base_scores = torch.zeros((len(frames), maximum), dtype=torch.float32, device=device)
    labels = torch.full((len(frames), maximum), -1, dtype=torch.long, device=device)
    padding = torch.ones((len(frames), maximum), dtype=torch.bool, device=device)
    for index, frame in enumerate(frames):
        count = frame.candidate_count
        if count:
            features[index, :count] = frame.features.to(device)
            base_scores[index, :count] = frame.base_scores.to(device)
            labels[index, :count] = frame.labels.to(device)
            padding[index, :count] = False
    return features, base_scores, labels, padding
