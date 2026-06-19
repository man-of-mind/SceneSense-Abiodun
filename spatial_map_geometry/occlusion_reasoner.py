from __future__ import annotations

from typing import List, Sequence

from .association import xy_distance_m
from .geometry import overlap_area, point_in_polygon
from .schemas import LocalSensorMap, OcclusionHypothesis, SpatialObject


def _has_matching_object(
    obj: SpatialObject,
    candidate_map: LocalSensorMap,
    distance_threshold_m: float,
    require_class_match: bool,
) -> bool:
    for candidate in candidate_map.objects:
        if require_class_match and candidate.class_name != obj.class_name:
            continue
        if xy_distance_m(obj, candidate) <= distance_threshold_m:
            return True
    return False


def infer_overlap_disagreements(
    local_maps: Sequence[LocalSensorMap],
    distance_threshold_m: float = 3.0,
    min_overlap_area_m2: float = 10.0,
    require_class_match: bool = True,
) -> List[OcclusionHypothesis]:
    """Infer conservative missing-object hypotheses from overlapping FoVs.

    If an object from map A falls inside map B's FoV and A/B footprints overlap,
    but B has no nearby matching object, we emit a possible occlusion/miss
    hypothesis. This is not proof of occlusion.
    """

    hypotheses: List[OcclusionHypothesis] = []
    for source in local_maps:
        for target in local_maps:
            if source.stream_id == target.stream_id:
                continue
            area = overlap_area(source.fov_polygon, target.fov_polygon)
            if area < min_overlap_area_m2:
                continue
            for obj in source.objects:
                if not point_in_polygon(obj.xy, target.fov_polygon):
                    continue
                if _has_matching_object(obj, target, distance_threshold_m, require_class_match):
                    continue
                confidence = max(0.05, min(0.95, float(obj.confidence) * 0.7))
                hypotheses.append(
                    OcclusionHypothesis(
                        source_stream_id=source.stream_id,
                        missing_from_stream_id=target.stream_id,
                        object_id=obj.object_id,
                        class_name=obj.class_name,
                        x=float(obj.x),
                        y=float(obj.y),
                        reason="object_in_overlap_but_missing_from_other_stream",
                        confidence=confidence,
                        overlap_area_m2=float(area),
                        notes=[
                            "Possible occlusion, detection miss, stale map, edge-of-FoV case, or source false positive.",
                            "This starter reasoner is intentionally conservative.",
                        ],
                    )
                )
    return hypotheses
