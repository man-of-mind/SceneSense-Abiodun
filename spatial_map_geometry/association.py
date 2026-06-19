from __future__ import annotations

import math
from typing import Dict, List, Sequence

from .schemas import AssociationResult, LocalSensorMap, SpatialObject


def xy_distance_m(a: SpatialObject, b: SpatialObject) -> float:
    return float(math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))


def _cluster_centroid(members: Sequence[SpatialObject]) -> tuple[float, float]:
    if not members:
        return (0.0, 0.0)
    return (
        float(sum(obj.x for obj in members) / len(members)),
        float(sum(obj.y for obj in members) / len(members)),
    )


def _max_pairwise_distance(members: Sequence[SpatialObject]) -> float:
    if len(members) < 2:
        return 0.0
    out = 0.0
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            out = max(out, xy_distance_m(a, b))
    return float(out)


def associate_objects(
    local_maps: Sequence[LocalSensorMap],
    distance_threshold_m: float = 3.0,
    require_class_match: bool = True,
) -> List[AssociationResult]:
    """Greedy cross-stream object association.

    This is intentionally simple. It gives us a baseline association layer for
    the first spatial-map prototype.
    """

    clusters: List[List[SpatialObject]] = []
    for local_map in local_maps:
        for obj in local_map.objects:
            best_index = None
            best_distance = float("inf")
            for idx, cluster in enumerate(clusters):
                if any(member.source_stream_id == obj.source_stream_id for member in cluster):
                    continue
                if require_class_match and any(member.class_name != obj.class_name for member in cluster):
                    continue
                centroid_x, centroid_y = _cluster_centroid(cluster)
                dist = math.hypot(float(obj.x) - centroid_x, float(obj.y) - centroid_y)
                if dist <= distance_threshold_m and dist < best_distance:
                    best_index = idx
                    best_distance = dist
            if best_index is None:
                clusters.append([obj])
            else:
                clusters[best_index].append(obj)

    results: List[AssociationResult] = []
    class_counters: Dict[str, int] = {}
    for cluster in clusters:
        class_name = cluster[0].class_name if cluster else "unknown"
        class_counters[class_name] = class_counters.get(class_name, 0) + 1
        centroid_x, centroid_y = _cluster_centroid(cluster)
        results.append(
            AssociationResult(
                canonical_id=f"{class_name}_{class_counters[class_name]:03d}",
                class_name=class_name,
                members=list(cluster),
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                max_pairwise_distance_m=_max_pairwise_distance(cluster),
            )
        )
    return results
