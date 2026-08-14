"""Adapters from existing SceneSense map snapshots to the Phase-2 wire schema."""

from __future__ import annotations

from typing import Mapping

from .schemas import MapContribution, MapObjectObservation, with_exact_payload_bytes


def _component(raw: Mapping[str, object], name: str) -> float:
    velocity = raw.get("velocity")
    if isinstance(velocity, Mapping) and name in velocity:
        return float(velocity[name])
    return float(raw.get(f"v{name}_mps", 0.0))


def snapshot_stream_to_contribution(
    snapshot: Mapping[str, object],
    *,
    source_stream_id: str,
    recipient_ue_id: str,
    sequence_number: int,
    captured_at_s: float,
    published_at_s: float,
    profile_id: str,
    payload_bytes: int | None = None,
) -> MapContribution:
    """Extract one source's *raw* objects without leaking simulator identity.

    The existing server's fused objects already combine sources, so a helper
    contribution must originate from ``raw_spatial_map_objects`` filtered by
    ``source_stream_id``.  Motion defaults to zero until the source publishes
    an explicit velocity estimate; CARLA ground-truth velocity is not imported.
    """

    raw_objects = snapshot.get("raw_spatial_map_objects", [])
    if not isinstance(raw_objects, list):
        raise ValueError("snapshot raw_spatial_map_objects must be a list")
    observations = []
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping) or str(raw.get("source_stream_id", "")) != source_stream_id:
            continue
        location = raw.get("location")
        if not isinstance(location, Mapping):
            raise ValueError("raw map object is missing a world-frame location")
        observations.append(
            MapObjectObservation(
                source_track_id=str(raw.get("id", f"{source_stream_id}:{sequence_number}:{index}")),
                class_name=str(raw.get("type", "unknown")).lower(),
                x_m=float(location["x"]),
                y_m=float(location["y"]),
                vx_mps=_component(raw, "x"),
                vy_mps=_component(raw, "y"),
                confidence=float(raw.get("score", 0.0)),
                observed_at_s=float(captured_at_s),
                occlusion_state=str(raw.get("occlusion_state", "unknown")),
                hazard_score=float(raw.get("hazard_score", 0.0)),
            )
        )
    contribution = MapContribution(
        contribution_id=f"{source_stream_id}:{recipient_ue_id}:{sequence_number}",
        source_ue_id=source_stream_id,
        recipient_ue_id=recipient_ue_id,
        sequence_number=int(sequence_number),
        captured_at_s=float(captured_at_s),
        published_at_s=float(published_at_s),
        profile_id=str(profile_id),
        payload_bytes=0,
        objects=tuple(observations),
    )
    # ``payload_bytes`` is retained as a backwards-compatible call-site hint;
    # the wire contract always records the exact serialized application bytes.
    _ = payload_bytes
    return with_exact_payload_bytes(contribution)
