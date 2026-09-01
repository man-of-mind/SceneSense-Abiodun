"""Exact camera-relative actor and pedestrian pose reproduction helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .core import (
    ZBufferVisibilityError,
    matrix_to_transform_payload,
    relative_transform_matrix,
    transform_matrix,
    transform_payload,
)


def _payload_transform(payload: Mapping[str, Any]) -> Any:
    import carla

    location, rotation = payload["location"], payload["rotation"]
    return carla.Transform(
        carla.Location(
            x=float(location["x"]), y=float(location["y"]), z=float(location["z"])
        ),
        carla.Rotation(
            pitch=float(rotation["pitch"]),
            yaw=float(rotation["yaw"]),
            roll=float(rotation["roll"]),
        ),
    )


def carla_transform_from_matrix(matrix: np.ndarray) -> Any:
    return _payload_transform(matrix_to_transform_payload(matrix))


def capture_walker_bones(actor: Any) -> list[dict[str, Any]]:
    rows = []
    for bone in actor.get_bones().bone_transforms:
        relative = getattr(bone, "relative", None)
        if relative is None:
            raise ZBufferVisibilityError(
                f"walker bone {getattr(bone, 'name', '?')} has no relative transform"
            )
        rows.append(
            {
                "name": str(bone.name),
                "relative": transform_payload(relative),
                "relative_matrix": transform_matrix(relative).tolist(),
            }
        )
    if not rows:
        raise ZBufferVisibilityError(f"walker {actor.id} returned no bone pose")
    return rows


def apply_walker_bones(actor: Any, rows: list[Mapping[str, Any]]) -> None:
    import carla

    if not rows:
        raise ZBufferVisibilityError("cannot reproduce an empty walker bone pose")
    actor.set_bones(
        carla.WalkerBoneControlIn(
            [(str(row["name"]), _payload_transform(row["relative"])) for row in rows]
        )
    )
    actor.show_pose()
    actor.blend_pose(1.0)


def walker_bone_pose_error(
    expected: list[Mapping[str, Any]], observed: list[Mapping[str, Any]]
) -> float:
    left = {
        str(row["name"]): np.asarray(row["relative_matrix"], dtype=np.float64)
        for row in expected
    }
    right = {
        str(row["name"]): np.asarray(row["relative_matrix"], dtype=np.float64)
        for row in observed
    }
    if not left or set(left) != set(right):
        raise ZBufferVisibilityError("walker bone-name reconciliation failed")
    return max(float(np.max(np.abs(left[name] - right[name]))) for name in left)


def capture_actor_state(actor: Any, camera_transform: Any, class_name: str) -> dict[str, Any]:
    actor_transform = actor.get_transform()
    state = {
        "actor_id": int(actor.id),
        "class_name": str(class_name),
        "blueprint": str(actor.type_id),
        "blueprint_attributes": dict(actor.attributes),
        "actor_transform": transform_payload(actor_transform),
        "camera_transform": transform_payload(camera_transform),
        "camera_relative_actor_matrix": relative_transform_matrix(
            camera_transform, actor_transform
        ).tolist(),
        "walker_bones": [],
    }
    if str(actor.type_id).startswith("walker.pedestrian."):
        state["walker_bones"] = capture_walker_bones(actor)
    return state


def configure_clone(clone: Any, state: Mapping[str, Any]) -> None:
    try:
        clone.set_simulate_physics(False)
    except (AttributeError, RuntimeError):
        pass
    if str(state["blueprint"]).startswith("walker.pedestrian."):
        apply_walker_bones(clone, list(state["walker_bones"]))


def set_blueprint_attributes(blueprint: Any, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        if not blueprint.has_attribute(str(key)):
            continue
        try:
            blueprint.set_attribute(str(key), str(value))
        except (RuntimeError, ValueError):
            continue
