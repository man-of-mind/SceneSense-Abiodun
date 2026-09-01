"""Capture and reproduce silhouette-relevant CARLA actor state."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .core import (
    VisibilityGroundTruthError,
    matrix_to_transform_payload,
    relative_transform_matrix,
    transform_matrix,
    transform_payload,
)


def _payload_transform(payload: Mapping[str, Any]) -> Any:
    import carla

    location = payload["location"]
    rotation = payload["rotation"]
    return carla.Transform(
        carla.Location(x=float(location["x"]), y=float(location["y"]), z=float(location["z"])),
        carla.Rotation(
            pitch=float(rotation["pitch"]), yaw=float(rotation["yaw"]), roll=float(rotation["roll"])
        ),
    )


def carla_transform_from_matrix(matrix: np.ndarray) -> Any:
    return _payload_transform(matrix_to_transform_payload(matrix))


def capture_walker_bones(actor: Any) -> list[dict[str, Any]]:
    output = actor.get_bones()
    rows = []
    for bone in output.bone_transforms:
        transform = None
        source_field = ""
        for field in ("relative", "transform", "component", "world"):
            candidate = getattr(bone, field, None)
            if candidate is not None:
                transform = candidate
                source_field = field
                break
        if transform is None:
            raise VisibilityGroundTruthError(f"walker bone {getattr(bone, 'name', '?')} lacks transform")
        rows.append({
            "name": str(bone.name),
            "source_field": source_field,
            "transform": transform_payload(transform),
            "matrix": transform_matrix(transform).tolist(),
        })
    if not rows:
        raise VisibilityGroundTruthError(f"walker {actor.id} returned no bone pose")
    return rows


def apply_walker_bones(actor: Any, rows: list[Mapping[str, Any]]) -> None:
    import carla

    if not rows:
        raise VisibilityGroundTruthError("cannot reproduce an empty walker bone pose")
    control = carla.WalkerBoneControlIn([
        (str(row["name"]), _payload_transform(row["transform"])) for row in rows
    ])
    actor.set_bones(control)
    actor.show_pose()
    actor.blend_pose(1.0)


def walker_bone_pose_error(
    expected: list[Mapping[str, Any]], observed: list[Mapping[str, Any]],
) -> float:
    left = {str(row["name"]): np.asarray(row["matrix"], dtype=np.float64) for row in expected}
    right = {str(row["name"]): np.asarray(row["matrix"], dtype=np.float64) for row in observed}
    if not left or set(left) != set(right):
        raise VisibilityGroundTruthError("walker bone-name reconciliation failed")
    return max(float(np.max(np.abs(left[name] - right[name]))) for name in left)


def capture_actor_state(
    actor: Any,
    camera_transform: Any,
    *,
    sample_id: str,
    frame_id: int,
    class_name: str,
    range_m: float,
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    actor_transform = actor.get_transform()
    relative = relative_transform_matrix(camera_transform, actor_transform)
    state: dict[str, Any] = {
        "sample_id": sample_id,
        "frame_id": int(frame_id),
        "gt_actor_id": int(actor.id),
        "class_name": str(class_name),
        "blueprint": str(actor.type_id),
        "blueprint_attributes": dict(actor.attributes),
        "actor_transform": transform_payload(actor_transform),
        "camera_transform": transform_payload(camera_transform),
        "camera_relative_actor_matrix": relative.tolist(),
        "range_m": float(range_m),
        "source_geometry": dict(source_row),
        "walker_bones": [],
        "vehicle_state": None,
    }
    if str(actor.type_id).startswith("walker.pedestrian."):
        state["walker_bones"] = capture_walker_bones(actor)
    elif str(actor.type_id).startswith("vehicle."):
        control = actor.get_control()
        wheels = {}
        try:
            import carla
            for name in ("FL_Wheel", "FR_Wheel", "BL_Wheel", "BR_Wheel"):
                location = getattr(carla.VehicleWheelLocation, name)
                wheels[name] = float(actor.get_wheel_steer_angle(location))
        except (AttributeError, RuntimeError):
            wheels = {}
        state["vehicle_state"] = {
            "control": {
                "throttle": float(control.throttle), "steer": float(control.steer),
                "brake": float(control.brake), "hand_brake": bool(control.hand_brake),
                "reverse": bool(control.reverse), "manual_gear_shift": bool(control.manual_gear_shift),
                "gear": int(control.gear),
            },
            "light_state": int(actor.get_light_state()),
            "wheel_steer_angle_deg": wheels,
            "doors": "traffic actors use blueprint-default closed doors; CARLA exposes no door-state getter",
        }
    return state


def configure_clone(clone: Any, state: Mapping[str, Any]) -> None:
    import carla

    try:
        clone.set_simulate_physics(False)
    except (AttributeError, RuntimeError):
        pass
    if str(state["blueprint"]).startswith("walker.pedestrian."):
        apply_walker_bones(clone, list(state["walker_bones"]))
    elif str(state["blueprint"]).startswith("vehicle.") and state.get("vehicle_state"):
        control = state["vehicle_state"]["control"]
        clone.apply_control(carla.VehicleControl(
            throttle=float(control["throttle"]), steer=float(control["steer"]),
            brake=float(control["brake"]), hand_brake=bool(control["hand_brake"]),
            reverse=bool(control["reverse"]), manual_gear_shift=bool(control["manual_gear_shift"]),
            gear=int(control["gear"]),
        ))
        try:
            clone.set_light_state(carla.VehicleLightState(int(state["vehicle_state"]["light_state"])))
        except (TypeError, RuntimeError):
            pass
