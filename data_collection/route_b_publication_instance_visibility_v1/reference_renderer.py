"""Sequential, isolated CARLA renderer for exact unoccluded actor silhouettes."""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .actor_state import (
    capture_walker_bones,
    carla_transform_from_matrix,
    configure_clone,
    walker_bone_pose_error,
)
from .core import (
    VisibilityGroundTruthError,
    decode_instance_bgra,
    image_bgra,
    instance_mask,
    measure_visibility,
    reproduce_transform_matrix,
    sha256,
    transform_matrix,
    transform_payload,
    write_png_x,
)


REFERENCE_CAMERA_TRANSFORM = {
    "location": {"x": 0.0, "y": 0.0, "z": 800.0},
    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
}


def _camera_transform() -> Any:
    import carla

    return carla.Transform(carla.Location(x=0.0, y=0.0, z=800.0), carla.Rotation())


def _wait_exact(sensor_queue: queue.Queue, frame: int, timeout_s: float = 10.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        item = sensor_queue.get(timeout=max(0.01, deadline - time.monotonic()))
        if int(item.frame) < int(frame):
            continue
        if int(item.frame) == int(frame):
            return item
        break
    raise VisibilityGroundTruthError(f"reference instance camera missed frame {frame}")


def _set_blueprint_attributes(blueprint: Any, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        if blueprint.has_attribute(str(key)):
            try:
                blueprint.set_attribute(str(key), str(value))
            except (RuntimeError, ValueError):
                continue


class ReferenceRenderer:
    def __init__(self, world: Any, output_dir: Path, *, width: int, height: int, fov: float) -> None:
        self.world = world
        self.output_dir = output_dir
        self.width, self.height, self.fov = int(width), int(height), float(fov)
        self.queue: queue.Queue = queue.Queue()
        blueprint = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
        for key, value in (
            ("image_size_x", self.width), ("image_size_y", self.height),
            ("fov", self.fov), ("sensor_tick", 0.0),
        ):
            blueprint.set_attribute(key, str(value))
        self.camera_transform = _camera_transform()
        self.camera = world.spawn_actor(blueprint, self.camera_transform)
        self.camera.listen(self.queue.put)
        self.rendered = 0
        self.background_nonzero_instance_pixels = -1
        self.max_transform_matrix_error = 0.0

    def prove_empty_rig(self) -> dict[str, Any]:
        frame = int(self.world.tick())
        image = _wait_exact(self.queue, frame)
        semantic, ids = decode_instance_bgra(image_bgra(image))
        self.background_nonzero_instance_pixels = int(np.count_nonzero(ids))
        vehicle_or_person = int(np.count_nonzero(np.isin(semantic, [4, 12, 14, 15, 16])))
        if self.background_nonzero_instance_pixels or vehicle_or_person:
            raise VisibilityGroundTruthError(
                "PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED: isolated sky rig contains rendered geometry"
            )
        return {
            "camera_transform": transform_payload(self.camera_transform),
            "background_nonzero_instance_pixels": self.background_nonzero_instance_pixels,
            "background_vehicle_or_person_pixels": vehicle_or_person,
            "external_geometry_absent": True,
        }

    def render(self, state: Mapping[str, Any], visible_mask_path: Path) -> dict[str, Any]:
        bp = self.world.get_blueprint_library().find(str(state["blueprint"]))
        _set_blueprint_attributes(bp, state.get("blueprint_attributes", {}))
        desired = reproduce_transform_matrix(
            self.camera_transform, np.asarray(state["camera_relative_actor_matrix"], dtype=np.float64)
        )
        clone = self.world.try_spawn_actor(bp, carla_transform_from_matrix(desired))
        if clone is None:
            raise VisibilityGroundTruthError(f"cannot spawn isolated clone for {state['sample_id']}/{state['gt_actor_id']}")
        try:
            configure_clone(clone, state)
            clone.set_transform(carla_transform_from_matrix(desired))
            frame = int(self.world.tick())
            _wait_exact(self.queue, frame)
            clone.set_transform(carla_transform_from_matrix(desired))
            frame = int(self.world.tick())
            image = _wait_exact(self.queue, frame)
            actual = transform_matrix(clone.get_transform())
            error = float(np.max(np.abs(actual - desired)))
            self.max_transform_matrix_error = max(self.max_transform_matrix_error, error)
            if error > 1e-4:
                raise VisibilityGroundTruthError(f"reference transform reproduction error {error}")
            bone_error = None
            if str(state["blueprint"]).startswith("walker.pedestrian."):
                bone_error = walker_bone_pose_error(
                    list(state["walker_bones"]), capture_walker_bones(clone)
                )
                if bone_error > 1e-3:
                    raise VisibilityGroundTruthError(f"walker bone-pose reproduction error {bone_error}")
            _semantic, ids = decode_instance_bgra(image_bgra(image))
            reference = instance_mask(ids, int(clone.id))
            import cv2
            visible_raw = cv2.imread(str(visible_mask_path), cv2.IMREAD_UNCHANGED)
            if visible_raw is None:
                raise VisibilityGroundTruthError(f"missing visible mask {visible_mask_path}")
            metrics = measure_visibility(visible_raw != 0, reference)
            relative = Path("unoccluded_masks") / str(state["sample_id"]) / f"actor_{state['gt_actor_id']}.png"
            path = self.output_dir / relative
            reference_hash = write_png_x(path, reference)
            self.rendered += 1
            return {
                **metrics,
                "unoccluded_mask_path": str(relative),
                "unoccluded_mask_sha256": reference_hash,
                "reference_clone_actor_id": int(clone.id),
                "reference_frame_id": int(image.frame),
                "reference_transform_max_abs_error": error,
                "walker_bone_pose_max_abs_error": bone_error,
                "walker_bone_pose_copied": bone_error is None or bone_error <= 1e-3,
                "reference_camera_transform": transform_payload(self.camera_transform),
            }
        finally:
            try:
                clone.destroy()
                self.world.tick()
            except RuntimeError:
                pass

    def close(self) -> bool:
        try:
            self.camera.stop()
        except RuntimeError:
            pass
        try:
            destroyed = bool(self.camera.destroy())
            self.world.tick()
            return destroyed
        except RuntimeError:
            return False
