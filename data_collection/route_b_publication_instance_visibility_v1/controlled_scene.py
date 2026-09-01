#!/usr/bin/env python3
"""One controlled CARLA vehicle/pedestrian renderer qualification scene."""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .actor_state import capture_actor_state, configure_clone
from .core import (
    INSTANCE_ENCODING,
    VISIBILITY_DEFINITION,
    VisibilityGroundTruthError,
    decode_instance_bgra,
    image_bgra,
    instance_mask,
    measure_visibility,
    prove_actor_id_mapping,
    sha256,
    write_json_x,
    write_png_x,
)
from .reference_renderer import ReferenceRenderer


WIDTH, HEIGHT, FOV = 1280, 720, 120.0
VEHICLE_TAGS, PERSON_TAGS = {14, 15, 16, 17, 18, 19}, {12, 13}


def _camera_bp(world: Any, type_id: str) -> Any:
    bp = world.get_blueprint_library().find(type_id)
    for key, value in (
        ("image_size_x", WIDTH), ("image_size_y", HEIGHT),
        ("fov", FOV), ("sensor_tick", 0.0),
    ):
        bp.set_attribute(key, str(value))
    return bp


def _wait_exact(sensor_queue: queue.Queue, frame: int, timeout_s: float = 10.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        item = sensor_queue.get(timeout=max(0.01, deadline - time.monotonic()))
        if int(item.frame) < frame:
            continue
        if int(item.frame) == frame:
            return item
        break
    raise VisibilityGroundTruthError(f"controlled sensor missed frame {frame}")


def _spawn_disabled(world: Any, blueprint: Any, transform: Any) -> Any:
    import carla

    staging = carla.Transform(carla.Location(x=0.0, y=0.0, z=950.0), transform.rotation)
    actor = world.try_spawn_actor(blueprint, staging)
    if actor is None:
        raise VisibilityGroundTruthError(f"controlled actor spawn failed: {blueprint.id}")
    try:
        actor.set_simulate_physics(False)
    except (AttributeError, RuntimeError):
        pass
    actor.set_transform(transform)
    return actor


def _choose(world: Any, preferred: str, pattern: str) -> Any:
    library = world.get_blueprint_library()
    try:
        return library.find(preferred)
    except RuntimeError:
        values = list(library.filter(pattern))
        if not values:
            raise VisibilityGroundTruthError(f"no blueprint for {pattern}")
        return values[0]


def _intrinsics() -> np.ndarray:
    focal = WIDTH / (2.0 * math.tan(math.radians(FOV) / 2.0))
    return np.asarray([[focal, 0.0, WIDTH / 2.0], [0.0, focal, HEIGHT / 2.0], [0.0, 0.0, 1.0]])


def _condition_transform(target: Any, camera: Any, *, x: float, pixel_shift: float) -> Any:
    import carla

    camera_z = float(camera.location.z)
    target_location = target.location
    ratio = x / float(target_location.x - camera.location.x)
    relative_y = float(target_location.y - camera.location.y) * ratio
    relative_z = float(target_location.z - camera_z) * ratio
    focal = float(_intrinsics()[0, 0])
    relative_y += float(pixel_shift) * x / focal
    return carla.Transform(
        carla.Location(
            x=float(camera.location.x) + x,
            y=float(camera.location.y) + relative_y,
            z=camera_z + relative_z,
        ),
        target.rotation,
    )


def run(host: str, port: int, output_dir: Path) -> dict[str, Any]:
    import carla
    import cv2
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.publication_instance_visibility_evaluation_v1.protocol import load_registered_protocol

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    registration = load_registered_protocol()
    client = carla.Client(host, int(port)); client.set_timeout(30.0)
    world = client.get_world()
    original = world.get_settings()
    settings = world.get_settings(); settings.synchronous_mode = True; settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    actors: list[Any] = []
    rgb_queue, instance_queue = queue.Queue(), queue.Queue()
    try:
        camera_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=500.0), carla.Rotation())
        rgb = world.spawn_actor(_camera_bp(world, "sensor.camera.rgb"), camera_tf)
        instance = world.spawn_actor(_camera_bp(world, "sensor.camera.instance_segmentation"), camera_tf)
        actors.extend((rgb, instance)); rgb.listen(rgb_queue.put); instance.listen(instance_queue.put)
        vehicle_bp = _choose(world, "vehicle.audi.a2", "vehicle.*")
        walker_bp = _choose(world, "walker.pedestrian.0001", "walker.pedestrian.*")
        vehicle_tf = carla.Transform(carla.Location(x=16.0, y=-4.0, z=498.5), carla.Rotation())
        walker_tf = carla.Transform(carla.Location(x=12.0, y=3.0, z=498.3), carla.Rotation())
        vehicle = _spawn_disabled(world, vehicle_bp, vehicle_tf)
        walker = _spawn_disabled(world, walker_bp, walker_tf)
        actors.extend((vehicle, walker))
        world.tick(); world.tick()

        def capture(name: str) -> tuple[Any, Any, np.ndarray, np.ndarray]:
            frame = int(world.tick())
            rgb_image, instance_image = _wait_exact(rgb_queue, frame), _wait_exact(instance_queue, frame)
            if (int(rgb_image.frame) != int(instance_image.frame)
                    or float(rgb_image.timestamp) != float(instance_image.timestamp)):
                raise VisibilityGroundTruthError(f"controlled RGB/instance synchronization failed: {name}")
            raw = image_bgra(instance_image); _semantic, ids = decode_instance_bgra(raw)
            bgr = np.frombuffer(rgb_image.raw_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 4))[:, :, :3]
            condition_dir = output_dir / "conditions" / name; condition_dir.mkdir(parents=True, exist_ok=False)
            if not cv2.imwrite(str(condition_dir / "rgb.png"), bgr):
                raise VisibilityGroundTruthError("controlled RGB write failed")
            if not cv2.imwrite(str(condition_dir / "instance.png"), raw):
                raise VisibilityGroundTruthError("controlled instance write failed")
            return rgb_image, instance_image, raw, ids

        clear_rgb, clear_instance, clear_raw, clear_ids = capture("clear")
        semantic, _ = decode_instance_bgra(clear_raw)
        class_tokens = {
            "vehicle": sorted(int(value) for value in np.unique(clear_ids[np.isin(semantic, list(VEHICLE_TAGS))]) if int(value) > 0),
            "person": sorted(int(value) for value in np.unique(clear_ids[np.isin(semantic, list(PERSON_TAGS))]) if int(value) > 0),
        }
        if len(class_tokens["vehicle"]) != 1 or len(class_tokens["person"]) != 1:
            raise VisibilityGroundTruthError(
                "PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED: controlled actor-specific "
                f"instance components unavailable or ambiguous: {class_tokens}"
            )
        mapping = prove_actor_id_mapping([
            {"actor_id": int(vehicle.id), "rendered_instance_ids": class_tokens["vehicle"]},
            {"actor_id": int(walker.id), "rendered_instance_ids": class_tokens["person"]},
        ])
        targets = {"vehicle": vehicle, "person": walker}
        states, clear_masks = {}, {}
        for class_name, actor in targets.items():
            rendered_id = int(mapping["actor_to_rendered_instance_id"][str(actor.id)])
            mask = instance_mask(clear_ids, rendered_id); clear_masks[class_name] = mask
            if not np.any(mask):
                raise VisibilityGroundTruthError(f"controlled clear {class_name} mask is empty")
            tags = {int(value) for value in np.unique(semantic[mask])}
            expected = VEHICLE_TAGS if class_name == "vehicle" else PERSON_TAGS
            if not tags or not tags.issubset(expected):
                raise VisibilityGroundTruthError(
                    f"PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED: {class_name} tags {sorted(tags)}"
                )
            relative = Path("conditions/clear") / f"visible_{class_name}.png"
            write_png_x(output_dir / relative, mask)
            state = capture_actor_state(
                actor, clear_instance.transform, sample_id="controlled_clear", frame_id=int(clear_instance.frame),
                class_name=class_name,
                range_m=actor.get_location().distance(clear_instance.transform.location),
                source_row={"controlled": True},
            )
            state.update({
                "visible_mask_path": str(relative), "camera_intrinsics": _intrinsics().tolist(),
                "camera_resolution": [WIDTH, HEIGHT], "camera_fov": FOV,
            })
            states[class_name] = state

        reference = ReferenceRenderer(world, output_dir, width=WIDTH, height=HEIGHT, fov=FOV)
        rig = reference.prove_empty_rig()
        references, clear_metrics = {}, {}
        try:
            for class_name, state in states.items():
                result = reference.render(state, output_dir / state["visible_mask_path"])
                references[class_name] = result
                import cv2
                ref = cv2.imread(str(output_dir / result["unoccluded_mask_path"]), cv2.IMREAD_UNCHANGED) != 0
                references[class_name]["mask"] = ref
                metric = measure_visibility(clear_masks[class_name], ref)
                union = int(np.count_nonzero(clear_masks[class_name] | ref))
                metric["mask_iou"] = int(metric["overlap_pixels"]) / max(1, union)
                clear_metrics[class_name] = metric
        finally:
            reference_cleanup = reference.close()

        bbox_width = {
            name: int(row["unoccluded_bbox_x1"]) - int(row["unoccluded_bbox_x0"])
            for name, row in clear_metrics.items()
        }

        def spawn_occluders(partial: bool) -> list[Any]:
            output = []
            for class_name, target in targets.items():
                bp = vehicle_bp if class_name == "vehicle" else walker_bp
                shift = 0.55 * bbox_width[class_name] if partial else 0.0
                target_tf = target.get_transform()
                x = (target_tf.location.x - camera_tf.location.x) - 1.0
                transform = _condition_transform(target_tf, camera_tf, x=x, pixel_shift=shift)
                occluder = _spawn_disabled(world, bp, transform)
                if class_name == "person":
                    configure_clone(occluder, states[class_name])
                    occluder.set_transform(transform)
                output.append(occluder)
            world.tick()
            return output

        conditions: dict[str, dict[str, Any]] = {"clear": clear_metrics}
        for name, partial in (("partial", True), ("full", False)):
            occluders = spawn_occluders(partial)
            try:
                _rgb_image, inst_image, _raw, ids = capture(name)
                values = {}
                for class_name, actor in targets.items():
                    visible = instance_mask(ids, int(actor.id))
                    relative = Path("conditions") / name / f"visible_{class_name}.png"
                    write_png_x(output_dir / relative, visible)
                    values[class_name] = measure_visibility(visible, references[class_name]["mask"])
                conditions[name] = values
            finally:
                for actor in reversed(occluders):
                    actor.destroy()
                world.tick()

        gates = {
            "rgb_instance_frame_synchronization_exact": True,
            "actor_instance_mapping_bijective": bool(mapping["bijection_proven"]),
            "positive_unoccluded_area": all(
                int(conditions["clear"][name]["unoccluded_pixels"]) > 0 for name in targets
            ),
            "finite_bounded_visibility": all(
                math.isfinite(float(conditions[condition][name]["visibility"]))
                and 0.0 <= float(conditions[condition][name]["visibility"]) <= 1.0
                for condition in conditions for name in targets
            ),
            "clear_visibility_ge_0_98": all(
                float(conditions["clear"][name]["visibility"]) >= 0.98 for name in targets
            ),
            "clear_coordinate_iou_ge_0_98": all(
                float(conditions["clear"][name]["mask_iou"]) >= 0.98 for name in targets
            ),
            "partial_strictly_between_clear_and_full": all(
                float(conditions["clear"][name]["visibility"])
                > float(conditions["partial"][name]["visibility"])
                > float(conditions["full"][name]["visibility"])
                for name in targets
            ),
            "full_visibility_le_0_02": all(
                float(conditions["full"][name]["visibility"]) <= 0.02 for name in targets
            ),
            "visible_outside_reference_reported": all(
                "visible_outside_reference_pixels" in conditions[condition][name]
                for condition in conditions for name in targets
            ),
            "walker_bone_pose_copied": bool(references["person"]["walker_bone_pose_copied"]),
            "isolated_rig_external_geometry_absent": bool(rig["external_geometry_absent"]),
            "reference_camera_cleanup": bool(reference_cleanup),
        }
        qualified = all(gates.values())
        renderer_proof = {
            "actor_id_mapping_proven": gates["actor_instance_mapping_bijective"],
            "reference_intrinsics_equal": True,
            "reference_coordinates_equal": gates["clear_coordinate_iou_ge_0_98"],
            "external_geometry_absent": gates["isolated_rig_external_geometry_absent"],
            "walker_bone_pose_copy_proven": gates["walker_bone_pose_copied"],
            "instance_encoding": INSTANCE_ENCODING,
            "mapping": mapping,
        }
        result = {
            "schema": "route_b_publication_controlled_visibility_qualification_v1",
            "terminal": (
                "PUBLICATION_VISIBILITY_CONTROLLED_SCENE_QUALIFIED" if qualified
                else "PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED"
            ),
            "qualified": qualified,
            "definition": VISIBILITY_DEFINITION,
            "conditions": conditions,
            "gates": gates,
            "renderer_proof": renderer_proof,
            "registration": {key: registration[key] for key in (
                "lock_sha256", "protocol_sha256", "bound_files_verified"
            )},
        }
        write_json_x(output_dir / "controlled_scene_result.json", result)
        return result
    finally:
        for actor in reversed(actors):
            try:
                if str(actor.type_id).startswith("sensor."):
                    actor.stop()
            except RuntimeError:
                pass
            try:
                actor.destroy()
            except RuntimeError:
                pass
        try:
            world.tick()
        except RuntimeError:
            pass
        world.apply_settings(original)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.host, args.port, args.output_dir.resolve())
    except Exception as exc:
        print(f"PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["qualified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
