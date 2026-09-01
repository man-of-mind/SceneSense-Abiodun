"""Route B v2 collector extension for exact actor-instance visibility evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import queue
from pathlib import Path
from typing import Any

import numpy as np

import data_collection.run_route_b_perception_collection_v2 as v2

from .actor_state import capture_actor_state
from .core import (
    INSTANCE_ENCODING,
    VISIBILITY_DEFINITION,
    VisibilityGroundTruthError,
    decode_instance_bgra,
    image_bgra,
    instance_mask,
    require_renderer_proof,
    sha256,
    transform_matrix,
    transform_payload,
    write_json_x,
    write_png_x,
)
from .reference_renderer import ReferenceRenderer


INSTANCE_TAGS = {"vehicle": {14, 15, 16, 17, 18, 19}, "person": {12, 13}}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_raw_png_x(path: Path, image: np.ndarray) -> str:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]):
        raise VisibilityGroundTruthError(f"failed to write {path}")
    return sha256(path)


def _geometrically_qualified(row: dict[str, str], width: int, height: int) -> bool:
    cx, cy = float(row["gt_center_x"]), float(row["gt_center_y"])
    return (
        float(row["gt_distance_m"]) <= 40.0
        and float(row["gt_bbox_area_px"]) >= 12.0
        and 0.0 <= cx < float(width)
        and 0.0 <= cy < float(height)
    )


class PublicationInstanceVisibilityCollector(v2.PerceptionCollectorV2):
    """Adds one synchronized instance stream and deferred isolated rendering."""

    controlled_proof_path: Path | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.instance_queue: queue.Queue = queue.Queue()
        self.instance_camera: Any | None = None
        self.instance_cache: dict[str, Any] = {}
        self.instance_alignment: list[dict[str, Any]] = []
        self.instance_manifest: list[dict[str, Any]] = []
        self.actor_states: list[dict[str, Any]] = []
        self.reference_results: list[dict[str, Any]] = []
        self.reference_summary: dict[str, Any] = {}
        self.reference_error = ""
        self.mapping_ambiguities = 0
        self.visibility_fallback_invocations = 0
        self.instance_roundtrip_verified = False
        self.controlled_proof: dict[str, Any] = {}
        super().__init__(**kwargs)

    def _spawn_sensors(self) -> None:
        super()._spawn_sensors()
        blueprint = self.world.get_blueprint_library().find("sensor.camera.instance_segmentation")
        for key, value in (
            ("image_size_x", self.args.camera_width),
            ("image_size_y", self.args.camera_height),
            ("fov", self.args.camera_fov),
            ("sensor_tick", 0.0),
        ):
            blueprint.set_attribute(key, str(value))
        relative = self.parked.fusion_runtime._ego_camera_transform(self.args)
        self.instance_camera = self.world.spawn_actor(blueprint, relative, attach_to=self.ego)
        self.instance_camera.listen(self.instance_queue.put)
        rgb_attrs = {key: str(self.camera.attributes.get(key)) for key in (
            "image_size_x", "image_size_y", "fov", "sensor_tick"
        )}
        instance_attrs = {key: str(self.instance_camera.attributes.get(key)) for key in rgb_attrs}
        if rgb_attrs != instance_attrs:
            raise VisibilityGroundTruthError(f"RGB/instance camera attributes differ: {rgb_attrs} vs {instance_attrs}")

    def _lock_cadence_phase(self) -> None:
        super()._lock_cadence_phase()
        frames = []
        while True:
            try:
                frames.append(int(self.instance_queue.get_nowait().frame))
            except queue.Empty:
                break
        if len(frames) < 2 or {b - a for a, b in zip(frames, frames[1:])} != {1}:
            raise VisibilityGroundTruthError(f"instance camera cadence not continuous: {frames}")
        self.instance_warmup_frames = frames

    def _write_metadata(self) -> None:
        from pole_lraspp_multimodal_fusion.object_head_pilot_v1.publication_instance_visibility_evaluation_v1.protocol import load_registered_protocol

        proof_path = self.controlled_proof_path
        if proof_path is None or not proof_path.is_file():
            raise VisibilityGroundTruthError("controlled renderer proof is required before traffic smoke")
        proof_document = json.loads(proof_path.read_text(encoding="utf-8"))
        self.controlled_proof = proof_document["renderer_proof"]
        require_renderer_proof(self.controlled_proof)
        registration = load_registered_protocol()
        self.instance_dir = self.output_dir / "instance"
        self.visible_dir = self.output_dir / "visible_masks"
        self.state_dir = self.output_dir / "actor_states"
        self.visible_semantic_dir = self.output_dir / "visible_semantic"
        for path in (self.instance_dir, self.visible_dir, self.state_dir, self.visible_semantic_dir):
            path.mkdir(exist_ok=False)
        super()._write_metadata()
        metadata_path = self.output_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({
            "schema": "route_b_publication_instance_visibility_collection_v1",
            "publication_instance_visibility": {
                "definition": VISIBILITY_DEFINITION,
                "instance_encoding": INSTANCE_ENCODING,
                "normal_camera_blueprint": str(self.instance_camera.type_id),
                "configured_attributes": {
                    key: str(self.instance_camera.attributes.get(key))
                    for key in ("image_size_x", "image_size_y", "fov", "sensor_tick")
                },
                "rgb_instance_transform_identical": True,
                "actor_id_mapping_proof": str(proof_path),
                "actor_id_mapping_proof_sha256": sha256(proof_path),
                "reference_render": "deferred sequential isolated CARLA sky rig after traffic route",
                "visibility_fallbacks": [],
                "depth_box_or_semantic_only_visibility_fallback": False,
                "non_visibility_geometry_source": "unchanged Route B v2 object rows; 40 m and 12 px",
            },
            "publication_registration": {
                key: registration[key] for key in (
                    "lock_path", "protocol_path", "lock_sha256", "protocol_sha256", "bound_files_verified"
                )
            },
        })
        self.parked.save_json(metadata_path, metadata)

    def prepare_input(self, frame_id: int, route_tick: int, radar_measurement: Any, sweep_index: int) -> None:
        instance_image = self._drain_exact("instance", frame_id, 5.0)
        if instance_image is None:
            raise VisibilityGroundTruthError(f"missing synchronized instance frame {frame_id}")
        self.instance_cache = {"frame_id": int(frame_id), "measurement": instance_image}
        super().prepare_input(frame_id, route_tick, radar_measurement, sweep_index)
        rgb = next((item for item in reversed(self.prepared_records) if int(item["frame_id"]) == int(frame_id)), None)
        self.instance_alignment.append({
            "frame_id": int(frame_id),
            "instance_frame_id": int(instance_image.frame),
            "instance_timestamp_s": float(instance_image.timestamp),
            "world_frame_exact": int(instance_image.frame) == int(frame_id),
            "persisted": bool(rgb and rgb["persisted"]),
        })

    def _persist(self, **kwargs: Any) -> None:
        frame_id = int(kwargs["frame_id"])
        image = kwargs["image"]
        instance_image = self.instance_cache.get("measurement")
        if instance_image is None or int(instance_image.frame) != frame_id:
            raise VisibilityGroundTruthError(f"saved frame {frame_id} does not own instance image")
        sample_id = f"{self.experiment_id}_{self.saved:06d}_frame{frame_id}"
        super()._persist(**kwargs)
        rows = [row for row in _read_csv(self.object_boxes_path) if row["sample_id"] == sample_id]
        qualified = [
            row for row in rows
            if row["label"] in ("vehicle", "person")
            and _geometrically_qualified(row, self.args.camera_width, self.args.camera_height)
        ]
        raw = image_bgra(instance_image)
        semantic, rendered_ids = decode_instance_bgra(raw)
        instance_rel = Path("instance") / f"{sample_id}.png"
        instance_path = self.output_dir / instance_rel
        instance_hash = _write_raw_png_x(instance_path, raw)
        if not self.instance_roundtrip_verified:
            import cv2
            roundtrip = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
            if roundtrip is None or not np.array_equal(roundtrip, raw):
                raise VisibilityGroundTruthError("instance PNG lossless round-trip failed")
            self.instance_roundtrip_verified = True

        exact_semantic = np.zeros(rendered_ids.shape, dtype=np.uint8)
        actor_records = []
        for row in qualified:
            actor_id = int(row["gt_actor_id"])
            actor = self.world.get_actor(actor_id)
            if actor is None:
                raise VisibilityGroundTruthError(f"qualified actor disappeared: {sample_id}/{actor_id}")
            mask = instance_mask(rendered_ids, actor_id)
            observed_tags = sorted(int(value) for value in np.unique(semantic[mask])) if np.any(mask) else []
            if len(observed_tags) > 1:
                self.mapping_ambiguities += 1
                raise VisibilityGroundTruthError(
                    f"actor {actor_id} has multiple semantic tags {observed_tags}"
                )
            label_value = 1 if row["label"] == "vehicle" else 2
            exact_semantic[mask] = label_value
            visible_rel = Path("visible_masks") / sample_id / f"actor_{actor_id}.png"
            visible_path = self.output_dir / visible_rel
            visible_hash = write_png_x(visible_path, mask)
            state = capture_actor_state(
                actor, instance_image.transform, sample_id=sample_id, frame_id=frame_id,
                class_name=row["label"], range_m=float(row["gt_distance_m"]), source_row=row,
            )
            state.update({
                "camera_intrinsics": np.asarray(self.intrinsics_full, dtype=np.float64).tolist(),
                "camera_resolution": [self.args.camera_width, self.args.camera_height],
                "camera_fov": float(self.args.camera_fov),
                "normal_instance_frame_id": int(instance_image.frame),
                "normal_instance_timestamp_s": float(instance_image.timestamp),
                "visible_mask_path": str(visible_rel),
                "visible_mask_sha256": visible_hash,
                "visible_pixels": int(np.count_nonzero(mask)),
                "rendered_instance_id": actor_id,
                "rendered_semantic_tags": observed_tags,
                "mapping_unambiguous": not observed_tags or len(observed_tags) == 1,
                "touches_frame_edge": bool(
                    float(row["gt_bbox_x"]) <= 0.0 or float(row["gt_bbox_y"]) <= 0.0
                    or float(row["gt_bbox_x"]) + float(row["gt_bbox_w"]) >= self.args.camera_width
                    or float(row["gt_bbox_y"]) + float(row["gt_bbox_h"]) >= self.args.camera_height
                ),
                "geometry_qualified": True,
            })
            state_rel = Path("actor_states") / sample_id / f"actor_{actor_id}.json"
            write_json_x(self.output_dir / state_rel, state)
            state["actor_state_path"] = str(state_rel)
            state["actor_state_sha256"] = sha256(self.output_dir / state_rel)
            self.actor_states.append(state)
            actor_records.append({
                "gt_actor_id": actor_id, "class_name": row["label"],
                "visible_mask_path": str(visible_rel), "visible_mask_sha256": visible_hash,
                "visible_pixels": int(np.count_nonzero(mask)), "rendered_semantic_tags": observed_tags,
            })
        semantic_rel = Path("visible_semantic") / f"{sample_id}.png"
        semantic_hash = _write_raw_png_x(self.output_dir / semantic_rel, exact_semantic)
        rgb_transform_error = float(np.max(np.abs(
            transform_matrix(image.transform) - transform_matrix(instance_image.transform)
        )))
        if rgb_transform_error > 1e-6 or int(image.frame) != int(instance_image.frame):
            raise VisibilityGroundTruthError(f"RGB/instance synchronization drift at {sample_id}")
        self.instance_manifest.append({
            "sample_id": sample_id, "frame_id": frame_id,
            "rgb_frame_id": int(image.frame), "instance_frame_id": int(instance_image.frame),
            "rgb_timestamp_s": float(image.timestamp),
            "instance_timestamp_s": float(instance_image.timestamp),
            "timestamp_delta_s": abs(float(image.timestamp) - float(instance_image.timestamp)),
            "transform_max_abs_error": rgb_transform_error,
            "instance_path": str(instance_rel), "instance_sha256": instance_hash,
            "visible_semantic_path": str(semantic_rel), "visible_semantic_sha256": semantic_hash,
            "geometry_qualified_actors": len(actor_records), "actors": actor_records,
        })

    def _render_references(self) -> None:
        renderer = ReferenceRenderer(
            self.world, self.output_dir, width=self.args.camera_width,
            height=self.args.camera_height, fov=self.args.camera_fov,
        )
        try:
            rig = renderer.prove_empty_rig()
            for index, state in enumerate(self.actor_states, 1):
                visible_path = self.output_dir / state["visible_mask_path"]
                rendered = renderer.render(state, visible_path)
                self.reference_results.append({**state, **rendered})
                if index == 1 or index % 100 == 0:
                    print(f"reference renders {index}/{len(self.actor_states)}", flush=True)
            self.reference_summary = {
                **rig,
                "requested": len(self.actor_states),
                "rendered": renderer.rendered,
                "max_transform_matrix_error": renderer.max_transform_matrix_error,
                "all_positive_unoccluded_area": bool(self.reference_results)
                    and all(int(row["unoccluded_pixels"]) > 0 for row in self.reference_results),
                "all_visibility_finite_bounded": bool(self.reference_results)
                    and all(math.isfinite(float(row["visibility"])) and 0.0 <= float(row["visibility"]) <= 1.0
                            for row in self.reference_results),
                "all_walker_bones_copied": all(
                    row["class_name"] != "person" or row["walker_bone_pose_copied"]
                    for row in self.reference_results
                ),
            }
        finally:
            self.reference_summary["reference_camera_cleanup"] = renderer.close()
        write_json_x(self.output_dir / "instance_manifest.json", self.instance_manifest)
        write_json_x(self.output_dir / "visibility_measurements.json", self.reference_results)
        write_json_x(self.output_dir / "reference_render_summary.json", self.reference_summary)
        paths = []
        for item in self.instance_manifest:
            paths.extend((item["instance_path"], item["visible_semantic_path"]))
            paths.extend(actor["visible_mask_path"] for actor in item["actors"])
        for row in self.reference_results:
            paths.extend((row["actor_state_path"], row["unoccluded_mask_path"]))
        paths.extend(("metadata.json", "manifest.csv", "object_boxes.csv", "instance_manifest.json",
                      "visibility_measurements.json", "reference_render_summary.json"))
        files = []
        for relative in sorted(set(paths)):
            path = self.output_dir / relative
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
        payload = hashlib.sha256("".join(row["sha256"] for row in files).encode("ascii")).hexdigest()
        write_json_x(self.output_dir / "EVIDENCE_MANIFEST.json", {
            "schema": "route_b_publication_instance_visibility_evidence_manifest_v1",
            "files": files, "payload_sha256": payload,
            "create_only": True, "fallback_invocations": self.visibility_fallback_invocations,
        })
        with (self.output_dir / "REFERENCE_RENDER_COMPLETE").open("x", encoding="utf-8") as stream:
            stream.write("PUBLICATION_INSTANCE_REFERENCE_RENDER_COMPLETE\n")

    def stop_sensors(self) -> bool:
        instance_ok = True
        if self.instance_camera is not None:
            try:
                self.instance_camera.stop()
            except RuntimeError:
                instance_ok = False
            try:
                instance_ok = bool(self.instance_camera.destroy()) and instance_ok
            except RuntimeError:
                instance_ok = False
        normal_ok = super().stop_sensors()
        try:
            self._render_references()
        except Exception as exc:
            self.reference_error = f"{type(exc).__name__}: {exc}"
            print(f"publication reference rendering failed: {self.reference_error}", flush=True)
            return False
        return bool(instance_ok and normal_ok)
