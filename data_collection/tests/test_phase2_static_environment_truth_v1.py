from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from data_collection.phase2_static_environment_truth_v1 import (
    OBJECTS_CSV_NAME,
    SNAPSHOT_JSON_NAME,
    capture_static_environment_truth_v1,
    stable_environment_object_id_v1,
    verify_static_environment_truth_v1,
)


def vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def rotation(roll: float, pitch: float, yaw: float) -> SimpleNamespace:
    return SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw)


def environment_object(
    native_id: int,
    semantic_class: str,
    *,
    x: float,
    name: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=native_id,
        name=name,
        type=SimpleNamespace(name=semantic_class),
        transform=SimpleNamespace(
            location=vector(x, 2.0, 0.25),
            rotation=rotation(0.0, 0.0, 15.0),
        ),
        bounding_box=SimpleNamespace(
            location=vector(x + 0.1, 2.1, 0.9),
            extent=vector(2.1, 0.9, 0.8),
            rotation=rotation(0.0, 1.0, 16.0),
        ),
    )


class _Map:
    name = "Carla/Maps/Town10HD_Opt"

    def to_opendrive(self) -> str:
        return "<OpenDRIVE><road id='20'/></OpenDRIVE>"


class _World:
    def __init__(self, objects_by_label: dict[str, list[object]]) -> None:
        self.objects_by_label = objects_by_label
        self.environment_queries: list[str] = []
        self.actor_api_called = False

    def get_map(self) -> _Map:
        return _Map()

    def get_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(
            frame=456,
            timestamp=SimpleNamespace(elapsed_seconds=12.5),
        )

    def get_environment_objects(self, label: str) -> list[object]:
        self.environment_queries.append(label)
        return list(self.objects_by_label[label])

    def get_actors(self) -> None:
        self.actor_api_called = True
        raise AssertionError("dynamic actor API must not enter the static snapshot")


class StaticEnvironmentTruthV1Tests(unittest.TestCase):
    def _world(self) -> _World:
        return _World(
            {
                "Vehicles": [
                    environment_object(22, "Vehicles", x=5.0, name="parked.taxi"),
                    environment_object(11, "Vehicles", x=1.0, name="parked.van"),
                ],
                "Other": [
                    environment_object(90, "Other", x=8.0, name="bus.stop.prop")
                ],
            }
        )

    def _capture(self, output: Path, world: _World | None = None) -> dict:
        return capture_static_environment_truth_v1(
            world or self._world(),
            output,
            semantic_labels=("Vehicles", "Other"),
            required_semantic_classes=("Vehicles", "Other"),
            enabled_state_by_id={11: True, 22: False, 90: True},
            selection_contract="town10_static_vehicle_obstacle_props.v1",
        )

    def test_capture_is_create_only_hashed_and_excludes_dynamic_actors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            world = self._world()

            result = self._capture(output, world)
            verified = verify_static_environment_truth_v1(
                output,
                expected_map_name="Carla/Maps/Town10HD_Opt",
                expected_map_sha256=result["map_sha256"],
                expected_capture_frame_id=456,
                expected_selection_contract=(
                    "town10_static_vehicle_obstacle_props.v1"
                ),
                expected_required_semantic_classes=("Vehicles", "Other"),
            )

            self.assertEqual(result["verdict"], "PASS")
            self.assertEqual(verified["verdict"], "PASS")
            self.assertEqual(verified["object_count"], 3)
            self.assertEqual(world.environment_queries, ["Vehicles", "Other"])
            self.assertFalse(world.actor_api_called)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {OBJECTS_CSV_NAME, SNAPSHOT_JSON_NAME, "artifact_manifest.json"},
            )

            with (output / OBJECTS_CSV_NAME).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [int(row["carla_environment_object_id"]) for row in rows],
                [11, 22, 90],
            )
            self.assertEqual([row["enabled"] for row in rows], ["true", "false", "true"])
            self.assertEqual(
                rows[0]["environment_object_id"],
                stable_environment_object_id_v1(result["map_sha256"], 11),
            )
            self.assertEqual(rows[0]["bbox_coordinate_frame"], "carla_world_as_returned")

    def test_stable_id_depends_only_on_map_hash_and_native_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            self._capture(first)
            reversed_world = self._world()
            reversed_world.objects_by_label["Vehicles"].reverse()
            self._capture(second, reversed_world)

            def ids(path: Path) -> list[str]:
                with (path / OBJECTS_CSV_NAME).open(newline="", encoding="utf-8") as stream:
                    return [row["environment_object_id"] for row in csv.DictReader(stream)]

            self.assertEqual(ids(first), ids(second))

    def test_existing_output_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            output.mkdir()
            marker = output / "owned.txt"
            marker.write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                self._capture(output)

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_missing_enabled_state_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            with self.assertRaisesRegex(ValueError, "enabled state"):
                capture_static_environment_truth_v1(
                    self._world(),
                    output,
                    semantic_labels=("Vehicles", "Other"),
                    required_semantic_classes=("Vehicles", "Other"),
                    enabled_state_by_id={11: True, 22: False},
                    selection_contract="town10_static_vehicle_obstacle_props.v1",
                )
            self.assertFalse(output.exists())

    def test_duplicate_native_id_across_queries_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            world = self._world()
            world.objects_by_label["Other"].append(
                environment_object(11, "Other", x=99.0, name="duplicate")
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self._capture(output, world)
            self.assertFalse(output.exists())

    def test_invalid_oriented_bbox_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            world = self._world()
            world.objects_by_label["Other"][0].bounding_box.extent.x = 0.0
            with self.assertRaisesRegex(ValueError, "extent"):
                self._capture(output, world)
            self.assertFalse(output.exists())

    def test_verifier_rejects_tampering_and_unmanifested_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            self._capture(output)
            with (output / OBJECTS_CSV_NAME).open("a", encoding="utf-8") as stream:
                stream.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_static_environment_truth_v1(output)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            self._capture(output)
            (output / "unexpected.txt").write_text("not manifested", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                verify_static_environment_truth_v1(output)

    def test_verifier_rejects_wrong_expected_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            self._capture(output)
            with self.assertRaisesRegex(ValueError, "map name"):
                verify_static_environment_truth_v1(
                    output,
                    expected_map_name="Carla/Maps/Town01",
                )
            with self.assertRaisesRegex(ValueError, "selection contract"):
                verify_static_environment_truth_v1(
                    output,
                    expected_selection_contract="wrong_selection.v1",
                )

    def test_snapshot_declares_dynamic_truth_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "static_truth"
            self._capture(output)
            snapshot = json.loads((output / SNAPSHOT_JSON_NAME).read_text(encoding="utf-8"))
            self.assertFalse(snapshot["dynamic_actor_truth"]["included"])
            self.assertEqual(
                snapshot["dynamic_actor_truth"]["contract"],
                "separate_per_frame_actor_origin_stream",
            )


if __name__ == "__main__":
    unittest.main()
