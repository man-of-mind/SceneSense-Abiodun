from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import carla

from data_collection.phase2_static_environment_truth_v1 import (
    MANIFEST_JSON_NAME,
    OBJECTS_CSV_NAME,
)
from data_collection.run_phase2_calibration_audit import (
    STATIC_ENVIRONMENT_ENABLED_STATE_BASIS,
    STATIC_ENVIRONMENT_SEMANTIC_HASH_BASIS,
    STATIC_ENVIRONMENT_SELECTION_CONTRACT,
    _capture_static_environment_truth_before_dynamic_actors,
    _compare_static_environment_records,
    _sha256,
    _static_environment_truth_config,
)


def _vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _rotation(yaw: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw)


def _environment_object(
    native_id: int,
    semantic_label: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=native_id,
        name=f"static.{str(semantic_label).lower()}.{native_id}",
        type=semantic_label,
        transform=SimpleNamespace(
            location=_vector(float(native_id), 2.0, 0.2),
            rotation=_rotation(5.0),
        ),
        bounding_box=SimpleNamespace(
            location=_vector(float(native_id), 2.0, 0.9),
            extent=_vector(2.0, 0.9, 0.8),
            rotation=_rotation(5.0),
        ),
    )


class _Actors(list):
    def filter(self, pattern: str) -> list[object]:
        if pattern == "vehicle.*":
            return [item for item in self if str(item).startswith("vehicle.")]
        return []


class _Map:
    name = "Carla/Maps/Town10HD_Opt"

    def to_opendrive(self) -> str:
        return "<OpenDRIVE><road id='20'/></OpenDRIVE>"


class _World:
    def __init__(
        self,
        dynamic_actors: list[object] | None = None,
        *,
        frame: int = 101,
        timestamp_s: float = 1.25,
    ) -> None:
        self.dynamic_actors = _Actors(dynamic_actors or [])
        self.frame = int(frame)
        self.timestamp_s = float(timestamp_s)
        self.queries: list[str] = []
        self.objects = {
            "Car": [_environment_object(11, carla.CityObjectLabel.Car)],
            "Truck": [_environment_object(22, carla.CityObjectLabel.Truck)],
            "Bus": [_environment_object(33, carla.CityObjectLabel.Bus)],
        }

    def get_actors(self) -> _Actors:
        return self.dynamic_actors

    def get_environment_objects(self, label: object) -> list[object]:
        name = str(label)
        self.queries.append(name)
        return list(self.objects[name])

    def get_map(self) -> _Map:
        return _Map()

    def get_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(
            frame=self.frame,
            timestamp=SimpleNamespace(elapsed_seconds=self.timestamp_s),
        )


def _config() -> dict:
    return {
        "carla": {
            "expected_town": "Town10HD_Opt",
            "reload_world_before_trajectory": True,
        },
        "static_environment_truth": {
            "enabled": True,
            "semantic_labels": ["Car", "Truck", "Bus"],
            "required_semantic_classes": ["Car"],
            "selection_contract": STATIC_ENVIRONMENT_SELECTION_CONTRACT,
            "enabled_state_basis": STATIC_ENVIRONMENT_ENABLED_STATE_BASIS,
        },
    }


class Phase2CalibrationStaticTruthIntegrationTests(unittest.TestCase):
    def test_absent_block_preserves_historical_audit_semantics(self) -> None:
        self.assertIsNone(_static_environment_truth_config({}))

    def test_contract_rejects_catalog_broadening_and_nonfresh_world(self) -> None:
        config = _config()
        config["static_environment_truth"]["semantic_labels"].append("Other")
        with self.assertRaisesRegex(ValueError, "exactly Car, Truck, Bus"):
            _static_environment_truth_config(config)

        config = _config()
        config["carla"]["reload_world_before_trajectory"] = False
        with self.assertRaisesRegex(ValueError, "fresh world reload"):
            _static_environment_truth_config(config)

    def test_capture_seals_path_hash_result_and_all_enabled_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_dir = Path(tmp) / "trajectory"
            trajectory_dir.mkdir()
            world = _World()

            record = _capture_static_environment_truth_before_dynamic_actors(
                world,
                trajectory_dir,
                _config(),
            )

            self.assertIsNotNone(record)
            assert record is not None
            static_dir = trajectory_dir / record["path"]
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["capture_result"]["verdict"], "PASS")
            self.assertEqual(record["capture_result"]["object_count"], 3)
            self.assertEqual(
                record["artifact_manifest_sha256"],
                _sha256(static_dir / MANIFEST_JSON_NAME),
            )
            self.assertEqual(len(record["static_geometry_semantic_sha256"]), 64)
            self.assertEqual(record["environment_object_toggle_calls_before_snapshot"], 0)
            self.assertEqual(record["queried_object_counts"], {
                "Car": 1,
                "Truck": 1,
                "Bus": 1,
            })
            # One inventory pass constructs the explicit registry and the
            # capture module independently re-queries the same frozen catalog.
            self.assertEqual(
                world.queries,
                ["Car", "Truck", "Bus", "Car", "Truck", "Bus"],
            )
            with (static_dir / OBJECTS_CSV_NAME).open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["enabled"] for row in rows}, {"true"})
            self.assertEqual(
                {row["semantic_class"] for row in rows},
                {"Car", "Truck", "Bus"},
            )

    def test_semantic_hash_is_independent_of_capture_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_dir = Path(tmp) / "first"
            second_dir = Path(tmp) / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = _capture_static_environment_truth_before_dynamic_actors(
                _World(frame=101, timestamp_s=1.25),
                first_dir,
                _config(),
            )
            second = _capture_static_environment_truth_before_dynamic_actors(
                _World(frame=999, timestamp_s=88.0),
                second_dir,
                _config(),
            )
            assert first is not None and second is not None
            self.assertEqual(
                first["static_geometry_semantic_sha256"],
                second["static_geometry_semantic_sha256"],
            )

    def test_matched_pair_gate_uses_semantic_not_artifact_hash(self) -> None:
        def record(semantic_hash: str, artifact_hash: str) -> dict:
            return {
                "status": "complete",
                "selection_contract": STATIC_ENVIRONMENT_SELECTION_CONTRACT,
                "static_geometry_semantic_hash_basis": (
                    STATIC_ENVIRONMENT_SEMANTIC_HASH_BASIS
                ),
                "static_geometry_semantic_sha256": semantic_hash,
                "artifact_manifest_sha256": artifact_hash,
            }

        shared = "a" * 64
        result = _compare_static_environment_records(
            record(shared, "1" * 64),
            record(shared, "2" * 64),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result["pass"])

        drift = _compare_static_environment_records(
            record(shared, "1" * 64),
            record("b" * 64, "2" * 64),
        )
        assert drift is not None
        self.assertFalse(drift["pass"])
        self.assertIn("static_geometry_semantic_drift", drift["failures"])

        missing = _compare_static_environment_records(record(shared, "1" * 64), None)
        assert missing is not None
        self.assertFalse(missing["pass"])
        self.assertIn("missing_static_environment_record", missing["failures"])

    def test_capture_fails_before_writing_if_any_dynamic_actor_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_dir = Path(tmp) / "trajectory"
            trajectory_dir.mkdir()
            with self.assertRaisesRegex(RuntimeError, "precede every dynamic actor"):
                _capture_static_environment_truth_before_dynamic_actors(
                    _World(["vehicle.ego"]),
                    trajectory_dir,
                    _config(),
                )
            self.assertFalse((trajectory_dir / "static_environment_truth").exists())


if __name__ == "__main__":
    unittest.main()
