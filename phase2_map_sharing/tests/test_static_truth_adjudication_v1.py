from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from data_collection.phase2_static_environment_truth_v1 import (
    OBJECTS_CSV_NAME,
    capture_static_environment_truth_v1,
)
from phase2_map_sharing.adjudicate_future_hazards import _future_label
from phase2_map_sharing.static_truth_adjudication_v1 import (
    TRUTH_SOURCE_DYNAMIC,
    TRUTH_SOURCE_STATIC,
    TRUTH_SOURCE_UNMATCHED,
    constant_static_future_truth_v1,
    load_trajectory_static_catalogs_v1,
    load_verified_static_catalog_v1,
    match_unmatched_warnings_to_static_v1,
    maybe_load_verified_static_catalog_v1,
)


def _location(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _rotation(yaw: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw)


def _environment_object(
    native_id: int,
    semantic_class: str,
    *,
    x_m: float,
    y_m: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=native_id,
        name=f"static.{semantic_class.lower()}.{native_id}",
        type=semantic_class,
        transform=SimpleNamespace(
            location=_location(x_m, y_m, 0.0),
            rotation=_rotation(7.0),
        ),
        bounding_box=SimpleNamespace(
            location=_location(x_m, y_m, 1.25),
            extent=_location(2.1, 0.95, 1.25),
            rotation=_rotation(7.0),
        ),
    )


class _Map:
    name = "Carla/Maps/Town10HD_Opt"

    def to_opendrive(self) -> str:
        return "<OpenDRIVE><road id='20'/></OpenDRIVE>"


class _World:
    def __init__(self) -> None:
        self.objects = {
            "Car": [_environment_object(11, "Car", x_m=10.0)],
            "Truck": [_environment_object(22, "Truck", x_m=30.0)],
            "Bus": [_environment_object(33, "Bus", x_m=50.0)],
        }

    def get_map(self) -> _Map:
        return _Map()

    def get_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(
            frame=123,
            timestamp=SimpleNamespace(elapsed_seconds=4.5),
        )

    def get_environment_objects(self, label: object) -> list[object]:
        return list(self.objects[str(label)])


def _capture_catalog(root: Path) -> Path:
    static_dir = root / "static_environment_truth"
    capture_static_environment_truth_v1(
        _World(),
        static_dir,
        semantic_labels=["Car", "Truck", "Bus"],
        required_semantic_classes=["Car"],
        enabled_state_by_id={11: True, 22: True, 33: True},
        selection_contract="unit_test_car_truck_bus_catalog",
    )
    return static_dir


class StaticCatalogTests(unittest.TestCase):
    def test_verifies_normalizes_vehicle_classes_and_preserves_obb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = load_verified_static_catalog_v1(
                _capture_catalog(Path(tmp))
            )
        self.assertEqual(set(catalog["class_name"]), {"vehicle"})
        car = catalog[catalog["semantic_class"] == "Car"].iloc[0]
        self.assertAlmostEqual(float(car["length_m"]), 4.2)
        self.assertAlmostEqual(float(car["width_m"]), 1.9)
        self.assertAlmostEqual(float(car["height_m"]), 2.5)
        self.assertEqual(car["truth_source"], TRUTH_SOURCE_STATIC)

    def test_existing_tampered_catalog_fails_and_absent_catalog_is_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(maybe_load_verified_static_catalog_v1(root))
            static_dir = _capture_catalog(root)
            with (static_dir / OBJECTS_CSV_NAME).open("a", encoding="utf-8") as stream:
                stream.write("tamper\n")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                maybe_load_verified_static_catalog_v1(root)


class StaticRequirementTests(unittest.TestCase):
    def test_declared_enabled_missing_catalog_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            (batch / "resolved_config.yaml").write_text(
                "static_environment_truth:\n  enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "declared-required static environment truth"
            ):
                load_trajectory_static_catalogs_v1(batch, ["trajectory_001"])

    def test_decision_pilot_stage_missing_catalog_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            (batch / "batch_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "scenesense.phase2_calibration_audit_batch.v1",
                        "stage_id": "phase2_decision_opportunity_pilot_v1",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "declared-required static environment truth"
            ):
                load_trajectory_static_catalogs_v1(batch, ["trajectory_001"])

    def test_declared_tampered_catalog_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            static_dir = _capture_catalog(batch / "trajectory_001")
            with (static_dir / OBJECTS_CSV_NAME).open("a", encoding="utf-8") as stream:
                stream.write("tamper\n")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                load_trajectory_static_catalogs_v1(
                    batch,
                    ["trajectory_001"],
                    declared_sources=(
                        (
                            "integration_config",
                            {"static_environment_truth": {"enabled": True}},
                        ),
                    ),
                )

    def test_declared_present_catalog_is_verified_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            _capture_catalog(batch / "trajectory_001")
            catalogs, requirement = load_trajectory_static_catalogs_v1(
                batch,
                ["trajectory_001"],
                declared_sources=(
                    (
                        "integration_config",
                        {"pilot_provenance": {"schema": "unit-test-pilot"}},
                    ),
                ),
            )
        self.assertTrue(requirement["required"])
        self.assertEqual(set(catalogs), {"trajectory_001"})
        self.assertEqual(set(catalogs["trajectory_001"]["class_name"]), {"vehicle"})

    def test_undeclared_historical_batch_remains_actor_only_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalogs, requirement = load_trajectory_static_catalogs_v1(
                Path(tmp),
                ["historical_trajectory"],
                declared_sources=(("historical", {"static_environment_truth": {}}),),
            )
        self.assertFalse(requirement["required"])
        self.assertTrue(requirement["historical_actor_only_compatibility"])
        self.assertEqual(catalogs, {})


class StaticMatchingTests(unittest.TestCase):
    @staticmethod
    def warning(class_name: str = "vehicle") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "class_name": class_name,
                    "track_world_x": 10.0,
                    "track_world_y": 0.0,
                }
            ],
            index=[7],
        )

    def test_dynamic_match_has_precedence_over_closer_static_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = load_verified_static_catalog_v1(
                _capture_catalog(Path(tmp))
            )
        dynamic = {
            7: {
                "current_truth_matched": 1,
                "current_truth_actor_id": "dynamic-42",
                "current_truth_role_name": "ambient_vehicle",
                "current_truth_distance_m": 1.0,
            }
        }
        matched = match_unmatched_warnings_to_static_v1(
            self.warning(), dynamic, catalog, gate_m=5.0
        )[7]
        self.assertEqual(matched["current_truth_actor_id"], "dynamic-42")
        self.assertEqual(matched["truth_source"], TRUTH_SOURCE_DYNAMIC)
        self.assertIsNone(matched["current_truth_static_environment_object_id"])

    def test_class_mismatch_does_not_match_static_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = load_verified_static_catalog_v1(
                _capture_catalog(Path(tmp))
            )
        dynamic = {
            7: {
                "current_truth_matched": 0,
                "current_truth_actor_id": None,
                "current_truth_role_name": None,
                "current_truth_distance_m": None,
            }
        }
        matched = match_unmatched_warnings_to_static_v1(
            self.warning("pedestrian"), dynamic, catalog, gate_m=5.0
        )[7]
        self.assertEqual(matched["current_truth_matched"], 0)
        self.assertEqual(matched["truth_source"], TRUTH_SOURCE_UNMATCHED)

    def test_carla_car_matches_only_one_unmatched_vehicle_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = load_verified_static_catalog_v1(
                _capture_catalog(Path(tmp))
            )
        warnings = pd.DataFrame(
            [
                {
                    "class_name": "vehicle",
                    "track_world_x": 10.0,
                    "track_world_y": 0.0,
                },
                {
                    "class_name": "vehicle",
                    "track_world_x": 10.1,
                    "track_world_y": 0.0,
                },
            ],
            index=[7, 8],
        )
        dynamic = {
            index: {
                "current_truth_matched": 0,
                "current_truth_actor_id": None,
                "current_truth_role_name": None,
                "current_truth_distance_m": None,
            }
            for index in warnings.index
        }
        matches = match_unmatched_warnings_to_static_v1(
            warnings, dynamic, catalog, gate_m=1.0
        )
        matched = [row for row in matches.values() if row["current_truth_matched"]]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["truth_source"], TRUTH_SOURCE_STATIC)
        self.assertEqual(
            matched[0]["current_truth_static_environment_object_id"],
            matched[0]["current_truth_actor_id"],
        )


class StaticFutureTruthTests(unittest.TestCase):
    def test_same_static_match_can_be_future_safe_or_hazardous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = load_verified_static_catalog_v1(
                _capture_catalog(Path(tmp))
            )
        actor_id = str(
            catalog[catalog["semantic_class"] == "Car"].iloc[0]["actor_id"]
        )
        frame_times = pd.DataFrame(
            {
                "frame_id": list(range(51)),
                "carla_timestamp": [index / 10.0 for index in range(51)],
            }
        )
        truth = constant_static_future_truth_v1(
            catalog, actor_id=actor_id, frame_times=frame_times
        )
        self.assertTrue((truth["bbox_extent_x_m"] == 2.1).all())
        self.assertTrue((truth["length_m"] == 4.2).all())
        safe_ego = pd.DataFrame(
            {
                "frame_id": list(range(51)),
                "recipient_x": [0.0] * 51,
                "recipient_y": [0.0] * 51,
                "recipient_yaw_deg": [0.0] * 51,
                "recipient_speed_mps": [0.0] * 51,
            }
        )
        hazard_ego = safe_ego.copy()
        hazard_ego.loc[20:, "recipient_x"] = 8.0
        safe = _future_label(
            {"warning_at_s": 0.0},
            truth,
            safe_ego,
            horizon_s=5.0,
            safety_radius_m=3.0,
            cadence_s=0.1,
            ego_dimensions=None,
        )
        hazard = _future_label(
            {"warning_at_s": 0.0},
            truth,
            hazard_ego,
            horizon_s=5.0,
            safety_radius_m=3.0,
            cadence_s=0.1,
            ego_dimensions=None,
        )
        self.assertEqual(safe["future_label"], "truth_hazard_negative")
        self.assertEqual(safe["false_warning"], 1)
        self.assertEqual(hazard["future_label"], "truth_hazard_positive")
        self.assertEqual(hazard["false_warning"], 0)


if __name__ == "__main__":
    unittest.main()
