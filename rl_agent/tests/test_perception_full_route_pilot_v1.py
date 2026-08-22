#!/usr/bin/env python3
"""Offline contract tests; these tests never import or start CARLA."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from abiodun.rl_agent import perception_full_route_pilot_v1 as scaffold


PROJECT = Path(__file__).resolve().parents[3]
RL_AGENT = PROJECT / "abiodun" / "rl_agent"
COLLECTION_CONFIG = RL_AGENT / "configs" / "perception_full_route_collection_v1.json"


def synthetic_route() -> dict:
    """Synthetic contract geometry for validation tests, never a Route B proposal."""
    return {
        "schema": "scenesense.canonical_route.v1",
        "route_id": "unit-test-route-b-not-for-collection",
        "version": "1.0.0-test",
        "map": {"town": "Town10HD", "map_identifier": "Town10HD_SYNTHETIC_TEST_ONLY"},
        "coordinate_frame": {
            "name": "carla_world",
            "units": "meters",
            "handedness": "left_handed",
            "axes": {"x": "forward", "y": "right", "z": "up"},
        },
        "route_direction": {"direction": "forward", "waypoint_order": "travel_order"},
        "waypoints": [
            {"sequence_index": 0, "x": 0.0, "y": 0.0, "z": 0.0, "route_segment_id": "synthetic-segment"},
            {"sequence_index": 1, "x": 3.0, "y": 0.0, "z": 0.0, "route_segment_id": "synthetic-segment"},
            {"sequence_index": 2, "x": 0.0, "y": 0.0, "z": 0.0, "route_segment_id": "synthetic-segment"},
        ],
        "segments": [{"segment_id": "synthetic-segment", "start_waypoint_index": 0, "end_waypoint_index": 2}],
        "ego_spawn_transform": {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "route_start_waypoint_index": 0,
            "route_start_position_tolerance_m": 0.1,
        },
        "loop_closure": {
            "mode": "closed_loop",
            "close_last_to_first": True,
            "position_tolerance_m": 0.1,
            "heading_tolerance_deg": 5.0,
        },
        "route_length": {
            "declared_m": 6.0,
            "calculation": "polyline_waypoints_with_optional_loop_seam",
            "measurement_tolerance_m": 0.001,
        },
        "expected_duration": {"minimum_s": 1.0, "nominal_s": 2.0, "maximum_s": 3.0},
        "qualification": {
            "status": "QUALIFIED",
            "qualified_by": "unit-test-only",
            "qualified_at_utc": "2026-08-21T00:00:00Z",
            "qualification_bundle_id": "synthetic-unit-test-bundle",
            "qualification_manifest_sha256": "a" * 64,
        },
    }


def split_record(episode: str, frame: str, bundle: str, region: str = "region-1") -> dict:
    seed_number = int(bundle.split("-")[-1])
    return {
        "schema": "scenesense.perception_split_input.v1",
        "episode_id": episode,
        "frame_id": frame,
        "episode_status": "COMPLETE",
        "seed": {
            "seed_bundle_id": bundle,
            "carla_seed": seed_number,
            "traffic_manager_seed": seed_number + 1000,
        },
        "route_id": "canonical-route-b",
        "route_sha256": "b" * 64,
        "route_region_id": region,
    }


class CanonicalRouteContractTests(unittest.TestCase):
    def test_valid_route_and_external_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            route_path = Path(temp) / "canonical_route_b.json"
            route_path.write_text(json.dumps(synthetic_route(), sort_keys=True), encoding="utf-8")
            digest = hashlib.sha256(route_path.read_bytes()).hexdigest()
            _, summary = scaffold.verify_canonical_route(route_path, digest)
        self.assertEqual(summary["status"], "CANONICAL_ROUTE_VERIFIED")
        self.assertEqual(summary["route_file_sha256"], digest)
        self.assertEqual(summary["computed_route_length_m"], 6.0)

    def test_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            route_path = Path(temp) / "canonical_route_b.json"
            route_path.write_text(json.dumps(synthetic_route()), encoding="utf-8")
            with self.assertRaisesRegex(scaffold.WaitingForCanonicalRoute, "hash drift"):
                scaffold.verify_canonical_route(route_path, "0" * 64)

    def test_missing_hash_and_file_fail_closed(self) -> None:
        with self.assertRaises(scaffold.WaitingForCanonicalRoute):
            scaffold.verify_canonical_route(None, None)
        with self.assertRaises(scaffold.WaitingForCanonicalRoute):
            scaffold.verify_canonical_route("missing-route-b.json", "0" * 64)

    def test_missing_required_field_is_rejected(self) -> None:
        route = synthetic_route()
        del route["ego_spawn_transform"]
        with self.assertRaisesRegex(scaffold.ContractError, "ego_spawn_transform"):
            scaffold.validate_route_document(route)

    def test_bad_loop_and_length_are_rejected(self) -> None:
        route = synthetic_route()
        route["loop_closure"]["position_tolerance_m"] = 0.0
        route["waypoints"][-1]["x"] = 0.5
        with self.assertRaisesRegex(scaffold.ContractError, "endpoint separation"):
            scaffold.validate_route_document(route)
        route = synthetic_route()
        route["route_length"]["declared_m"] = 100.0
        with self.assertRaisesRegex(scaffold.ContractError, "declared route length"):
            scaffold.validate_route_document(route)

    def test_cli_missing_route_returns_waiting_without_carla(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = scaffold.main(["route-preflight"])
        self.assertEqual(status, 3)
        self.assertIn(scaffold.TERMINAL_WAITING, output.getvalue())
        self.assertNotIn("carla", scaffold.__dict__)


class CreateOnlyAndMatrixTests(unittest.TestCase):
    def test_create_only_directory_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = scaffold.create_only_directory(root / "new")
            with self.assertRaises(scaffold.CreateOnlyError):
                scaffold.create_only_directory(directory)
            target = scaffold.write_create_only(directory / "record", b"one\n")
            with self.assertRaises(scaffold.CreateOnlyError):
                scaffold.write_create_only(target, b"two\n")
            self.assertEqual(target.read_bytes(), b"one\n")

    def test_dry_run_matrix_is_deterministic_and_not_ue_multiplied(self) -> None:
        config = json.loads(COLLECTION_CONFIG.read_text(encoding="utf-8"))
        first = scaffold.build_dry_run_matrix(config)
        second = scaffold.build_dry_run_matrix(copy.deepcopy(config))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 27)
        self.assertEqual({row["density_profile"] for row in first}, {"low", "medium", "dense"})
        self.assertTrue(all(row["route_id"] is None and row["status"] == scaffold.TERMINAL_WAITING for row in first))
        forbidden = ("ue_action", "network_profile", "model_family", "quantization", "roi")
        self.assertTrue(all(not any(key in row for key in forbidden) for row in first))

    def test_actor_cleanup_order_is_deterministic_reverse_spawn(self) -> None:
        actors = [
            {"actor_id": 9, "spawn_index": 0},
            {"actor_id": 2, "spawn_index": 2},
            {"actor_id": 5, "spawn_index": 1},
        ]
        self.assertEqual(scaffold.owned_actor_cleanup_order(actors), [2, 5, 9])
        self.assertEqual(scaffold.owned_actor_cleanup_order(reversed(actors)), [2, 5, 9])

    def test_readiness_bundle_is_waiting_and_create_only(self) -> None:
        paths = scaffold._project_paths()
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp)
            bundle, digest = scaffold.emit_readiness_bundle(
                output_root,
                "20260821_000000_UTC",
                paths["collection"],
                paths["pilot"],
                paths["evaluation"],
                paths["schemas"],
                paths["sources"],
            )
            self.assertEqual(len(digest), 64)
            self.assertEqual(json.loads((bundle / "PREFLIGHT.json").read_text())["terminal"], scaffold.TERMINAL_WAITING)
            self.assertEqual(len(json.loads((bundle / "DRY_RUN_MATRIX.json").read_text())["rows"]), 27)
            self.assertTrue((bundle / "REVIEW_REQUIRED").is_file())
            with self.assertRaises(scaffold.CreateOnlyError):
                scaffold.emit_readiness_bundle(
                    output_root,
                    "20260821_000000_UTC",
                    paths["collection"],
                    paths["pilot"],
                    paths["evaluation"],
                    paths["schemas"],
                    paths["sources"],
                )


class LeakageSafeSplitTests(unittest.TestCase):
    def test_complete_episode_seed_split_and_final_test_lock(self) -> None:
        records = [
            split_record("train-episode", "0001", "seed-101"),
            split_record("train-episode", "0002", "seed-101"),
            split_record("val-episode", "0001", "seed-127"),
            split_record("test-episode", "0001", "seed-137"),
        ]
        manifest = scaffold.split_episode_records(
            records,
            {"train": ["seed-101"], "validation": ["seed-127"], "final_test": ["seed-137"]},
        )
        self.assertFalse(manifest["leakage_detected"])
        self.assertEqual(manifest["splits"]["train"]["episode_ids"], ["train-episode"])
        self.assertEqual(manifest["splits"]["final_test"]["access_state"], "LOCKED_UNTOUCHED")
        for pair in manifest["overlap_checks"].values():
            self.assertTrue(all(not values for values in pair.values()))

    def test_same_episode_cannot_cross_seed_or_split(self) -> None:
        records = [
            split_record("crossed", "0001", "seed-101"),
            split_record("crossed", "0002", "seed-127"),
        ]
        with self.assertRaisesRegex(scaffold.ContractError, "crosses"):
            scaffold.split_episode_records(
                records,
                {"train": ["seed-101"], "validation": ["seed-127"], "final_test": ["seed-137"]},
            )

    def test_incomplete_episode_is_rejected(self) -> None:
        record = split_record("incomplete", "0001", "seed-101")
        record["episode_status"] = "INCOMPLETE"
        with self.assertRaisesRegex(scaffold.ContractError, "incomplete episode"):
            scaffold.split_episode_records(
                [record],
                {"train": ["seed-101"], "validation": ["seed-127"], "final_test": ["seed-137"]},
            )

    def test_route_region_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(scaffold.ContractError, "multiple splits"):
            scaffold.split_route_region_records(
                [split_record("episode", "0001", "seed-101", "region-x")],
                {"train": ["region-x"], "validation": ["region-x"], "final_test": ["region-y"]},
            )

    def test_final_test_authorization_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(scaffold.ContractError, "locked"):
            scaffold.require_test_evaluation_authorization(
                None,
                dataset_manifest_sha256="c" * 64,
                pilot_manifest_sha256="d" * 64,
            )


class FrozenDocumentTests(unittest.TestCase):
    def test_all_json_configs_and_schemas_parse(self) -> None:
        paths = scaffold._project_paths()
        documents = [paths["collection"], paths["pilot"], paths["evaluation"], *paths["schemas"]]
        for path in documents:
            with self.subTest(path=path):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_ae64_checkpoint_policy_never_uses_best_pt(self) -> None:
        pilot = json.loads((RL_AGENT / "configs" / "perception_ae64_pilot_v1.json").read_text(encoding="utf-8"))
        output = pilot["training_freeze"]["checkpoint_output"]
        self.assertNotEqual(output["filename_template"], "best.pt")
        self.assertIn("best.pt", output["forbidden_filenames"])
        self.assertEqual(pilot["scope"]["authorized_family"], "AE64")
        self.assertFalse(pilot["starting_evidence"]["untouched_test_evaluated"])


if __name__ == "__main__":
    unittest.main()
