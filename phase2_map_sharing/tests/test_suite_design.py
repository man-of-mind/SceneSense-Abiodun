from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from phase2_map_sharing.design_suite_manifest import (
    build_manifest,
    build_power_sensitivity,
    summarize,
    write_design,
)


class SuiteDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.config = yaml.safe_load(
            (
                root
                / "phase2_map_sharing/configs/phase2_suite_ab_design_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.manifest = build_manifest(cls.config)

    def test_suite_labels_cannot_be_inverted(self) -> None:
        labels = dict(self.manifest.groupby("suite_id")["suite_label"].first())
        self.assertEqual(labels["A"], "designed_decision_opportunities")
        self.assertEqual(labels["B"], "naturalistic_operation")

    def test_group_and_trajectory_counts_are_frozen(self) -> None:
        groups = self.manifest.drop_duplicates("group_id")
        self.assertEqual(len(groups[groups.suite_id == "A"]), 120)
        self.assertEqual(len(groups[groups.suite_id == "B"]), 90)
        self.assertEqual(len(self.manifest), 330)

    def test_positive_benign_pair_never_crosses_split(self) -> None:
        suite_a = self.manifest[self.manifest.suite_id == "A"]
        self.assertTrue((suite_a.groupby("group_id").size() == 2).all())
        self.assertTrue((suite_a.groupby("group_id")["split"].nunique() == 1).all())
        self.assertTrue((suite_a.groupby("group_id")["carla_seed"].nunique() == 1).all())

    def test_population_contract_is_explicit_and_suite_specific(self) -> None:
        suite_a = self.manifest[self.manifest.suite_id == "A"]
        suite_b = self.manifest[self.manifest.suite_id == "B"]
        self.assertEqual(set(suite_a.traffic_density), {"not_applicable"})
        self.assertEqual(set(suite_a.traffic_density_status), {"not_applicable"})
        self.assertEqual(
            set(suite_a.ambient_population_mode), {"scenario_owned_only"}
        )
        self.assertEqual(set(suite_a.ambient_population_process_required), {0})
        self.assertEqual(
            set(suite_b.traffic_density_status), {"realized_nuisance_factor"}
        )
        self.assertEqual(set(suite_b.ambient_population_mode), {"naturalistic_tm"})
        self.assertEqual(set(suite_b.ambient_population_process_required), {1})

    def test_each_designed_cell_has_one_one_three_split(self) -> None:
        groups = self.manifest[self.manifest.suite_id == "A"].drop_duplicates(
            "group_id"
        )
        table = groups.groupby(
            [
                "geometry_or_route_id",
                "closing_speed_band",
                "time_to_hazard_band",
                "split",
            ]
        ).size().unstack(fill_value=0)
        self.assertTrue(
            all(
                tuple(int(row[name]) for name in ("calibration", "validation", "test"))
                == (1, 1, 3)
                for _, row in table.iterrows()
            )
        )

    def test_confirmatory_test_has_no_heavy_raw_windows(self) -> None:
        test = self.manifest[self.manifest.split == "test"]
        self.assertTrue((test.raw_window_duration_s == 0.0).all())
        self.assertEqual(set(test.raw_retention_tier), {"causal_lightweight_only"})

    def test_naturalistic_anchor_strata_are_pre_registered_and_balanced(self) -> None:
        naturalistic = self.manifest[
            self.manifest.suite_id == "B"
        ]
        for route_id in (
            "town10hd_opt_signalized_demo_region",
            "town10hd_opt_safe_perimeter",
        ):
            rows = naturalistic[naturalistic.geometry_or_route_id == route_id]
            self.assertEqual(
                set(rows.pair_contract_id),
                {"town10hd_opt_same_lane_helper_ahead_v1"},
            )
            self.assertEqual(
                set(rows.route_start_anchor_id),
                {f"a{index}" for index in range(6)},
            )
            counts = rows.groupby(["route_start_anchor_id", "split"]).size()
            for anchor_id in (f"a{index}" for index in range(6)):
                self.assertEqual(
                    tuple(
                        int(counts.loc[(anchor_id, split)])
                        for split in ("calibration", "validation", "test")
                    ),
                    (1, 1, 3),
                )
            self.assertTrue(rows.recipient_route_sha256.str.len().eq(64).all())
            self.assertTrue(rows.helper_route_sha256.str.len().eq(64).all())

    def test_naturalistic_visual_review_status_is_route_specific(self) -> None:
        signalized = self.manifest[
            self.manifest.geometry_or_route_id
            == "town10hd_opt_signalized_demo_region"
        ]
        perimeter = self.manifest[
            self.manifest.geometry_or_route_id == "town10hd_opt_safe_perimeter"
        ]
        self.assertEqual(set(signalized.geometry_or_route_status), {"reviewed_visual_route"})
        self.assertEqual(set(perimeter.geometry_or_route_status), {"reviewed_visual_route"})

    def test_every_primary_row_locks_explicit_epic_renderer(self) -> None:
        self.assertEqual(set(self.manifest.renderer_quality_level), {"Epic"})
        self.assertEqual(
            set(self.manifest.renderer_server_launch_flag),
            {"-quality-level=Epic"},
        )
        self.assertEqual(set(self.manifest.renderer_contract_role), {"primary"})
        renderer = self.config["common"]["renderer_quality"]
        self.assertEqual(renderer["existing_stress_level"], "Low")
        self.assertFalse(renderer["future_low_collection_authorized"])

    def test_midblock_pedestrian_geometry_is_reviewed_and_route_frozen(self) -> None:
        geometry = next(
            item
            for item in self.config["suite_a"]["geometries"]
            if item["geometry_id"]
            == "parked_van_midblock_occluded_pedestrian"
        )
        self.assertEqual(
            geometry["implementation_status"], "reviewed_visual_geometry"
        )
        self.assertEqual(
            geometry["source_geometry_id"],
            "town10hd_opt_midblock_curbside_van_v1",
        )
        root = Path(__file__).resolve().parents[2]
        self.assertTrue((root / geometry["recipient_route"]).is_file())
        self.assertTrue((root / geometry["helper_route"]).is_file())
        rows = self.manifest[
            self.manifest.geometry_or_route_id
            == "parked_van_midblock_occluded_pedestrian"
        ]
        self.assertEqual(len(rows), 40)
        self.assertEqual(
            set(rows.geometry_or_route_status), {"reviewed_visual_geometry"}
        )

    def test_cross_traffic_vehicle_geometry_is_reviewed_and_route_frozen(self) -> None:
        geometry = next(
            item
            for item in self.config["suite_a"]["geometries"]
            if item["geometry_id"] == "occluded_cross_traffic_vehicle"
        )
        self.assertEqual(
            geometry["implementation_status"], "reviewed_visual_geometry"
        )
        self.assertEqual(
            geometry["source_geometry_id"],
            "town10hd_opt_occluded_cross_traffic_vehicle_v1",
        )
        root = Path(__file__).resolve().parents[2]
        for role in ("recipient", "helper", "target"):
            self.assertTrue((root / geometry[f"{role}_route"]).is_file())
        rows = self.manifest[
            self.manifest.geometry_or_route_id
            == "occluded_cross_traffic_vehicle"
        ]
        self.assertEqual(len(rows), 40)
        self.assertEqual(
            set(rows.geometry_or_route_status), {"reviewed_visual_geometry"}
        )

    def test_parked_pullout_geometry_is_reviewed_and_route_frozen(self) -> None:
        geometry = next(
            item
            for item in self.config["suite_a"]["geometries"]
            if item["geometry_id"] == "parked_vehicle_pullout"
        )
        self.assertEqual(
            geometry["implementation_status"], "reviewed_visual_geometry"
        )
        self.assertEqual(
            geometry["source_geometry_id"],
            "town10hd_opt_parked_vehicle_pullout_v1",
        )
        root = Path(__file__).resolve().parents[2]
        for role in ("recipient", "helper", "target"):
            self.assertTrue((root / geometry[f"{role}_route"]).is_file())
        rows = self.manifest[
            self.manifest.geometry_or_route_id == "parked_vehicle_pullout"
        ]
        self.assertEqual(len(rows), 40)
        self.assertEqual(
            set(rows.geometry_or_route_status), {"reviewed_visual_geometry"}
        )

    def test_queue_reveal_geometry_is_reviewed_and_route_frozen(self) -> None:
        geometry = next(
            item
            for item in self.config["suite_a"]["geometries"]
            if item["geometry_id"] == "queue_reveal_lead_vehicle"
        )
        self.assertEqual(
            geometry["implementation_status"], "reviewed_visual_geometry"
        )
        self.assertEqual(
            geometry["source_geometry_id"],
            "town10hd_opt_queue_reveal_lead_vehicle_v1",
        )
        root = Path(__file__).resolve().parents[2]
        for role in ("recipient", "helper", "occluder"):
            self.assertTrue((root / geometry[f"{role}_route"]).is_file())
        rows = self.manifest[
            self.manifest.geometry_or_route_id == "queue_reveal_lead_vehicle"
        ]
        self.assertEqual(len(rows), 40)
        self.assertEqual(
            set(rows.geometry_or_route_status), {"reviewed_visual_geometry"}
        )

    def test_power_and_storage_are_gated_not_overclaimed(self) -> None:
        power = build_power_sensitivity(self.config, self.manifest)
        summary = summarize(self.config, self.manifest, power)
        self.assertFalse(summary["collection_authorized"])
        self.assertEqual(summary["power_status"], "conditional_on_calibration_simulation_gate")
        self.assertGreater(summary["sensitivity_power_at_sd_1_25_s"], 0.80)
        self.assertTrue(summary["storage_estimate_within_cap"])
        self.assertTrue(summary["blocking_gates"])
        effect = summary["smallest_effect_interpretation"]
        self.assertEqual(effect["sensor_frame_count"], 5)
        self.assertEqual(effect["policy_decision_count"], 10)
        self.assertEqual(
            effect["distance_equivalent_m_by_closing_speed_band"],
            {"low": [1.0, 2.0], "high": [3.0, 5.0]},
        )
        nuisance = summary["warning_nuisance_gate"]
        self.assertEqual(
            nuisance["timing_endpoint_basis"], "registered_target_warning_only"
        )
        self.assertEqual(
            nuisance["aggregation"]["false_warning_episode_rate_per_minute"],
            "sum_false_warning_episodes_over_sum_eligible_benign_exposure_minutes",
        )
        self.assertEqual(
            nuisance["aggregation"]["uncertainty_unit"],
            "paired_trajectory_cluster",
        )
        self.assertEqual(
            nuisance["adjudicated_false_warning_active_frame_rate_max"], 0.10
        )
        self.assertEqual(nuisance["false_warning_episodes_per_minute_max"], 1.0)
        self.assertIn(
            "calibration_absolute_warning_nuisance_gate",
            summary["blocking_gates"],
        )

    def test_manifest_is_deterministic(self) -> None:
        second = build_manifest(self.config)
        self.assertEqual(
            self.manifest.to_csv(index=False), second.to_csv(index=False)
        )

    def test_written_design_is_hashed_and_runtime_unauthorized(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config_path = (
            root / "phase2_map_sharing/configs/phase2_suite_ab_design_v1.yaml"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "design"
            write_design(config_path, output)
            provenance = json.loads(
                (output / "design_provenance.json").read_text(encoding="utf-8")
            )
            artifacts = json.loads(
                (output / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(provenance["runtime_authorized"])
            self.assertEqual(len(artifacts["files"]), 4)


if __name__ == "__main__":
    unittest.main()
