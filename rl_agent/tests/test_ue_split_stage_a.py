from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from rl_agent.ue_split_stage_a import (
    StageAError,
    assemble,
    canonical_profile_id,
    load_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "rl_agent/configs/ue_split_stage_a_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StageAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = Path(tempfile.mkdtemp(prefix="ue-split-stage-a-tests-"))
        cls.output = cls.temp / "review_a"
        cls.result = assemble(
            CONFIG,
            cls.output,
            now=datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp, ignore_errors=True)

    def test_review_bundle_has_exact_expected_shapes(self) -> None:
        expected = {
            "ue_split_evidence_pool.csv": (72, None),
            "ue_split_quality_strata.csv": (648, None),
            "ue_split_quality_floor_sensitivity.csv": (216, None),
            "ue_split_network_regimes.csv": (4, None),
            "ue_split_transport_evidence.csv": (12, None),
            "ue_split_profile_regime_screen.csv": (288, None),
            "ue_split_boundary_candidates.csv": (8, None),
            "ue_split_latency_tolerance_proxy.csv": (32, None),
            "ue_split_staleness_latency_anchors.csv": (6, None),
            "ue_split_staleness_error_sensitivity.csv": (8, None),
        }
        for name, (rows, _) in expected.items():
            self.assertEqual(len(pd.read_csv(self.output / name)), rows, name)

    def test_review_bundle_never_claims_a_frozen_action_catalog(self) -> None:
        review = json.loads((self.output / "REVIEW_REQUIRED.json").read_text())
        manifest = json.loads((self.output / "manifest.json").read_text())
        sensitivity = pd.read_csv(
            self.output / "ue_split_quality_floor_sensitivity.csv"
        )
        self.assertEqual(review["decision_state"], "REVIEW_REQUIRED")
        self.assertIsNone(review["quality_floor_id"])
        self.assertIsNone(review["eligible_action_count"])
        self.assertIsNone(manifest["factor_contract"]["eligible_action_count"])
        self.assertFalse(sensitivity["final_eligible"].any())
        self.assertEqual(
            set(sensitivity["absolute_object_quality_gate_status"]),
            {"UNRESOLVED_NO_SELECTED_ABSOLUTE_FLOOR"},
        )
        self.assertFalse((self.output / "ue_split_action_catalog.csv").exists())
        self.assertFalse((self.output / "ue_split_profile_network_surface.csv").exists())
        self.assertFalse((self.output / "COMPLETED.json").exists())

    def test_historical_network_rows_are_not_direct_ten_hz_map_evidence(self) -> None:
        transport = pd.read_csv(self.output / "ue_split_transport_evidence.csv")
        screen = pd.read_csv(self.output / "ue_split_profile_regime_screen.csv")
        self.assertFalse(transport["target_rate_match"].any())
        self.assertFalse(transport["map_update_done_observed"].any())
        self.assertFalse(transport["directly_measured_at_target_10hz"].any())
        self.assertFalse(screen["directly_measured"].any())
        self.assertEqual(set(screen["network_certification"]), {"UNRESOLVED"})

    def test_segmentation_is_diagnostic_not_an_eligibility_veto(self) -> None:
        catalog = pd.read_csv(self.output / "ue_split_evidence_pool.csv")
        sensitivity = pd.read_csv(
            self.output / "ue_split_quality_floor_sensitivity.csv"
        )
        self.assertEqual(set(catalog["segmentation_role"]), {"secondary_diagnostic"})
        self.assertFalse(sensitivity["segmentation_used_as_veto"].any())

    def test_retained_split_manifests_are_actually_checked_disjoint(self) -> None:
        manifest = json.loads((self.output / "manifest.json").read_text())
        grid_test = {
            item["id"]: item for item in manifest["audit"]["tests"]
        }["A_SPLIT_DISJOINTNESS"]
        self.assertEqual(grid_test["status"], "PASS")
        self.assertEqual((grid_test["train"], grid_test["val"], grid_test["test"]), (10911, 2110, 2162))

    def test_all_measurement_candidates_remain_unauthorized(self) -> None:
        unresolved = pd.read_csv(
            self.output / "ue_split_unresolved_measurements.csv"
        )
        boundaries = pd.read_csv(self.output / "ue_split_boundary_candidates.csv")
        self.assertFalse(unresolved["measurement_authorized"].any())
        self.assertEqual(int(unresolved["new_run_required"].sum()), 1)
        self.assertFalse(boundaries["measurement_authorized"].any())

    def test_manifest_output_seals_and_review_pointer_validate(self) -> None:
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["outputs"]:
            path = self.output / item["path"]
            self.assertEqual(sha256(path), item["sha256"], item["path"])
            if path.suffix == ".csv":
                self.assertEqual(len(pd.read_csv(path)), item["rows"], item["path"])
        review = json.loads((self.output / "REVIEW_REQUIRED.json").read_text())
        self.assertEqual(review["manifest_sha256"], sha256(manifest_path))
        self.assertEqual(
            manifest["repository"]["assembler_sha256"],
            sha256(ROOT / manifest["repository"]["assembler_path"]),
        )

    def test_output_is_create_only(self) -> None:
        with self.assertRaisesRegex(StageAError, "refusing to overwrite"):
            assemble(CONFIG, self.output)

    def test_tables_are_deterministic_for_same_inputs(self) -> None:
        other = self.temp / "review_b"
        assemble(
            CONFIG,
            other,
            now=datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc),
        )
        for name in (
            "ue_split_evidence_pool.csv",
            "ue_split_quality_strata.csv",
            "ue_split_quality_floor_sensitivity.csv",
            "ue_split_network_regimes.csv",
            "ue_split_transport_evidence.csv",
            "ue_split_profile_regime_screen.csv",
            "ue_split_boundary_candidates.csv",
            "ue_split_latency_tolerance_proxy.csv",
            "ue_split_staleness_latency_anchors.csv",
            "ue_split_staleness_error_sensitivity.csv",
            "ue_split_unresolved_measurements.csv",
        ):
            self.assertEqual(sha256(self.output / name), sha256(other / name), name)


class StageAContractTests(unittest.TestCase):
    def test_profile_id_binds_model_knobs_codec_and_checkpoint(self) -> None:
        value = canonical_profile_id(
            "ae32",
            "per_channel_uint4",
            0.9,
            3,
            "10cebbeede4da992e68850d8f38358e89000b62524be25b68c88517d7b58f9b2",
        )
        self.assertEqual(value, "ae32__u4__q0.9__zstd3__ckpt10cebbeede4d")

    def test_config_rejects_any_live_or_freeze_authority(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-stage-a-config-") as directory:
            path = Path(directory) / "bad.yaml"
            config = yaml.safe_load(CONFIG.read_text())
            config["authority"]["new_oai_run"] = True
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            with self.assertRaisesRegex(StageAError, "prohibits authority"):
                load_config(path)

    def test_config_rejects_preselected_quality_floor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-stage-a-config-") as directory:
            path = Path(directory) / "bad.yaml"
            config = yaml.safe_load(CONFIG.read_text())
            config["service_contract"]["quality_floor_id"] = "posthoc_floor"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            with self.assertRaisesRegex(StageAError, "must not preselect"):
                load_config(path)

    def test_assembly_rejects_pinned_input_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-stage-a-config-") as directory:
            path = Path(directory) / "bad.yaml"
            output = Path(directory) / "output"
            config = yaml.safe_load(CONFIG.read_text())
            config["repository_root"] = str(ROOT)
            config["quality_evidence"]["eval_settings_sha256"] = "0" * 64
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            with self.assertRaisesRegex(StageAError, "hash drift"):
                assemble(path, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
