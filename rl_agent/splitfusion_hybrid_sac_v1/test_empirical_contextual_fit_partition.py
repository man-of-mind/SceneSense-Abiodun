"""CPU-only tests for the isolated empirical fit-partition contract."""

from __future__ import annotations

import hashlib
import random
import unittest
from dataclasses import FrozenInstanceError, replace

from . import empirical_contextual_fit_partition as partition
from .action_contract import CATALOG_SHA256
from .empirical_contextual_contract import (
    MODELED_SMOKE_SUPPORT_SHA256,
    PILOT_UTILITY_SPEC_SHA256,
)
from .empirical_quality_surface import (
    DATABASE_FILE_SHA256,
    REWARD_SPEC_FILE_SHA256,
    SELECTION_FILE_SHA256,
)
from .empirical_radio_context import CALIBRATION_SHA256
from .state_reward_transition_contract import POLICY_FEATURE_ORDER


class EmpiricalContextualFitPartitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = partition.load_registered_empirical_fit_partition()

    def test_exact_scene_and_radio_counts(self) -> None:
        self.assertEqual(
            dict(self.contract.scene_counts),
            {"train": 391, "fit_validation": 85},
        )
        self.assertEqual(
            dict(self.contract.radio_counts),
            {"train": 319, "fit_validation": 80},
        )
        self.assertEqual(len(self.contract.scene_assignments), 476)
        self.assertEqual(len(self.contract.radio_assignments), 399)
        self.assertEqual(len(self.contract.scene_blocks), 60)
        self.assertEqual(len(self.contract.radio_blocks), 40)
        self.assertEqual(
            {
                split: sum(block.split == split for block in self.contract.scene_blocks)
                for split in ("train", "fit_validation")
            },
            {"train": 49, "fit_validation": 11},
        )
        self.assertEqual(
            {
                split: sum(block.split == split for block in self.contract.radio_blocks)
                for split in ("train", "fit_validation")
            },
            {"train": 32, "fit_validation": 8},
        )

    def test_scene_rule_is_independently_reproduced(self) -> None:
        for row in self.contract.scene_assignments:
            self.assertEqual(row.temporal_block, row.frame_id // 100)
            domain = (
                f"splitfusion.empirical_fit_partition.v1:{row.temporal_block}"
            ).encode("ascii")
            residue = int.from_bytes(hashlib.sha256(domain).digest(), "big") % 5
            expected = "fit_validation" if residue == 0 else "train"
            self.assertEqual(row.split, expected)

    def test_radio_rule_and_per_profile_counts(self) -> None:
        expected = {
            "ADVERSE_STABLE": {"train": 79, "fit_validation": 20},
            "FADE_RECOVERY": {"train": 80, "fit_validation": 20},
            "FAVORABLE_STABLE": {"train": 80, "fit_validation": 20},
            "MID_VARIABLE": {"train": 80, "fit_validation": 20},
        }
        self.assertEqual(
            {
                profile: dict(counts)
                for profile, counts in self.contract.radio_profile_counts.items()
            },
            expected,
        )
        for row in self.contract.radio_assignments:
            self.assertEqual(row.trace_step_block, row.trace_step_index // 10)
            self.assertEqual(
                row.split,
                "fit_validation" if row.trace_step_block % 5 == 0 else "train",
            )
        # CSV physical line 270 is the sole joint-invalid row and is excluded.
        self.assertNotIn(
            270, {row.csv_row_number for row in self.contract.radio_assignments}
        )

    def test_every_identity_appears_exactly_once_and_blocks_do_not_cross(self) -> None:
        scene_ids = [row.sample_id for row in self.contract.scene_assignments]
        radio_ids = [row.csv_row_number for row in self.contract.radio_assignments]
        self.assertEqual(len(scene_ids), len(set(scene_ids)))
        self.assertEqual(len(radio_ids), len(set(radio_ids)))

        scene_block_splits = {}
        for row in self.contract.scene_assignments:
            scene_block_splits.setdefault(row.temporal_block, set()).add(row.split)
        self.assertTrue(all(len(splits) == 1 for splits in scene_block_splits.values()))

        radio_block_splits = {}
        for row in self.contract.radio_assignments:
            key = (row.network_profile, row.trace_id, row.trace_step_block)
            radio_block_splits.setdefault(key, set()).add(row.split)
        self.assertTrue(all(len(splits) == 1 for splits in radio_block_splits.values()))

        self.assertEqual(
            sum(len(block.sample_ids) for block in self.contract.scene_blocks), 476
        )
        self.assertEqual(
            sum(len(block.csv_row_numbers) for block in self.contract.radio_blocks),
            399,
        )

    def test_exact_d1_source_bindings(self) -> None:
        bindings = self.contract.source_bindings
        self.assertEqual(bindings["action_catalog_sha256"], CATALOG_SHA256)
        self.assertEqual(
            bindings["modeled_smoke_support_sha256"],
            MODELED_SMOKE_SUPPORT_SHA256,
        )
        self.assertEqual(
            bindings["pilot_utility_spec_sha256"], PILOT_UTILITY_SPEC_SHA256
        )
        self.assertEqual(
            bindings["quality_selection_sha256"], SELECTION_FILE_SHA256
        )
        self.assertEqual(
            bindings["quality_database_sha256"], DATABASE_FILE_SHA256
        )
        self.assertEqual(
            bindings["quality_reward_definition_sha256"],
            REWARD_SPEC_FILE_SHA256,
        )
        self.assertEqual(
            bindings["radio_calibration_sha256"], CALIBRATION_SHA256
        )
        for key in (
            "corrected_p40_binding_sha256",
            "d1_eligible_fit_context_index_sha256",
            "d1_pilot_binding_sha256",
            "d1_quality_surface_binding_sha256",
            "network_surrogate_sha256",
        ):
            self.assertRegex(bindings[key], r"^[0-9a-f]{64}$")

    def test_canonical_bytes_and_registered_hash_are_deterministic(self) -> None:
        fresh = partition.load_registered_empirical_fit_partition()
        self.assertEqual(self.contract.canonical_bytes(), fresh.canonical_bytes())
        self.assertEqual(
            hashlib.sha256(self.contract.canonical_bytes()).hexdigest(),
            partition.REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        )
        self.assertEqual(
            self.contract.canonical_sha256(),
            partition.REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        )

    def test_profile_and_partition_identities_cannot_enter_policy_features(self) -> None:
        self.assertEqual(self.contract.policy_feature_order, POLICY_FEATURE_ORDER)
        joined = " ".join(self.contract.policy_feature_order).lower()
        for forbidden in (
            "network_profile",
            "profile_id",
            "trace_id",
            "csv_row",
            "sample_id",
            "temporal_block",
            "split",
        ):
            self.assertNotIn(forbidden, joined)
        self.assertFalse(hasattr(self.contract, "as_policy_observation"))

    def test_no_quality_held_api_and_normalization_disclosure_is_explicit(self) -> None:
        self.assertFalse(hasattr(self.contract, "evaluate_held"))
        self.assertEqual(
            {row.split for row in self.contract.scene_assignments},
            {"train", "fit_validation"},
        )
        disclosure = " ".join(self.contract.disclosures)
        self.assertIn("NOT_COVARIATE_NORMALIZATION_HELD", disclosure)
        self.assertIn("ALL_512_FIT_COVARIATES", disclosure)
        self.assertIn("399_JOINT_VALID_RADIO_ROWS", disclosure)
        self.assertIn("NO_REWARD_QUALITY_PAYLOAD_LATENCY_OR_DELIVERY", disclosure)

    def test_contract_is_immutable_and_tampering_fails_closed(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.contract.schema = "foreign"
        with self.assertRaises(TypeError):
            self.contract.source_bindings["new"] = "0" * 64
        first = self.contract.scene_assignments[0]
        with self.assertRaises(partition.FitPartitionError):
            replace(
                first,
                split=(
                    "fit_validation" if first.split == "train" else "train"
                ),
            )
        bad_bindings = dict(self.contract.source_bindings)
        bad_bindings["action_catalog_sha256"] = "bad"
        with self.assertRaises(partition.FitPartitionError):
            replace(self.contract, source_bindings=bad_bindings)

    def test_loading_does_not_advance_module_global_random(self) -> None:
        before = random.getstate()
        partition.load_registered_empirical_fit_partition()
        self.assertEqual(random.getstate(), before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

