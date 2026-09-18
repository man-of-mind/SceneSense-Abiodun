"""Focused tests for the design-only anchor reward sensitivity audit."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_v1.anchor_reward_sensitivity import (
    BUDGET_MS,
    DEFAULT_ACTION_PROFILE,
    DEFAULT_ACTION_SUMMARY,
    QualitySpec,
    SensitivityAuditError,
    average_ranks,
    enumerate_quality_specs,
    load_sources,
    reward_proxy,
    run_audit,
    score_quality,
    spearman,
)
from rl_agent.splitfusion_hybrid_sac_v1.anchor_store import (
    ACTION_SUMMARY_SHA256,
    PROFILE_LATENCY_SHA256,
)


class AnchorRewardSensitivityTest(unittest.TestCase):
    @staticmethod
    def _valid_spec(**overrides: object) -> QualitySpec:
        values: dict[str, object] = {
            "spec_id": "fixture",
            "localization_combiner": "geometric",
            "person_localization_share": 0.6,
            "person_segmentation_share": 0.6,
            "tau_person_m": 1.2,
            "tau_vehicle_m": 1.0,
            "tau_label": "fixture",
            "segmentation_modulation_beta": 0.3,
            "is_initial_hypothesis": False,
        }
        values.update(overrides)
        return QualitySpec(**values)  # type: ignore[arg-type]

    def test_grid_is_complete_and_has_one_initial_hypothesis(self) -> None:
        specs = enumerate_quality_specs()
        self.assertEqual(len(specs), 162)
        self.assertEqual(len({spec.spec_id for spec in specs}), 162)
        initial = [spec for spec in specs if spec.is_initial_hypothesis]
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0].localization_combiner, "geometric")
        self.assertEqual(initial[0].person_localization_share, 0.6)
        self.assertEqual(initial[0].person_segmentation_share, 0.6)
        self.assertEqual(initial[0].tau_person_m, 1.2)
        self.assertEqual(initial[0].tau_vehicle_m, 1.0)
        self.assertEqual(initial[0].segmentation_modulation_beta, 0.30)

    def test_quality_formula_matches_manual_calculation(self) -> None:
        spec = QualitySpec(
            spec_id="fixture",
            localization_combiner="geometric",
            person_localization_share=0.6,
            person_segmentation_share=0.7,
            tau_person_m=1.2,
            tau_vehicle_m=1.0,
            tau_label="fixture",
            segmentation_modulation_beta=0.3,
            is_initial_hypothesis=False,
        )
        action = {
            "val_person_avo_recall": "0.64",
            "val_person_avo_xy_mae_m": "0.6",
            "val_vehicle_recall": "0.81",
            "val_vehicle_xy_mae_m": "0.4",
            "val_person_box_mask_iou": "0.4",
            "val_vehicle_iou": "0.8",
        }
        result = score_quality(action, spec)
        u_person = math.sqrt(0.64 * math.exp(-0.6 / 1.2))
        u_vehicle = math.sqrt(0.81 * math.exp(-0.4 / 1.0))
        expected_loc = u_person**0.6 * u_vehicle**0.4
        s_person = 0.4 / 0.527894080
        s_vehicle = 0.8 / 0.899012847
        expected_seg = s_person**0.7 * s_vehicle**0.3
        expected = expected_loc * (0.7 + 0.3 * expected_seg)
        self.assertAlmostEqual(result.u_loc_person, u_person)
        self.assertAlmostEqual(result.u_loc_vehicle, u_vehicle)
        self.assertAlmostEqual(result.q_loc, expected_loc)
        self.assertAlmostEqual(result.q_seg, expected_seg)
        self.assertAlmostEqual(result.q_perc, expected)

    def test_arithmetic_localization_is_explicitly_different(self) -> None:
        base = dict(
            spec_id="fixture",
            person_localization_share=0.6,
            person_segmentation_share=0.6,
            tau_person_m=1.2,
            tau_vehicle_m=1.0,
            tau_label="fixture",
            segmentation_modulation_beta=0.3,
            is_initial_hypothesis=False,
        )
        action = {
            "val_person_avo_recall": 0.25,
            "val_person_avo_xy_mae_m": 1.0,
            "val_vehicle_recall": 0.95,
            "val_vehicle_xy_mae_m": 0.1,
            "val_person_box_mask_iou": 0.3,
            "val_vehicle_iou": 0.85,
        }
        arithmetic = score_quality(
            action, QualitySpec(localization_combiner="arithmetic", **base)
        )
        geometric = score_quality(
            action, QualitySpec(localization_combiner="geometric", **base)
        )
        self.assertNotAlmostEqual(arithmetic.q_loc, geometric.q_loc)

    def test_missing_latency_is_not_scored_as_zero(self) -> None:
        self.assertIsNone(reward_proxy(0.8, None, 0.25))
        self.assertAlmostEqual(
            reward_proxy(0.8, 100.0, 0.25),
            0.8 - 0.25 * 100.0 / BUDGET_MS,
        )

    def test_repository_source_tables_reconcile_as_72_by_4(self) -> None:
        actions, profiles, binding = load_sources(
            DEFAULT_ACTION_SUMMARY, DEFAULT_ACTION_PROFILE
        )
        self.assertEqual(len(actions), 72)
        self.assertEqual(len(profiles), 288)
        self.assertEqual(binding["action_72_summary"]["rows"], 72)
        self.assertEqual(binding["action_profile_quality_latency_v3"]["rows"], 288)
        self.assertEqual(len(binding["action_72_summary"]["sha256"]), 64)
        self.assertEqual(
            len(binding["action_profile_quality_latency_v3"]["sha256"]), 64
        )
        self.assertEqual(
            binding["action_72_summary"]["sha256"], ACTION_SUMMARY_SHA256
        )
        self.assertEqual(
            binding["action_profile_quality_latency_v3"]["sha256"],
            PROFILE_LATENCY_SHA256,
        )
        self.assertEqual(binding["binding_implementation"], "AnchorEvidenceStore")
        self.assertEqual(binding["frozen_action_catalog"]["reconciled_action_count"], 72)
        self.assertEqual(len(binding["frozen_action_catalog"]["sha256"]), 64)

    def test_source_hash_drift_fails_before_sensitivity_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            drifted = Path(directory) / "action_72_summary.csv"
            drifted.write_bytes(DEFAULT_ACTION_SUMMARY.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                SensitivityAuditError, "anchor-store evidence binding failed.*SHA-256"
            ):
                load_sources(drifted, DEFAULT_ACTION_PROFILE)

    def test_quality_spec_rejects_invalid_scientific_domains(self) -> None:
        invalid = (
            {"localization_combiner": "median"},
            {"person_localization_share": -0.01},
            {"person_localization_share": 1.01},
            {"person_segmentation_share": float("nan")},
            {"segmentation_modulation_beta": -0.01},
            {"segmentation_modulation_beta": 1.01},
            {"tau_person_m": 0.0},
            {"tau_person_m": -1.0},
            {"tau_vehicle_m": float("inf")},
        )
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(SensitivityAuditError):
                    self._valid_spec(**override)

    def test_end_to_end_artifacts_and_manifest_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = run_audit(output_dir=Path(directory) / "audit")
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, metadata in manifest["artifacts"].items():
                artifact = output / name
                self.assertTrue(artifact.is_file(), name)
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    metadata["sha256"],
                )
                self.assertEqual(artifact.stat().st_size, metadata["bytes"])

            declared_content_hash = manifest.pop("content_sha256")
            canonical = json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), declared_content_hash)

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            joined_limitations = " ".join(summary["limitations"])
            self.assertIn("not the exact reward latency", joined_limitations)
            self.assertIn("survivor bias", joined_limitations)
            self.assertFalse(summary["latency_weight_frozen"])

    def test_average_ranks_and_spearman_handle_ties(self) -> None:
        self.assertEqual(average_ranks([3.0, 2.0, 2.0, 1.0]), [1.0, 2.5, 2.5, 4.0])
        self.assertAlmostEqual(spearman([3.0, 2.0, 1.0], [30.0, 20.0, 10.0]), 1.0)
        self.assertAlmostEqual(spearman([3.0, 2.0, 1.0], [10.0, 20.0, 30.0]), -1.0)


if __name__ == "__main__":
    unittest.main()
