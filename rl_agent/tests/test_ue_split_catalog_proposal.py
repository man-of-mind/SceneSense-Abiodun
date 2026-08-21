from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rl_agent.ue_split_catalog_proposal import (
    CatalogProposalError,
    apply_quality_gates,
    assemble_proposal,
    load_decision,
)


ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "rl_agent/decisions/ue_split_object_map_v1_floor_v1.yaml"
PARENT = ROOT / "rl_agent/experiments/ue_split_stage_a_v1/20260820_024055_review"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CatalogProposalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = Path(tempfile.mkdtemp(prefix="ue-split-catalog-proposal-tests-"))
        cls.output = cls.temp / "candidate_a"
        cls.result = assemble_proposal(
            DECISION,
            cls.output,
            now=datetime(2026, 8, 20, 4, 30, tzinfo=timezone.utc),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp, ignore_errors=True)

    def test_exact_candidate_shapes_and_tiers(self) -> None:
        gates = pd.read_csv(self.output / "ue_split_absolute_quality_gate.csv")
        candidates = pd.read_csv(self.output / "ue_split_candidate_catalog.csv")
        shortlist = pd.read_csv(self.output / "ue_split_audit_priority_shortlist.csv")
        strata = pd.read_csv(self.output / "ue_split_candidate_quality_strata.csv")
        screen = pd.read_csv(
            self.output / "ue_split_candidate_profile_regime_screen.csv"
        )
        self.assertEqual(len(gates), 72)
        self.assertEqual(int(gates["absolute_quality_pass"].sum()), 28)
        self.assertEqual(int(gates["normal_candidate"].sum()), 26)
        self.assertEqual(len(candidates), 27)
        self.assertEqual(
            candidates["candidate_tier"].value_counts().to_dict(),
            {"NORMAL": 26, "DEGRADED_RESCUE": 1},
        )
        self.assertEqual(len(shortlist), 7)
        self.assertEqual(len(strata), 27 * 9)
        self.assertEqual(len(screen), 27 * 4)

    def test_rescue_is_separate_and_fails_only_approved_normal_gate(self) -> None:
        candidates = pd.read_csv(self.output / "ue_split_candidate_catalog.csv")
        rescue = candidates.loc[candidates["candidate_tier"] == "DEGRADED_RESCUE"].iloc[0]
        self.assertEqual(rescue["display_profile_id"], "ae32/u4/q0.9")
        self.assertEqual(rescue["normal_failure_reasons"], "recall_pedestrian")
        self.assertGreaterEqual(float(rescue["recall_pedestrian"]), 0.84)
        self.assertFalse(bool(rescue["normal_candidate"]))
        self.assertTrue(bool(rescue["service_debt_on_use"]))
        self.assertTrue(bool(rescue["activation_only_when_no_normal_feasible"]))
        self.assertTrue(bool(rescue["requires_network_feasibility"]))
        self.assertFalse(bool(rescue["counts_as_normal_service_success"]))
        self.assertTrue(bool(rescue["requires_final_evidence_review"]))

    def test_candidate_fields_replace_stale_stage_a_semantics(self) -> None:
        candidates = pd.read_csv(self.output / "ue_split_candidate_catalog.csv")
        for column in (
            "stage_a_eligibility_status",
            "stage_a_exclusion_reason",
            "stage_a_object_detail_evidence",
        ):
            self.assertIn(column, candidates.columns)
        self.assertFalse(
            candidates["eligibility_status"].str.contains(
                "NO_SELECTED_QUALITY_FLOOR", regex=False
            ).any()
        )
        self.assertFalse(
            candidates["exclusion_reason"].str.contains(
                "PENDING_ABSOLUTE_FLOOR", regex=False
            ).any()
        )
        pinned = candidates.loc[candidates["display_profile_id"] == "ae32/u4/q0.5"].iloc[0]
        self.assertEqual(
            pinned["object_detail_evidence"],
            "PINNED_REPRODUCED_HORIZONTAL_RANGE_SMALL_UNRESOLVED",
        )

    def test_no_final_catalog_or_run_authority_is_emitted(self) -> None:
        marker = json.loads((self.output / "CANDIDATE_REVIEW_REQUIRED.json").read_text())
        candidates = pd.read_csv(self.output / "ue_split_candidate_catalog.csv")
        screen = pd.read_csv(
            self.output / "ue_split_candidate_profile_regime_screen.csv"
        )
        self.assertEqual(marker["decision_state"], "CANDIDATE_REVIEW_REQUIRED")
        self.assertIsNone(marker["eligible_action_count"])
        self.assertFalse(marker["measurement_authorized"])
        self.assertTrue(marker["no_run_authority"])
        self.assertFalse(candidates["final_eligible"].any())
        self.assertFalse(candidates["measurement_authorized"].any())
        self.assertFalse(screen["measurement_authorized"].any())
        for name in (
            "ue_split_action_catalog.csv",
            "ue_split_profile_network_surface.csv",
            "FROZEN.json",
            "COMPLETED.json",
        ):
            self.assertFalse((self.output / name).exists(), name)

    def test_range_evidence_is_reproduced_but_small_object_gate_stays_open(self) -> None:
        ranges = pd.read_csv(self.output / "ue_split_range_audit.csv")
        comparisons = pd.read_csv(self.output / "ue_split_range_comparison.csv")
        candidates = pd.read_csv(self.output / "ue_split_candidate_catalog.csv")
        self.assertEqual(len(ranges), 7 * 3 * 2)
        self.assertEqual(len(comparisons), 4 * 3 * 2)
        self.assertTrue((ranges["gt_rows_missing_bbox"] > 0).any())
        self.assertEqual(
            set(ranges["small_object_status"]), {"UNRESOLVED_FN_BOXES_MISSING"}
        )
        self.assertEqual(
            set(candidates["small_object_status"]),
            {"UNRESOLVED_SOURCE_GT_ABSENT_FN_BOXES_MISSING"},
        )
        totals = ranges.groupby(["display_profile_id", "class_name"])["n_gt"].sum()
        self.assertEqual(set(totals.xs("vehicle", level="class_name")), {2468})
        self.assertEqual(set(totals.xs("person", level="class_name")), {1431})

    def test_unresolved_work_is_truthful_but_never_authorized(self) -> None:
        unresolved = pd.read_csv(self.output / "ue_split_unresolved_evidence.csv")
        required = unresolved.set_index("item_id")["new_evidence_generation_required"]
        self.assertTrue(bool(required["normal_high_roi_object_detail"]))
        self.assertTrue(bool(required["degraded_rescue_object_detail"]))
        self.assertTrue(bool(required["fixed_10hz_boundaries"]))
        self.assertFalse(unresolved["new_run_authorized"].any())
        self.assertFalse(unresolved["measurement_authorized"].any())

    def test_manifest_output_hash_chain_and_marker_validate(self) -> None:
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["outputs"]:
            path = self.output / item["path"]
            self.assertEqual(sha256(path), item["sha256"], item["path"])
            if path.suffix == ".csv":
                self.assertEqual(len(pd.read_csv(path)), item["rows"], item["path"])
        marker = json.loads((self.output / "CANDIDATE_REVIEW_REQUIRED.json").read_text())
        self.assertEqual(marker["manifest_sha256"], sha256(manifest_path))
        self.assertEqual(
            manifest["repository"]["assembler_sha256"],
            sha256(ROOT / manifest["repository"]["assembler_path"]),
        )
        snapshot = self.output / manifest["repository"]["assembler_snapshot_path"]
        self.assertEqual(
            manifest["repository"]["assembler_snapshot_sha256"], sha256(snapshot)
        )
        self.assertEqual(sha256(snapshot), sha256(ROOT / manifest["repository"]["assembler_path"]))
        self.assertFalse(
            manifest["degraded_rescue"]["included_in_normal_candidate_count"]
        )

    def test_output_is_create_only(self) -> None:
        with self.assertRaisesRegex(CatalogProposalError, "refusing to overwrite"):
            assemble_proposal(DECISION, self.output)

    def test_outputs_are_deterministic_for_same_inputs(self) -> None:
        other = self.temp / "candidate_b"
        assemble_proposal(
            DECISION,
            other,
            now=datetime(2026, 8, 20, 4, 30, tzinfo=timezone.utc),
        )
        names = (
            "resolved_decision.yaml",
            "ue_split_absolute_quality_gate.csv",
            "ue_split_candidate_catalog.csv",
            "ue_split_audit_priority_shortlist.csv",
            "ue_split_candidate_quality_strata.csv",
            "ue_split_candidate_profile_regime_screen.csv",
            "ue_split_range_audit.csv",
            "ue_split_range_comparison.csv",
            "ue_split_unresolved_evidence.csv",
            "REPORT.md",
            "ASSEMBLER_SNAPSHOT.py",
        )
        for name in names:
            self.assertEqual(sha256(self.output / name), sha256(other / name), name)


class CatalogProposalContractTests(unittest.TestCase):
    def _decision_copy(self, directory: Path) -> Path:
        value = yaml.safe_load(DECISION.read_text())
        value["repository_root"] = str(ROOT)
        path = directory / "decision.yaml"
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        return path

    def test_authority_expansion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-catalog-decision-") as directory:
            path = self._decision_copy(Path(directory))
            value = yaml.safe_load(path.read_text())
            value["authority"]["new_oai_run"] = True
            path.write_text(yaml.safe_dump(value, sort_keys=False))
            with self.assertRaisesRegex(CatalogProposalError, "prohibits authority"):
                load_decision(path)

    def test_rescue_activation_contract_tamper_is_rejected(self) -> None:
        mutations = (
            ("enabled", False),
            ("only_when_no_normal_action_physically_feasible", False),
            ("counts_as_normal_service_success", True),
        )
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                prefix="ue-catalog-decision-"
            ) as directory:
                path = self._decision_copy(Path(directory))
                decision = yaml.safe_load(path.read_text())
                if field == "enabled":
                    decision["degraded_rescue"][field] = value
                else:
                    decision["degraded_rescue"]["activation_contract"][field] = value
                path.write_text(yaml.safe_dump(decision, sort_keys=False))
                with self.assertRaises(CatalogProposalError):
                    load_decision(path)

    def test_service_and_numeric_floor_tamper_is_rejected(self) -> None:
        mutations = ("required_output", "segmentation_role", "nan_floor")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="ue-catalog-decision-"
            ) as directory:
                path = self._decision_copy(Path(directory))
                decision = yaml.safe_load(path.read_text())
                if mutation == "required_output":
                    decision["service_contract"]["required_outputs"].pop()
                elif mutation == "segmentation_role":
                    decision["service_contract"]["segmentation_output_role"] = "primary"
                else:
                    decision["quality_floor"]["normal"]["recall_vehicle_min"] = float("nan")
                path.write_text(yaml.safe_dump(decision, sort_keys=False))
                with self.assertRaises(CatalogProposalError):
                    load_decision(path)

    def test_parent_hash_drift_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-catalog-decision-") as directory:
            directory = Path(directory)
            path = self._decision_copy(directory)
            value = yaml.safe_load(path.read_text())
            value["parent_review"]["manifest_sha256"] = "0" * 64
            path.write_text(yaml.safe_dump(value, sort_keys=False))
            output = directory / "output"
            with self.assertRaisesRegex(CatalogProposalError, "manifest hash drift"):
                assemble_proposal(path, output)
            self.assertFalse(output.exists())

    def test_segmentation_never_changes_object_quality_eligibility(self) -> None:
        decision = load_decision(DECISION)
        catalog = pd.read_csv(PARENT / "ue_split_evidence_pool.csv")
        sensitivity = pd.read_csv(PARENT / "ue_split_quality_floor_sensitivity.csv")
        original = apply_quality_gates(catalog, sensitivity, decision)
        changed = catalog.copy()
        changed[["miou", "iou_background", "iou_vehicle", "iou_person"]] = -999.0
        altered = apply_quality_gates(changed, sensitivity, decision)
        self.assertTrue(
            original["normal_candidate"].equals(altered["normal_candidate"])
        )
        self.assertFalse(altered["segmentation_used_as_veto"].any())

    def test_thresholds_are_inclusive_and_nonfinite_metrics_fail_closed(self) -> None:
        decision = load_decision(DECISION)
        catalog = pd.read_csv(PARENT / "ue_split_evidence_pool.csv")
        sensitivity = pd.read_csv(PARENT / "ue_split_quality_floor_sensitivity.csv")
        floor = decision["quality_floor"]["normal"]
        index = 0
        catalog.loc[index, "recall_vehicle"] = floor["recall_vehicle_min"]
        catalog.loc[index, "recall_pedestrian"] = floor["recall_pedestrian_min"]
        catalog.loc[index, "precision_vehicle"] = floor["precision_vehicle_min"]
        catalog.loc[index, "precision_pedestrian"] = floor["precision_pedestrian_min"]
        catalog.loc[index, "xy_mae_vehicle_m"] = floor["xy_mae_vehicle_max_m"]
        catalog.loc[index, "xy_mae_pedestrian_m"] = floor["xy_mae_pedestrian_max_m"]
        catalog.loc[index, "fp_per_frame"] = floor["fp_per_frame_max"]
        gates = apply_quality_gates(catalog, sensitivity, decision)
        gate_columns = [column for column in gates if column.startswith("gate_")]
        absolute_columns = [column for column in gate_columns if column != "gate_roi_incremental"]
        self.assertTrue(gates.loc[index, absolute_columns].all())
        catalog.loc[index, "recall_vehicle"] = np.nan
        with self.assertRaisesRegex(CatalogProposalError, "nonfinite"):
            apply_quality_gates(catalog, sensitivity, decision)

    def test_roi_incremental_screen_requires_strict_boolean_values(self) -> None:
        decision = load_decision(DECISION)
        catalog = pd.read_csv(PARENT / "ue_split_evidence_pool.csv")
        sensitivity = pd.read_csv(PARENT / "ue_split_quality_floor_sensitivity.csv")
        sensitivity["roi_incremental_screen_pass"] = sensitivity[
            "roi_incremental_screen_pass"
        ].astype(str)
        sensitivity.loc[0, "roi_incremental_screen_pass"] = "not-a-boolean"
        with self.assertRaisesRegex(CatalogProposalError, "strict booleans"):
            apply_quality_gates(catalog, sensitivity, decision)


if __name__ == "__main__":
    unittest.main()
