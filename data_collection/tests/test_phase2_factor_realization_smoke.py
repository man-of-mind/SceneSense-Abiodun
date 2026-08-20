from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout

from data_collection.validate_phase2_factor_realization_smoke import (
    ContractError,
    RESULT_SCHEMA,
    build_plan,
    load_config,
    main,
    validate_results,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FactorRealizationSmokePlanTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        # The shared tree is repinned only after all collaborating agents have
        # completed edits. The launcher separately enforces the persisted hash.
        design = Path(self.config["source_design"]["design_config"])
        self.config["source_design"]["design_config_sha256"] = _sha(design)
        self.plan = build_plan(self.config)

    def test_plan_is_exact_bounded_calibration_tranche(self):
        self.assertEqual(self.plan["trajectory_count"], 16)
        self.assertEqual(self.plan["group_count"], 8)
        self.assertEqual(self.plan["positive_trajectory_count"], 8)
        self.assertEqual(self.plan["benign_trajectory_count"], 8)
        self.assertEqual(
            self.plan["reuse_if_atomic_pass"], "replicate_0_calibration_tranche"
        )
        self.assertFalse(self.plan["collection_authorized_by_this_tool"])
        cells = {
            (
                row["geometry_or_route_id"],
                row["closing_speed_band"],
                row["time_to_hazard_band"],
            )
            for row in self.plan["rows"]
        }
        self.assertEqual(len(cells), 8)

    def test_policy_feature_contract_rejects_scenario_clock(self):
        drift = copy.deepcopy(self.config)
        drift["policy_feature_contract"]["placement_features"].append("elapsed_s")
        with self.assertRaisesRegex(ContractError, "forbidden policy feature"):
            build_plan(drift)

    def test_policy_feature_contract_rejects_authored_onset_control(self):
        drift = copy.deepcopy(self.config)
        drift["policy_feature_contract"]["placement_features"].append(
            "requested_hazard_onset_s"
        )
        with self.assertRaisesRegex(ContractError, "forbidden policy feature"):
            build_plan(drift)

    def test_capture_only_or_legacy_result_cannot_pass(self):
        legacy = {
            "schema": RESULT_SCHEMA,
            "stage_id": self.config["stage_id"],
            "source_manifest_sha256": self.plan["source_manifest_sha256"],
            "plan_sha256": self.plan["plan_sha256"],
            "trajectories": [],
        }
        with self.assertRaisesRegex(ContractError, "self hash"):
            validate_results(legacy, self.config, self.plan)

    def test_runtime_ready_cli_hands_off_only_the_manual_corner_review(self):
        output = io.StringIO()
        with redirect_stdout(output):
            returncode = main(["--require-runtime-ready"])
        summary = json.loads(output.getvalue())
        self.assertEqual(returncode, 0)
        self.assertTrue(summary["runtime_ready"])
        self.assertFalse(summary["collection_authorized_by_this_tool"])
        self.assertEqual(
            summary["next_action"],
            "run_hash_bound_manual_eight_corner_review_no_exact16_launch_yet",
        )


if __name__ == "__main__":
    unittest.main()
