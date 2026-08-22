from __future__ import annotations

import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path

import rl_agent.ue_n3_prepare_oai_ul_boundary_calibration as n3


class UEN3PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(n3.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory(prefix="ue-n3-plan-tests-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_config_is_offline_and_does_not_guess_commands(self) -> None:
        n3.validate_config(self.config)
        self.assertEqual(
            self.config["screening"]["desired_achieved_pusch_snr_db"],
            [6.0, 4.0, 3.0, 2.0],
        )
        self.assertTrue(
            all(
                value is None
                for value in self.config["screening"]["commanded_noise_power_db_by_target"].values()
            )
        )
        self.assertFalse(self.config["authority"]["oai_run_authorized"])
        self.assertFalse(self.config["authority"]["socket_execution_authorized"])
        self.assertEqual(
            self.config["traffic_probe"]["receiver_offline_test_status"],
            "PASSED",
        )
        self.assertTrue(n3.resolve(self.config["traffic_probe"]["receiver_path"]).is_file())

    def test_non_null_unreviewed_mapping_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["screening"]["commanded_noise_power_db_by_target"]["6.0"] = -3.0
        with self.assertRaisesRegex(n3.PlanError, "must not guess"):
            n3.validate_config(changed)

    def test_receiver_hash_drift_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["traffic_probe"]["receiver_sha256"] = "0" * 64
        with self.assertRaisesRegex(n3.PlanError, "receiver hash drift"):
            n3.validate_config(changed)

    def test_prepare_is_create_only_and_emits_blocked_matrix(self) -> None:
        output = self.root / "evidence"
        n3.prepare(n3.DEFAULT_CONFIG, output)
        terminal = json.loads(
            (output / self.config["outputs"]["terminal"]).read_text(encoding="utf-8")
        )
        self.assertEqual(terminal["status"], n3.SUCCESS_STATUS)
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertFalse(terminal["numeric_bound_promoted"])
        with (output / self.config["outputs"]["trial_matrix"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        screen = [row for row in rows if row["phase"] == "N3A_SUSTAIN_SCREEN"]
        self.assertEqual([float(row["desired_achieved_pusch_snr_db"]) for row in screen], [6, 4, 3, 2])
        self.assertTrue(all(not row["commanded_noise_power_db"] for row in screen))
        self.assertTrue(all(row["status"].startswith("BLOCKED_") for row in screen))
        with self.assertRaisesRegex(n3.PlanError, "create-only"):
            n3.prepare(n3.DEFAULT_CONFIG, output)

    def test_predecessor_hash_drift_is_rejected_before_output(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["predecessor"]["manifest_sha256"] = "0" * 64
        config_path = self.root / "changed.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        output = self.root / "not-created"
        with self.assertRaisesRegex(n3.PlanError, "hash drift"):
            n3.prepare(config_path, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
