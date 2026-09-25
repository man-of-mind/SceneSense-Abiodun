from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from . import run4_contract as contract
from . import scientific_basis as src


ROOT = Path(__file__).resolve().parents[2]


class ScientificBasisTests(unittest.TestCase):
    def test_real_quality_source_is_exactly_verified(self) -> None:
        self.assertEqual(
            src.verify_quality_source(ROOT), src.QUALITY_SOURCE_FILE_SHA256
        )

    def test_quality_and_scalar_reward_are_explicitly_separate(self) -> None:
        descriptor = src.SCIENTIFIC_BASIS_DESCRIPTOR
        self.assertEqual(
            descriptor["scalar_reward"]["schema_sha256"],
            contract.REWARD_SCHEMA_SHA256,
        )
        self.assertEqual(descriptor["scalar_reward"]["deadline_ms"], 170.0)
        self.assertIn("quality_target_only", src.QUALITY_APPROVAL_STATUS.lower())
        self.assertIn("not adopted", src.QUALITY_APPROVAL_SCOPE)

    def test_localization_is_primary_and_segmentation_is_bounded_modulation(self) -> None:
        definition = src.QUALITY_DEFINITION
        self.assertEqual(definition.localization_person_weight, 0.6)
        self.assertEqual(definition.localization_vehicle_weight, 0.4)
        self.assertEqual(definition.segmentation_modulation_beta, 0.3)
        # Qperc = Qloc * (0.7 + 0.3 Qseg), so segmentation cannot erase
        # localization or dominate the score.
        self.assertEqual(1.0 - definition.segmentation_modulation_beta, 0.7)

    def test_source_byte_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / src.QUALITY_SOURCE_RELATIVE_PATH
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"localization_combiner": "changed"}))
            with self.assertRaisesRegex(src.ScientificBasisError, "SHA-256 drift"):
                src.verify_quality_source(root)

    def test_import_is_io_free(self) -> None:
        code = r'''
import builtins
import importlib
import pathlib
import socket
import subprocess

def forbidden(*args, **kwargs):
    raise RuntimeError("forbidden import side effect")

builtins.open = forbidden
pathlib.Path.open = forbidden
pathlib.Path.read_bytes = forbidden
pathlib.Path.read_text = forbidden
subprocess.Popen = forbidden
subprocess.run = forbidden
socket.socket = forbidden
importlib.import_module("rl_agent.splitfusion_hybrid_sac_run4_v1.scientific_basis")
print("ok")
'''
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
