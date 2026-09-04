"""Two synthetic CPU checks for the inert Phase-11D low-bit runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from .. import ae_phase11d_lowbit_validation as phase11d


class Phase11DLowBitCatalogTest(unittest.TestCase):
    def test_same_family_same_q_uint8_reference_resolution(self) -> None:
        cuda_before = torch.cuda.is_initialized()
        setting = next(
            row
            for row in phase11d.CATALOG
            if row.family.name == "AE64" and row.bit_width == 4 and row.q_e4 == 7000
        )
        spec = phase11d.UINT8_REFERENCES["AE64"]
        references = {
            "AE64": {
                7000: {
                    "family": "AE64",
                    "q": 0.70,
                    "q_e4": 7000,
                    "checkpoint_sha256": spec.checkpoint_sha256,
                    "source_sha256": spec.sha256,
                }
            }
        }
        resolved = phase11d.resolve_same_family_uint8_reference(setting, references)
        self.assertEqual(resolved["family"], "AE64")
        self.assertEqual(resolved["q_e4"], setting.q_e4)
        self.assertEqual(resolved["checkpoint_sha256"], spec.checkpoint_sha256)
        with self.assertRaises(guards.HybridQConfigError):
            phase11d.resolve_same_family_uint8_reference(
                setting, {"AE64": {7000: {**resolved, "family": "AE32"}}}
            )
        self.assertEqual(torch.cuda.is_initialized(), cuda_before)

    def test_durable_reuse_refusal_and_complete_inventory(self) -> None:
        cuda_before = torch.cuda.is_initialized()
        self.assertEqual(len(phase11d.CATALOG), 48)
        self.assertEqual(len({row.key for row in phase11d.CATALOG}), 48)
        self.assertEqual(
            {(row.family.name, row.bit_width, row.q_e4) for row in phase11d.CATALOG},
            {
                (family, bits, q_e4)
                for family in ("noAE", "AE128", "AE64", "AE32")
                for bits in (6, 4)
                for q_e4 in (0, 3000, 5000, 7000, 9000, 9800)
            },
        )
        setting = phase11d.CATALOG[0]
        identity = {"sha256": "phase11d-synthetic-identity"}
        with tempfile.TemporaryDirectory() as raw_output:
            output = Path(raw_output)
            record_path = phase11d.setting_path(output, setting)
            phase11d._atomic_json(
                record_path,
                phase11d._minimal_durable_record_for_test(setting, identity),
            )
            reused = phase11d.reuse_or_refuse(
                output=output, setting=setting, identity=identity, keep_segmentation=False
            )
            self.assertIsNotNone(reused)
            self.assertTrue(phase11d.cleanup_path(output, setting).is_file())
            damaged = json.loads(record_path.read_text(encoding="utf-8"))
            damaged["frames"] = contract.VALIDATION_FRAMES - 1
            phase11d._atomic_json(record_path, damaged)
            with self.assertRaises(guards.HybridQConfigError):
                phase11d.reuse_or_refuse(
                    output=output, setting=setting, identity=identity, keep_segmentation=False
                )
            self.assertEqual(damaged, json.loads(record_path.read_text(encoding="utf-8")))
        self.assertEqual(torch.cuda.is_initialized(), cuda_before)
