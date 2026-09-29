"""Phase-5 tests: the sealed live binding manifest reproduces and detects drift.

File hashing only; no CARLA, OAI, CUDA, model or network process.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from . import readiness_v2 as RD


class ReadinessManifestTest(unittest.TestCase):
    def test_sealed_manifest_reproduces_from_working_tree(self) -> None:
        self.assertTrue(RD.verify_manifest()["verified"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_manifest_binds_the_registered_identities(self) -> None:
        manifest = json.loads(RD.MANIFEST_PATH.read_text())
        self.assertEqual((manifest["actor"]["seed"], manifest["actor"]["update"]),
                         (43, 10000))
        self.assertEqual(manifest["actor"]["boundary_sha256"],
                         "b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3")
        self.assertEqual(manifest["contracts"]["reward_deadline_ns"], 170_000_000)
        self.assertEqual(manifest["contracts"]["k_min"], 2)
        self.assertEqual(len(manifest["contracts"]["policy_feature_order"]), 21)
        self.assertEqual(manifest["telemetry"]["status"], "PASSED")
        self.assertEqual(manifest["radio"]["radio_binding_problems"], [])
        self.assertEqual(set(manifest["execution"]["selected_checkpoints"]),
                         {"perception", "ranker", "AE128", "AE64", "AE32"})
        plan = json.loads(RD.CONFIG_PATH.read_text())
        self.assertEqual(plan["status"], "PLAN_ONLY__NOT_AUTHORIZED__NOT_EXECUTED")
        self.assertEqual(plan["run"]["frames"], 300)

    def test_tampered_manifest_is_rejected(self) -> None:
        manifest = json.loads(RD.MANIFEST_PATH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            manifest["contracts"]["k_min"] = 1
            path.write_text(json.dumps(manifest))
            with self.assertRaises(RD.ReadinessError):
                RD.verify_manifest(path=path)
            body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
            manifest["manifest_sha256"] = RD.canonical_sha256(body)   # resealed
            path.write_text(json.dumps(manifest))
            with self.assertRaises(RD.ReadinessError):
                RD.verify_manifest(path=path)


if __name__ == "__main__":
    unittest.main()
