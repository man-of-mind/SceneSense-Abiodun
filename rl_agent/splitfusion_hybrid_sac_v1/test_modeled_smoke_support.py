"""CPU-only contract tests for the hash-bound modeled-smoke support set."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from . import empirical_quality_surface as quality_surface
from . import modeled_smoke_support as mss
from . import payload_network_surrogate as network_surrogate


class ModeledSmokeSupportContractTest(unittest.TestCase):
    def test_exact_inclusive_bounds_and_audit_counts(self) -> None:
        self.assertEqual(
            mss.MODELED_SMOKE_MODE_Q_E4_BOUNDS,
            (
                (8812, 9800),
                (8493, 9800),
                (7491, 9800),
                (8280, 9800),
                (7827, 9800),
                (5902, 9800),
                (6603, 9800),
                (5682, 9800),
                (1946, 9800),
                (3110, 9800),
                (1217, 9800),
                (0, 9791),
            ),
        )
        contract = mss.MODELED_SMOKE_SUPPORT
        self.assertTrue(contract.bounds_are_inclusive)
        self.assertEqual(contract.fit_frame_count, 512)
        self.assertEqual(contract.held_audit_frame_count, 256)
        self.assertEqual(contract.all_profile_payload_support, (6423, 427605))
        self.assertEqual(contract.payload_coordinate, "total_transmitted_bytes")
        self.assertEqual(contract.held_audit_supported_count, 13_373_151)
        self.assertEqual(contract.held_audit_total_count, 13_373_440)
        self.assertEqual(contract.held_audit_refused_count, 289)
        self.assertEqual(
            sum(upper - lower + 1 for lower, upper in contract.mode_q_e4_bounds)
            * contract.held_audit_frame_count,
            contract.held_audit_total_count,
        )
        self.assertEqual(
            contract.held_audit_supported_count
            + contract.held_audit_refused_count,
            contract.held_audit_total_count,
        )
        self.assertEqual(contract.evidence_class, "MODELED_SMOKE_SUPPORT")
        self.assertEqual(contract.use_scope, "CONTEXTUAL_SMOKE_CURRICULUM_ONLY")
        self.assertEqual(
            contract.deployment_action_contract_status,
            "NEVER_A_DEPLOYMENT_ACTION_CONTRACT",
        )
        self.assertIn("FIT_ONLY_512_FRAME", contract.derivation_method)

    def test_canonical_hash_is_independently_reproducible_and_static(self) -> None:
        document = mss.MODELED_SMOKE_SUPPORT.to_canonical_dict()
        encoded = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        independent = hashlib.sha256(encoded).hexdigest()
        self.assertEqual(independent, mss.MODELED_SMOKE_SUPPORT_SHA256)
        self.assertEqual(
            mss.MODELED_SMOKE_SUPPORT.canonical_sha256(), independent
        )
        self.assertIs(
            mss.require_registered_modeled_smoke_support(
                mss.MODELED_SMOKE_SUPPORT
            ),
            mss.MODELED_SMOKE_SUPPORT,
        )

    def test_source_pins_are_the_registered_model_source_pins(self) -> None:
        pins = {
            pin.component: pin.sha256
            for pin in mss.MODELED_SMOKE_SUPPORT.source_sha_pins
        }
        self.assertEqual(
            pins["quality_surface_complete"],
            quality_surface.COMPLETE_FILE_SHA256,
        )
        self.assertEqual(
            pins["quality_surface_run_manifest_file"],
            quality_surface.RUN_MANIFEST_FILE_SHA256,
        )
        self.assertEqual(
            pins["quality_surface_run_manifest_identity"],
            quality_surface.RUN_MANIFEST_SHA256,
        )
        self.assertEqual(
            pins["quality_surface_run_binding"],
            quality_surface.RUN_BINDING_SHA256,
        )
        self.assertEqual(
            pins["quality_surface_selection_file"],
            quality_surface.SELECTION_FILE_SHA256,
        )
        self.assertEqual(
            pins["quality_surface_selection_identity"],
            quality_surface.SELECTION_MANIFEST_SHA256,
        )
        self.assertEqual(
            pins["quality_surface_reward_spec"],
            quality_surface.REWARD_SPEC_FILE_SHA256,
        )
        self.assertEqual(
            pins["quality_surface_database"],
            quality_surface.DATABASE_FILE_SHA256,
        )
        self.assertEqual(
            pins["network_anchor_action_summary"],
            network_surrogate.ACTION_SUMMARY_SHA256,
        )
        self.assertEqual(
            pins["network_anchor_profile_latency"],
            network_surrogate.PROFILE_LATENCY_SHA256,
        )
        self.assertEqual(
            pins["action_catalog"], network_surrogate.CATALOG_SHA256
        )
        self.assertEqual(
            pins["network_analysis_builder"],
            network_surrogate.SOURCE_ANALYSIS_BUILDER_SHA256,
        )
        self.assertEqual(
            pins["network_analysis_summary"],
            network_surrogate.SOURCE_ANALYSIS_SUMMARY_SHA256,
        )
        self.assertEqual(
            pins["network_analysis_manifest"],
            network_surrogate.SOURCE_ANALYSIS_MANIFEST_SHA256,
        )
        self.assertEqual(
            pins["production_transport"],
            network_surrogate.PRODUCTION_TRANSPORT_SHA256,
        )
        self.assertEqual(
            pins["production_runtime_contract"],
            network_surrogate.PRODUCTION_RUNTIME_CONTRACT_SHA256,
        )

    def test_contract_is_immutable_and_foreign_or_malformed_values_are_refused(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            mss.MODELED_SMOKE_SUPPORT.fit_frame_count = 1
        for foreign in (
            object(),
            {"schema": mss.MODELED_SMOKE_SUPPORT_SCHEMA},
            replace(mss.MODELED_SMOKE_SUPPORT, schema="foreign.v1"),
            replace(
                mss.MODELED_SMOKE_SUPPORT,
                mode_q_e4_bounds=((0, 9800),) * 12,
            ),
            replace(
                mss.MODELED_SMOKE_SUPPORT,
                held_audit_refused_count=288,
            ),
            replace(
                mss.MODELED_SMOKE_SUPPORT,
                fit_frame_count=512.0,
            ),
            replace(
                mss.MODELED_SMOKE_SUPPORT,
                mode_q_e4_bounds=list(
                    mss.MODELED_SMOKE_SUPPORT.mode_q_e4_bounds
                ),
            ),
            replace(
                mss.MODELED_SMOKE_SUPPORT,
                bounds_are_inclusive=1,
            ),
        ):
            with self.assertRaises(mss.ModeledSmokeSupportError):
                mss.require_registered_modeled_smoke_support(foreign)


class ModeledSmokeSupportImportPurityTest(unittest.TestCase):
    def test_import_has_no_evidence_io_global_rng_or_cuda_query(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        probe = r'''
import json, sys, torch
violations = []

def hook(event, args):
    try:
        if event == "open":
            path = str(args[0])
            low = path.lower()
            if "site-packages" in low or "dist-packages" in low:
                return
            if low.endswith((".csv", ".json", ".sqlite", ".sqlite3")) or "/experiments/" in low:
                violations.append([event, path])
        elif event in ("subprocess.Popen", "os.system", "socket.socket", "socket.connect"):
            violations.append([event, str(args)[:120]])
    except Exception:
        pass

def forbidden_cuda(*args, **kwargs):
    raise AssertionError("CUDA queried during import")

torch.cuda.is_available = forbidden_cuda
torch.cuda.device_count = forbidden_cuda
torch.manual_seed(90210)
before = torch.get_rng_state().clone()
expected = torch.rand(8)
torch.set_rng_state(before)
sys.addaudithook(hook)
import rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support as support
observed = torch.rand(8)
assert torch.equal(observed, expected)
assert support.MODELED_SMOKE_SUPPORT.fit_frame_count == 512
print("VIOLATIONS:" + json.dumps(violations))
'''
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(
            completed.returncode, 0, f"probe failed: {completed.stderr[-2000:]}"
        )
        marker = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("VIOLATIONS:")
        ]
        self.assertEqual(len(marker), 1, completed.stdout[-2000:])
        self.assertEqual(json.loads(marker[0][len("VIOLATIONS:") :]), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
