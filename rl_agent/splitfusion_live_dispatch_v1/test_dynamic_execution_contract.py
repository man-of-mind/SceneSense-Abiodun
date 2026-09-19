"""Offline tests for the continuous-q live-dispatch Phase-1 contract.

A normal dotted import executes the package's eager ``__init__`` and may
therefore import torch.  No test loads a checkpoint, initializes CUDA, runs
inference, launches a service, or performs runtime/evidence I/O.  Contract
loading hashes the frozen files named by the reviewed runtime binding;
checkpoint bytes are never deserialized.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as ac

from . import dynamic_execution_contract as dec


class DynamicExecutionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = dec.load_dynamic_execution_contract()
        cls.catalog_document = json.loads(
            cls.contract.action_contract.catalog_path.read_text(encoding="utf-8")
        )

    def test_exact_frozen_bindings_and_twelve_modes(self) -> None:
        contract = self.contract
        self.assertEqual(
            contract.behavioral_source_binding_path,
            (
                dec.legacy_registry.REPOSITORY_ROOT
                / dec.BEHAVIORAL_SOURCE_BINDING_RELATIVE_PATH
            ).resolve(),
        )
        self.assertEqual(len(contract.behavioral_source_binding_sha256), 64)
        self.assertEqual(
            {(item.role, item.path) for item in contract.behavioral_sources},
            set(dec.EXPECTED_BEHAVIORAL_SOURCE_ROLE_PATHS.items()),
        )
        self.assertEqual(
            hashlib.sha256(contract.behavioral_source_binding_path.read_bytes()).hexdigest(),
            contract.behavioral_source_binding_sha256,
        )
        self.assertEqual(contract.runtime_binding_sha256, dec.RUNTIME_BINDING_SHA256)
        self.assertEqual(contract.runtime_binding_schema, "scenesense.splitfusion_live_dispatch_runtime_binding.v1")
        self.assertEqual(contract.action_contract.catalog_sha256, ac.CATALOG_SHA256)
        self.assertEqual(len(contract.modes), 12)
        self.assertEqual(tuple(mode.mode_id for mode in contract.modes), tuple(range(12)))
        self.assertEqual(
            [(mode.family, mode.quantizer) for mode in contract.modes],
            [
                (family, quantizer)
                for family in contract.action_contract.family_order
                for quantizer in contract.action_contract.quantizer_order
            ],
        )
        self.assertEqual(len(contract.startup_artifacts), 27)
        self.assertEqual(
            len({item.path for item in contract.startup_artifacts}),
            len(contract.startup_artifacts),
        )
        self.assertEqual(
            len({item.role for item in contract.startup_artifacts}),
            len(contract.startup_artifacts),
        )
        self.assertEqual(
            hashlib.sha256(
                (dec.legacy_registry.REPOSITORY_ROOT / dec.ACTION_CONTRACT_SOURCE_RELATIVE_PATH).read_bytes()
            ).hexdigest(),
            dec.ACTION_CONTRACT_SOURCE_SHA256,
        )

        for mode in contract.modes:
            action_mode = contract.action_contract.mode(mode.mode_id)
            self.assertEqual(
                (mode.family, mode.family_id, mode.quantizer, mode.bit_width),
                (
                    action_mode.family,
                    action_mode.family_id,
                    action_mode.quantizer,
                    action_mode.bit_width,
                ),
            )
            self.assertEqual(mode.wire.layout, "CURRENT_CELL_MAJOR")
            self.assertEqual(mode.zstd_level, 1)
            expected_codec_id = {
                ("noAE", "UINT8"): 1,
                ("AE128", "UINT8"): 2,
                ("AE64", "UINT8"): 2,
                ("AE32", "UINT8"): 2,
            }.get((mode.family, mode.quantizer), 3)
            expected_magic = (
                "HQ8\\0"
                if (mode.family, mode.quantizer) == ("noAE", "UINT8")
                else "AE8\\0"
                if mode.quantizer == "UINT8"
                else "HQLB"
            )
            self.assertEqual(mode.wire.codec_id, expected_codec_id)
            self.assertEqual(mode.wire.magic_ascii, expected_magic)
            self.assertEqual(mode.wire.version, 1)
            self.assertEqual(
                mode.bit_width,
                {"UINT8": 8, "UINT6": 6, "UINT4": 4}[mode.quantizer],
            )
            self.assertEqual(len(mode.invariant_proof.invariant_sha256), 64)
            self.assertEqual(
                mode.invariant_proof.anchor_q_e4,
                contract.action_contract.q_anchor_order,
            )
            self.assertEqual(len(mode.invariant_proof.anchor_action_ids), 6)
            self.assertEqual(len(mode.invariant_proof.anchor_profile_ids), 6)
            self.assertEqual(
                hashlib.sha256(
                    mode.invariant_proof.invariant_descriptor_json.encode("ascii")
                ).hexdigest(),
                mode.invariant_proof.invariant_sha256,
            )
            with self.assertRaises(dataclasses.FrozenInstanceError):
                mode.mode_id = 99  # type: ignore[misc]

    def test_all_q_e4_values_derive_exact_counts_without_float_requantization(self) -> None:
        """Exercise all 12 x 9,801 representable mode/wire-quality pairs."""
        anchors = set(self.contract.action_contract.q_anchor_order)
        for mode_id in range(ac.EXPECTED_MODE_COUNT):
            previous_drop = -1
            for q_e4 in range(ac.Q_E4_MIN, ac.Q_E4_MAX + 1):
                profile = self.contract.resolve_q_e4(mode_id, q_e4)
                expected_drop = (q_e4 * ac.SPATIAL_CELLS + 5000) // 10000
                self.assertEqual(profile.mode_id, mode_id)
                self.assertEqual(profile.q_e4, q_e4)
                self.assertEqual(profile.q_exec, q_e4 / 10000)
                self.assertEqual(profile.drop_count, expected_drop)
                self.assertEqual(profile.keep_count, ac.SPATIAL_CELLS - expected_drop)
                self.assertEqual(
                    profile.keep_count + profile.drop_count, ac.SPATIAL_CELLS
                )
                self.assertEqual(profile.action_id is not None, q_e4 in anchors)
                self.assertEqual(profile.profile_id is not None, q_e4 in anchors)
                self.assertEqual(
                    profile.measurement_status,
                    dec.MEASURED_ANCHOR
                    if q_e4 in anchors
                    else dec.UNMEASURED_OFF_ANCHOR,
                )
                self.assertEqual(
                    profile.contract_status,
                    dec.BEHAVIORAL_SOURCE_BINDING_STATUS,
                )
                self.assertEqual(len(profile.execution_bundle_sha256), 64)
                self.assertGreaterEqual(profile.drop_count, previous_drop)
                previous_drop = profile.drop_count

    def test_every_mode_resolves_continuous_q_and_ranker_boundary(self) -> None:
        for mode in self.contract.modes:
            q0 = self.contract.resolve_q_e4(mode.mode_id, 0)
            off_anchor = self.contract.resolve_q_e4(mode.mode_id, 2345)
            qmax = self.contract.resolve_q_e4(mode.mode_id, 9800)
            self.assertTrue(q0.ranker_bypassed)
            self.assertIsNone(q0.ranker_checkpoint)
            self.assertFalse(off_anchor.ranker_bypassed)
            self.assertEqual(off_anchor.ranker_checkpoint, mode.ranker_checkpoint)
            self.assertFalse(qmax.ranker_bypassed)
            self.assertEqual(qmax.ranker_checkpoint, mode.ranker_checkpoint)
            self.assertEqual(
                (off_anchor.family, off_anchor.quantizer),
                (mode.family, mode.quantizer),
            )
            self.assertIsNone(off_anchor.action_id)
            self.assertIsNone(off_anchor.profile_id)

    def test_all_seventy_two_anchors_reconstruct_exact_catalog_identity(self) -> None:
        observed = set()
        for anchor in self.contract.action_contract.anchors:
            profile = self.contract.resolve_q_e4(anchor.mode.mode_id, anchor.q_e4)
            self.assertTrue(profile.is_registered_anchor)
            self.assertEqual(profile.action_id, anchor.action_id)
            self.assertEqual(profile.profile_id, anchor.profile_id)
            self.assertEqual(profile.measurement_status, dec.MEASURED_ANCHOR)
            observed.add((profile.action_id, profile.profile_id))
        self.assertEqual(len(observed), 72)
        self.assertEqual({item[0] for item in observed}, set(range(72)))

    def test_off_anchor_never_uses_nearest_anchor_or_fake_profile(self) -> None:
        anchor_values = self.contract.action_contract.q_anchor_order
        probes = sorted(
            {
                candidate
                for anchor in anchor_values
                for candidate in (anchor - 1, anchor + 1)
                if ac.Q_E4_MIN <= candidate <= ac.Q_E4_MAX
                and candidate not in anchor_values
            }
            | {1, 2345, 5001, 9799}
        )
        for mode in self.contract.modes:
            for q_e4 in probes:
                profile = self.contract.resolve_q_e4(mode.mode_id, q_e4)
                self.assertIsNone(profile.action_id)
                self.assertIsNone(profile.profile_id)
                self.assertFalse(profile.is_registered_anchor)
                self.assertEqual(profile.measurement_status, dec.UNMEASURED_OFF_ANCHOR)
                descriptor = profile.execution_bundle.to_canonical_dict()
                self.assertIsNone(descriptor["anchor_identity"])

        with self.assertRaises(dec.DynamicExecutionContractError):
            self.contract.resolve_q_e4(0, 0.3)  # type: ignore[arg-type]
        with self.assertRaises(dec.DynamicExecutionContractError):
            self.contract.resolve_q_e4(0, True)  # type: ignore[arg-type]
        with self.assertRaises(dec.DynamicExecutionContractError):
            self.contract.resolve_q_e4(0, 9801)
        with self.assertRaises(dec.DynamicExecutionContractError):
            self.contract.resolve_q_e4(12, 0)

    def test_bundle_is_canonical_deterministic_and_artifact_complete(self) -> None:
        first = self.contract.resolve_q_e4(7, 4321)
        second = self.contract.resolve_q_e4(7, 4321)
        other_q = self.contract.resolve_q_e4(7, 4322)
        other_mode = self.contract.resolve_q_e4(8, 4321)
        self.assertEqual(first, second)
        self.assertEqual(first.execution_bundle_sha256, second.execution_bundle_sha256)
        self.assertNotEqual(first.execution_bundle_sha256, other_q.execution_bundle_sha256)
        self.assertNotEqual(first.execution_bundle_sha256, other_mode.execution_bundle_sha256)
        canonical = json.dumps(
            first.execution_bundle.to_canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        self.assertEqual(canonical, first.execution_bundle.canonical_json)
        self.assertEqual(
            hashlib.sha256(canonical.encode("ascii")).hexdigest(),
            first.execution_bundle_sha256,
        )
        descriptor = first.execution_bundle.to_canonical_dict()
        self.assertEqual(descriptor["catalog"]["sha256"], ac.CATALOG_SHA256)
        self.assertEqual(
            descriptor["runtime_binding"]["sha256"], dec.RUNTIME_BINDING_SHA256
        )
        self.assertEqual(
            descriptor["action_contract_source"]["sha256"],
            dec.ACTION_CONTRACT_SOURCE_SHA256,
        )
        behavioral = descriptor["behavioral_source_binding"]
        self.assertEqual(
            behavioral["path"], dec.BEHAVIORAL_SOURCE_BINDING_RELATIVE_PATH
        )
        self.assertEqual(
            behavioral["schema"], dec.BEHAVIORAL_SOURCE_BINDING_SCHEMA
        )
        self.assertEqual(
            behavioral["status"], dec.BEHAVIORAL_SOURCE_BINDING_STATUS
        )
        self.assertEqual(
            behavioral["sha256"],
            self.contract.behavioral_source_binding_sha256,
        )
        self.assertEqual(
            {(item["role"], item["path"]) for item in behavioral["sources"]},
            set(dec.EXPECTED_BEHAVIORAL_SOURCE_ROLE_PATHS.items()),
        )
        self.assertEqual(
            len(descriptor["startup_artifacts"]), len(self.contract.startup_artifacts)
        )

    def test_behavioral_source_closure_is_identical_across_all_bundles(self) -> None:
        expected = None
        for mode_id in range(ac.EXPECTED_MODE_COUNT):
            for q_e4 in (0, 1, 3000, 4321, 9800):
                descriptor = self.contract.resolve_q_e4(
                    mode_id, q_e4
                ).execution_bundle.to_canonical_dict()
                closure = descriptor["behavioral_source_binding"]
                if expected is None:
                    expected = closure
                self.assertEqual(closure, expected)
        self.assertIsNotNone(expected)

    def test_manufactured_or_replaced_profile_fails_authoritative_verification(self) -> None:
        genuine = self.contract.resolve_q_e4(2, 2345)
        fake_hash = dataclasses.replace(genuine, execution_bundle_sha256="0" * 64)
        with self.assertRaises(dec.DynamicExecutionContractError):
            self.contract.verify_profile(fake_hash)
        fake_anchor = dataclasses.replace(genuine, action_id=2, profile_id="fake")
        with self.assertRaises(dec.DynamicExecutionContractError):
            self.contract.verify_profile(fake_anchor)
        self.contract.verify_profile(genuine)

    def test_object_setattr_tampering_fails_authoritative_verification(self) -> None:
        top_level = self.contract.resolve_q_e4(2, 2345)
        object.__setattr__(top_level, "q_exec", -1.0)
        with self.assertRaises(dec.DynamicExecutionContractError):
            self.contract.verify_profile(top_level)

        nested = self.contract.resolve_q_e4(2, 2345)
        object.__setattr__(
            nested.execution_bundle,
            "keep_count",
            nested.execution_bundle.keep_count + 1,
        )
        with self.assertRaises(dec.DynamicExecutionContractError):
            self.contract.verify_profile(nested)

    def test_six_row_invariant_proof_rejects_a_contradictory_anchor(self) -> None:
        mode = self.contract.action_contract.mode(0)
        rows_by_q = {
            int(row["q_e4"]): copy.deepcopy(row)
            for row in self.catalog_document["profiles"]
            if row["family"] == mode.family and row["quantizer"] == mode.quantizer
        }
        rows = [rows_by_q[q] for q in self.contract.action_contract.q_anchor_order]
        rows[-1]["decoder_identity"] = "TAMPERED_DECODER"
        with self.assertRaisesRegex(
            dec.DynamicExecutionContractError, "decoder_identity"
        ):
            dec._prove_mode_invariants(  # type: ignore[attr-defined]
                mode=mode,
                anchor_q_e4=self.contract.action_contract.q_anchor_order,
                rows=rows,
            )

    def test_runtime_binding_tamper_fails_before_resolution(self) -> None:
        document = json.loads(
            dec.legacy_registry.DEFAULT_RUNTIME_BINDING.read_text(encoding="utf-8")
        )
        document["status"] = "TAMPERED"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_binding.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                dec.DynamicExecutionContractError, "runtime-binding SHA-256 drift"
            ):
                dec.load_dynamic_execution_contract(path)

    def test_action_contract_source_drift_fails_closed(self) -> None:
        source = (
            dec.legacy_registry.REPOSITORY_ROOT
            / dec.ACTION_CONTRACT_SOURCE_RELATIVE_PATH
        ).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_contract.py"
            path.write_bytes(source + b"\n# tampered\n")
            with mock.patch.object(dec, "_action_contract_source_path", return_value=path):
                with self.assertRaisesRegex(
                    dec.DynamicExecutionContractError,
                    "action-contract source drift",
                ):
                    dec.load_dynamic_execution_contract()

    def test_each_behavioral_source_binding_tamper_fails_closed(self) -> None:
        document = json.loads(
            self.contract.behavioral_source_binding_path.read_text(
                encoding="utf-8"
            )
        )
        observed_roles = set()
        for index, source in enumerate(document["sources"]):
            observed_roles.add(source["role"])
            tampered = copy.deepcopy(document)
            tampered["sources"][index]["sha256"] = "0" * 64
            with self.subTest(role=source["role"]), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "behavioral_sources.json"
                path.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(
                    dec.DynamicExecutionContractError,
                    "behavioral source hash drift",
                ):
                    dec.load_dynamic_execution_contract(
                        behavioral_source_binding_path=path
                    )
        self.assertEqual(
            observed_roles,
            set(dec.EXPECTED_BEHAVIORAL_SOURCE_ROLE_PATHS),
        )

    def test_each_behavioral_source_byte_tamper_fails_closed(self) -> None:
        original_repository_path = dec._repository_path  # type: ignore[attr-defined]
        for role, relative_path in dec.EXPECTED_BEHAVIORAL_SOURCE_ROLE_PATHS.items():
            source = original_repository_path(relative_path).read_bytes()
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                tampered_path = Path(directory) / Path(relative_path).name
                tampered_path.write_bytes(source + b"\n# byte-tamper regression\n")

                def redirected(
                    candidate: str,
                    *,
                    _target: str = relative_path,
                    _path: Path = tampered_path,
                ) -> Path:
                    if candidate == _target:
                        return _path
                    return original_repository_path(candidate)

                with mock.patch.object(
                    dec, "_repository_path", side_effect=redirected
                ):
                    with self.assertRaisesRegex(
                        dec.DynamicExecutionContractError,
                        "behavioral source hash drift",
                    ):
                        dec.load_dynamic_execution_contract()

    def test_behavioral_source_binding_requires_exact_role_path_closure(self) -> None:
        document = json.loads(
            self.contract.behavioral_source_binding_path.read_text(
                encoding="utf-8"
            )
        )
        mutations = []
        missing = copy.deepcopy(document)
        missing["sources"] = missing["sources"][:-1]
        mutations.append(missing)
        wrong_path = copy.deepcopy(document)
        wrong_path["sources"][0]["path"] = dec.ACTION_CONTRACT_SOURCE_RELATIVE_PATH
        wrong_path["sources"][0]["sha256"] = dec.ACTION_CONTRACT_SOURCE_SHA256
        mutations.append(wrong_path)
        extra = copy.deepcopy(document)
        extra["sources"].append(
            {
                "role": "undeclared_extra_source",
                "path": dec.ACTION_CONTRACT_SOURCE_RELATIVE_PATH,
                "sha256": dec.ACTION_CONTRACT_SOURCE_SHA256,
            }
        )
        mutations.append(extra)
        for mutation in mutations:
            with self.subTest(source_count=len(mutation["sources"])), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "behavioral_sources.json"
                path.write_text(json.dumps(mutation), encoding="utf-8")
                with self.assertRaisesRegex(
                    dec.DynamicExecutionContractError,
                    "behavioral source role/path closure drift",
                ):
                    dec.load_dynamic_execution_contract(
                        behavioral_source_binding_path=path
                    )

    def test_unreviewed_manifest_location_is_rejected_even_if_bytes_match(self) -> None:
        source = self.contract.behavioral_source_binding_path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behavioral_sources.json"
            path.write_bytes(source)
            with self.assertRaisesRegex(
                dec.DynamicExecutionContractError,
                "unreviewed path",
            ):
                dec.load_dynamic_execution_contract(
                    behavioral_source_binding_path=path
                )

    def test_machine_readable_contract_only_status_is_fail_closed(self) -> None:
        document = json.loads(
            self.contract.behavioral_source_binding_path.read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            document["status"], dec.BEHAVIORAL_SOURCE_BINDING_STATUS
        )
        self.assertEqual(
            self.contract.resolve_q_e4(0, 0).contract_status,
            "CONTRACT_ONLY_NOT_LIVE_RUNTIME_INTEGRATED",
        )
        document["status"] = "LIVE_RUNTIME_INTEGRATED"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behavioral_sources.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                dec.DynamicExecutionContractError,
                "behavioral source-binding status drift",
            ):
                dec.load_dynamic_execution_contract(
                    behavioral_source_binding_path=path
                )

    def test_startup_artifact_source_drift_fails_closed(self) -> None:
        original = dec.legacy_registry.sha256_file

        def altered(path: Path) -> str:
            observed = original(path)
            if path.name == "epoch_026.pt":
                return "0" * 64
            return observed

        with mock.patch.object(dec.legacy_registry, "sha256_file", side_effect=altered):
            with self.assertRaisesRegex(
                dec.DynamicExecutionContractError,
                "startup artifact hash drift",
            ):
                dec.load_dynamic_execution_contract()

    def test_catalog_drift_fails_closed_independently_of_runtime_registry(self) -> None:
        source = self.contract.action_contract.catalog_path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_bytes(source + b"\n")
            with mock.patch.object(ac, "default_catalog_path", return_value=path):
                with self.assertRaisesRegex(
                    dec.DynamicExecutionContractError,
                    "action catalog failed closed",
                ):
                    dec.load_dynamic_execution_contract()

    def test_import_has_no_runtime_evidence_io_service_or_cuda_side_effect(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        probe = r'''
import json, sys
violations = []

def hook(event, args):
    try:
        if event == "open":
            path = str(args[0]).lower()
            if "site-packages" in path or "dist-packages" in path:
                return
            if (
                "action_catalog" in path
                or "runtime_binding" in path
                or "dynamic_execution_source_binding" in path
                or "/experiments/" in path
                or path.endswith(".pt")
            ):
                violations.append([event, path])
        elif event in (
            "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
            "socket.socket", "socket.connect"
        ):
            violations.append([event, str(args)[:160]])
    except Exception:
        pass

sys.addaudithook(hook)
import rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract as m
assert m.DYNAMIC_EXECUTION_PROFILE_SCHEMA.endswith(".v1")
assert m.BEHAVIORAL_SOURCE_BINDING_STATUS == "CONTRACT_ONLY_NOT_LIVE_RUNTIME_INTEGRATED"
# The package currently imports torch eagerly; that is permitted.  CUDA
# initialization, artifact reads and runtime/service activity are not.
torch_was_eagerly_imported = "torch" in sys.modules
import torch
assert not torch.cuda.is_initialized()
print("TORCH_EAGER:" + json.dumps(torch_was_eagerly_imported))
print("VIOLATIONS:" + json.dumps(violations))
'''
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        marker = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("VIOLATIONS:")
        ]
        self.assertEqual(len(marker), 1, completed.stdout[-2000:])
        self.assertEqual(json.loads(marker[0][len("VIOLATIONS:") :]), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
