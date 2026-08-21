from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from rl_agent.ue_a2_technical_smoke import (
    A2TechnicalSmokeError,
    ActualUDPLoopback,
    DEFAULT_CONFIG,
    ROOT,
    _import_pinned_source,
    _model_runtime_args,
    build_structural_payload,
    chunk_message,
    inspect_fixture,
    inspect_socket_buffers,
    load_config,
    reassemble_chunks,
    run_negative_contract_tests,
    run_model_smoke,
    run_preflight,
    run_production_codec_matrix,
    run_runtime_map_probes,
    structural_serialized_features,
    verify_registry_matrix,
)
from rl_agent.ue_split_wire_contract import (
    SplitWireContractError,
    validate_feature_payload,
    validate_runtime_binding,
    validate_serialized_feature_headers,
)


class UEA2TechnicalSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(DEFAULT_CONFIG)
        cls.profiles, cls.registry_audit = verify_registry_matrix(cls.config)

    def test_config_and_fixture_are_hash_bound_real_ue_inputs(self) -> None:
        fixture = inspect_fixture(self.config)
        self.assertEqual(fixture["status"], "PASS")
        self.assertEqual(
            fixture["sha256"],
            "7fcfad2255c6626b8b87ff3a1c85ec7d32e17c8c2b4eee2875f5f132be423b41",
        )
        self.assertEqual(fixture["bytes"], 2499548)
        self.assertEqual(fixture["required_arrays"]["frame_bgr"]["shape"], [720, 1280, 3])
        self.assertEqual(fixture["required_arrays"]["radar_tensor"]["shape"], [4, 432, 768])
        self.assertFalse(fixture["ground_truth_consumed"])
        self.assertFalse(fixture["phase2_logic_consumed"])

    def test_config_is_read_once_and_count_or_source_drift_fails_closed(self) -> None:
        import rl_agent.ue_a2_technical_smoke as smoke

        with mock.patch.object(smoke, "_read_json", wraps=smoke._read_json) as reader:
            load_config(DEFAULT_CONFIG)
        reader.assert_called_once_with(DEFAULT_CONFIG.resolve())

        drifted = copy.deepcopy(self.config)
        drifted["model_smoke"]["tail_decodes"] = 71
        with mock.patch.object(smoke, "_read_json", return_value=drifted):
            with self.assertRaises(A2TechnicalSmokeError) as raised:
                load_config(DEFAULT_CONFIG)
        self.assertEqual(raised.exception.code, "CONFIG_MODEL_SMOKE_COUNT_MISMATCH")

        with self.assertRaises(A2TechnicalSmokeError) as source_drift:
            _import_pinned_source(
                Path(self.config["sources"]["codec_path"]),
                expected_sha256="0" * 64,
                module_prefix="ue_a2_drift_rejection",
            )
        self.assertEqual(source_drift.exception.code, "PINNED_SOURCE_HASH_MISMATCH")

    def test_registry_preflight_keeps_exact_unfiltered_72_grid_pending(self) -> None:
        self.assertEqual(len(self.profiles), 72)
        self.assertEqual(self.registry_audit["profiles"], 72)
        self.assertEqual(self.registry_audit["unique_action_contracts"], 72)
        self.assertEqual(len(self.registry_audit["checkpoint_audits"]), 4)
        self.assertFalse(self.registry_audit["quality_mask_applied"])
        self.assertEqual(
            {profile.row["technical_validity_status"] for profile in self.profiles},
            {"REGISTERED_PENDING_SMOKE"},
        )

    def test_structural_payloads_cover_all_headers_and_identity_before_decode(self) -> None:
        for profile in self.profiles:
            with self.subTest(profile=profile.profile_id):
                serialized = structural_serialized_features(profile)
                validate_serialized_feature_headers(profile, serialized)
                payload = build_structural_payload(profile)
                validated = validate_feature_payload(profile, payload)
                self.assertEqual(
                    validated["profile_identity"]["profile_id"], profile.profile_id
                )

        sample = build_structural_payload(self.profiles[0])
        sample["profile_identity"] = None
        sample["features"] = "must not be inspected"
        with self.assertRaises(SplitWireContractError) as raised:
            validate_feature_payload(self.profiles[0], sample)
        self.assertEqual(raised.exception.code, "WIRE_IDENTITY_MISSING")

    def test_chunk_roundtrip_accepts_reordering_and_rejects_loss_or_duplicate(self) -> None:
        original = bytes((index * 17) % 256 for index in range(150123))
        chunks = chunk_message(original, 60000)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(reassemble_chunks(reversed(chunks)), original)
        with self.assertRaises(A2TechnicalSmokeError) as missing:
            reassemble_chunks(chunks[:-1])
        self.assertEqual(missing.exception.code, "CHUNK_SET_INCOMPLETE")
        with self.assertRaises(A2TechnicalSmokeError) as duplicate:
            reassemble_chunks([chunks[0], chunks[0], *chunks[1:]])
        self.assertEqual(duplicate.exception.code, "CHUNK_DUPLICATE")

    def test_production_cpu_codec_and_in_memory_wire_cover_all_72(self) -> None:
        rows, summary = run_production_codec_matrix(self.profiles, self.config)
        self.assertEqual(len(rows), 72)
        self.assertEqual(summary["profiles"], 72)
        self.assertFalse(summary["model_inference_executed"])
        self.assertGreater(summary["max_zstd_message_bytes"], 0)
        self.assertEqual(
            {row["production_codec_status"] for row in rows},
            {"PASS_CPU_SYNTHETIC_FEATURES"},
        )
        self.assertEqual(
            {row["technical_validity_status"] for row in rows},
            {"REGISTERED_PENDING_SMOKE"},
        )
        self.assertEqual({row["model_front_status"] for row in rows}, {"NOT_EXECUTED"})
        self.assertTrue(all(float(row["max_quantization_abs_error"]) >= 0.0 for row in rows))
        self.assertEqual(summary["codec_path"], self.config["sources"]["codec_path"])
        self.assertEqual(summary["codec_sha256"], self.config["sources"]["codec_sha256"])

    def test_codec_import_is_pinned_despite_same_named_parent_module(self) -> None:
        fake = types.ModuleType("carla_split_inference_udp_data_collect")
        fake.__file__ = str(ROOT.parent / "carla_split_inference_udp_data_collect.py")
        module_name = "carla_split_inference_udp_data_collect"
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = fake
        try:
            original_path = list(sys.path)
            codec = _import_pinned_source(
                Path(self.config["sources"]["codec_path"]),
                expected_sha256=self.config["sources"]["codec_sha256"],
                module_prefix="ue_a2_production_codec",
            )
            self.assertEqual(
                Path(codec.__file__).resolve(),
                ROOT / "carla_split_inference_udp_data_collect.py",
            )
            self.assertTrue(hasattr(codec, "QUANT_MODE_PER_CHANNEL_UINT6"))
            self.assertEqual(sys.path, original_path)
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous

    def test_negative_matrix_rejects_every_injected_contract_violation(self) -> None:
        registry_sha256_before = __import__("hashlib").sha256(
            Path(self.config["registry"]["path"]).read_bytes()
        ).hexdigest()
        result = run_negative_contract_tests(self.profiles, self.config)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["tests"], 34)
        self.assertEqual(result["decode_or_map_calls_after_rejection"], 0)
        self.assertEqual({row["status"] for row in result["records"]}, {"PASS"})
        records = {row["name"]: row for row in result["records"]}
        expected_new_cases = {
            "missing_strict_binding_arguments": "REGISTERED_PROFILE_ARGUMENTS_INCOMPLETE",
            "duplicate_profile_id_registry": "PROFILE_ID_DUPLICATE",
            "corrupt_registry_row_fingerprint": "REGISTRY_ROW_FINGERPRINT_MISMATCH",
            "wrong_feature_names": "FEATURE_LEVELS_MISMATCH",
        }
        for name, code in expected_new_cases.items():
            with self.subTest(name=name):
                self.assertEqual(records[name]["expected_code"], code)
                self.assertEqual(records[name]["observed_code"], code)
                self.assertEqual(records[name]["status"], "PASS")
        registry_sha256_after = __import__("hashlib").sha256(
            Path(self.config["registry"]["path"]).read_bytes()
        ).hexdigest()
        self.assertTrue(result["temporary_registry_copies_only"])
        self.assertEqual(registry_sha256_before, registry_sha256_after)
        self.assertEqual(result["source_registry_sha256_before"], registry_sha256_before)
        self.assertEqual(result["source_registry_sha256_after"], registry_sha256_after)

    def test_model_runtime_namespaces_bind_strict_front_and_edge(self) -> None:
        profile = self.profiles[0]
        checkpoint = ROOT / profile.row["checkpoint_path"]
        for role in ("front", "back"):
            with self.subTest(role=role):
                args = _model_runtime_args(profile, role=role)
                audit = validate_runtime_binding(
                    profile,
                    args,
                    checkpoint_path=checkpoint,
                    role=role,
                )
                self.assertEqual(audit["profile_id"], profile.profile_id)
                self.assertTrue(args.require_ue_profile_binding)
                self.assertEqual(args.ue_profile_registry_csv, str(profile.registry_path))

    def test_actual_udp_timeout_is_infrastructure_not_profile_invalidity(self) -> None:
        endpoint = object.__new__(ActualUDPLoopback)
        endpoint.config = self.config
        endpoint.requested_socket_buffer_bytes = 8_388_608
        endpoint.reported_receive_buffer_bytes = 8_388_608
        endpoint.sender = mock.Mock()
        endpoint.sender.send.return_value = (100, 1)
        endpoint.receiver = mock.Mock()
        endpoint.receiver.receive.return_value = None
        with self.assertRaises(A2TechnicalSmokeError) as raised:
            endpoint.roundtrip(
                {"profile_identity": {"profile_id": self.profiles[0].profile_id}},
                expected_compressed_bytes=100,
            )
        self.assertEqual(raised.exception.code, "ACTUAL_UDP_LOSS_OR_TIMEOUT")
        self.assertEqual(raised.exception.classification, "INFRASTRUCTURE")

    def test_runtime_map_probe_accounts_for_all_profiles_without_sockets(self) -> None:
        path_before = list(sys.path)
        fusion_modules_before = {
            name for name in sys.modules if name.startswith("pole_lraspp_multimodal_fusion")
        }
        statuses, summary = run_runtime_map_probes(self.profiles, self.config)
        self.assertEqual(len(statuses), 72)
        self.assertEqual(summary["profiles_passed"] + summary["profiles_failed"], 72)
        self.assertFalse(summary["carla_connected"])
        self.assertFalse(summary["sockets_created"])
        self.assertTrue(set(statuses.values()) <= {"PASS", "FAIL"})
        if summary["status"] == "PASS":
            self.assertEqual(set(statuses.values()), {"PASS"})
        else:
            self.assertEqual(summary["status"], "BLOCKED_IMPLEMENTATION")
            self.assertGreater(summary["profiles_failed"], 0)
        self.assertEqual(sys.path, path_before)
        self.assertEqual(
            {
                name
                for name in sys.modules
                if name.startswith("pole_lraspp_multimodal_fusion")
            },
            fusion_modules_before,
        )

    def test_socket_permission_or_small_buffer_is_infrastructure_not_profile_failure(self) -> None:
        with mock.patch("socket.socket", side_effect=PermissionError("sandbox")):
            result = inspect_socket_buffers(
                self.config, observed_max_message_bytes=1000
            )
        self.assertEqual(result["status"], "BLOCKED_INFRASTRUCTURE_SOCKET_PERMISSION")
        self.assertIn("SOCKET_PERMISSION_DENIED", result["blockers"])
        self.assertFalse(result["actual_udp_executed"])

    def test_preflight_bundle_never_claims_technical_validity(self) -> None:
        fake_rows = []
        for profile in self.profiles:
            row = {
                "action_index": int(profile.row["action_index"]),
                "profile_id": profile.profile_id,
                "model_family": profile.row["model_family"],
                "quantization_mode": profile.row["quantization_mode"],
                "quantization_bits": int(profile.row["quantization_bits"]),
                "roi_drop_fraction": profile.row["roi_drop_fraction"],
                "checkpoint_sha256": profile.row["checkpoint_sha256"],
                "action_contract_sha256": profile.action_contract_sha256,
                "registry_binding_status": "PASS",
                "fixture_status": "PASS",
                "production_codec_status": "PASS_CPU_SYNTHETIC_FEATURES",
                "in_memory_wire_status": "PASS",
                "wire_shape_status": "PASS",
                "map_schema_status": "PENDING_RUNTIME_PROBE",
                "model_front_status": "NOT_EXECUTED",
                "roi_execution_status": "NOT_EXECUTED",
                "ae_execution_status": "NOT_EXECUTED",
                "tail_execution_status": "NOT_EXECUTED",
                "actual_udp_status": "NOT_EXECUTED",
                "technical_validity_status": "REGISTERED_PENDING_SMOKE",
                "blocking_codes": "MODEL_AND_ACTUAL_UDP_PENDING",
                "pickle_bytes": 1,
                "zstd_bytes": 1,
                "udp_chunks": 1,
                "payload_bytes_uncompressed": 1,
                "max_quantization_abs_error": 0.0,
                "compressed_sha256": "0" * 64,
            }
            fake_rows.append(row)
        negatives = {
            "schema": "test",
            "status": "PASS",
            "tests": 1,
            "decode_or_map_calls_after_rejection": 0,
            "records": [],
        }
        map_status = {profile.profile_id: "PASS" for profile in self.profiles}
        map_summary = {
            "status": "PASS",
            "profiles_passed": 72,
            "profiles_failed": 0,
            "failures": {},
            "details": {},
            "runtime_path": self.config["sources"]["runtime_path"],
            "runtime_sha256": "0" * 64,
            "carla_connected": False,
            "sockets_created": False,
        }
        environment = {
            "blockers": ["CUDA_UNAVAILABLE"],
            "model_inference_executed": False,
            "actual_udp_executed": False,
        }
        transport = {
            "status": "BLOCKED_INFRASTRUCTURE_SOCKET_PERMISSION",
            "blockers": ["SOCKET_PERMISSION_DENIED"],
            "actual_udp_executed": False,
        }
        with tempfile.TemporaryDirectory(prefix="ue-a2-test-") as directory:
            output = Path(directory) / "bundle"
            with (
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.run_production_codec_matrix",
                    return_value=(
                        fake_rows,
                        {
                            "status": "PASS",
                            "profiles": 72,
                            "max_zstd_message_bytes": 1,
                            "total_zstd_message_bytes": 72,
                            "model_inference_executed": False,
                        },
                    ),
                ),
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.run_negative_contract_tests",
                    return_value=negatives,
                ),
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.run_runtime_map_probes",
                    return_value=(map_status, map_summary),
                ),
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.inspect_runtime_environment",
                    return_value=environment,
                ),
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.inspect_socket_buffers",
                    return_value=transport,
                ),
            ):
                result = run_preflight(DEFAULT_CONFIG, output_dir=output)
            self.assertEqual(result["status"], "PREFLIGHT_BLOCKED")
            self.assertEqual(result["technical_valid_profiles"], 0)
            terminal = json.loads(
                (output / "UE_A2_PREFLIGHT_BLOCKED.json").read_text(encoding="utf-8")
            )
            self.assertEqual(terminal["claim_scope"], "PREFLIGHT_ONLY_NO_TECHNICAL_VALIDITY")
            self.assertEqual(terminal["technical_valid_profiles"], 0)
            self.assertFalse(terminal["model_inference_executed"])
            with (output / "ue_a2_profile_smoke.csv").open(newline="") as handle:
                rows = list(__import__("csv").DictReader(handle))
            self.assertEqual(len(rows), 72)
            self.assertEqual(
                {row["technical_validity_status"] for row in rows},
                {"REGISTERED_PENDING_SMOKE"},
            )

    def test_model_smoke_infrastructure_block_never_calls_cuda_matrix(self) -> None:
        environment = {
            "blockers": ["CUDA_UNAVAILABLE"],
            "model_inference_executed": False,
            "actual_udp_executed": False,
        }
        transport = {
            "status": "BLOCKED_INFRASTRUCTURE_SOCKET_PERMISSION",
            "blockers": ["SOCKET_PERMISSION_DENIED"],
            "actual_udp_executed": False,
        }
        with tempfile.TemporaryDirectory(prefix="ue-a2-model-block-") as directory:
            output = Path(directory) / "bundle"
            with (
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.inspect_runtime_environment",
                    return_value=environment,
                ),
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.inspect_socket_buffers",
                    return_value=transport,
                ),
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.run_cuda_model_matrix"
                ) as cuda_matrix,
            ):
                result = run_model_smoke(DEFAULT_CONFIG, output_dir=output)
            cuda_matrix.assert_not_called()
            self.assertEqual(result["status"], "BLOCKED_INFRASTRUCTURE")
            self.assertEqual(result["infrastructure_blocked_profiles"], 72)
            terminal = json.loads(
                (output / "UE_A2_BLOCKED_INFRASTRUCTURE.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(terminal["model_inference_executed"])
            self.assertFalse(terminal["actual_udp_executed"])
            self.assertFalse(terminal["successor_registry_emitted"])
            self.assertFalse((output / "UE_A2_PASSED.json").exists())

    def test_model_smoke_pass_requires_exact_72_row_and_udp_counts(self) -> None:
        environment = {
            "blockers": [],
            "cuda_available": True,
            "model_inference_executed": False,
            "actual_udp_executed": False,
        }
        transport = {
            "status": "READY_FOR_ACTUAL_UDP_SMOKE",
            "blockers": [],
            "actual_udp_executed": False,
        }

        def fake_cuda_matrix(_profiles, _config, base_rows):
            rows = [dict(row) for row in base_rows]
            for row in rows:
                row.update(
                    {
                        "model_front_status": "PASS_STRICT",
                        "model_edge_status": "PASS_STRICT",
                        "production_codec_status": "PASS_MODEL_FEATURES",
                        "in_memory_wire_status": "PASS",
                        "wire_shape_status": "PASS",
                        "roi_execution_status": "PASS_PATH_VERIFIED",
                        "ae_execution_status": "PASS_BINDING_VERIFIED",
                        "tail_execution_status": "PASS_FINITE",
                        "actual_udp_status": "PASS",
                        "map_schema_status": "PASS",
                        "technical_validity_status": "TECHNICALLY_VALID",
                        "blocking_codes": "",
                    }
                )
            summary = {
                "status": "PASS",
                "strict_front_loads": 4,
                "strict_edge_loads": 4,
                "native_backbone_encodes": 4,
                "roi_ae_paths": 24,
                "tail_decodes": 72,
                "model_inference_executed": True,
                "actual_udp_executed": True,
            }
            actual_udp = {
                "status": "PASS",
                "actual_udp_executed": True,
                "messages_sent": 72,
                "messages_received": 72,
            }
            return rows, summary, actual_udp

        with tempfile.TemporaryDirectory(prefix="ue-a2-model-pass-") as directory:
            output = Path(directory) / "bundle"
            with (
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.inspect_runtime_environment",
                    return_value=environment,
                ),
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.inspect_socket_buffers",
                    return_value=transport,
                ),
                mock.patch(
                    "rl_agent.ue_a2_technical_smoke.run_cuda_model_matrix",
                    side_effect=fake_cuda_matrix,
                ),
            ):
                result = run_model_smoke(DEFAULT_CONFIG, output_dir=output)
            self.assertEqual(result["status"], "PASSED")
            self.assertEqual(result["technical_valid_profiles"], 72)
            terminal = json.loads(
                (output / "UE_A2_PASSED.json").read_text(encoding="utf-8")
            )
            self.assertEqual(terminal["technical_valid_profiles"], 72)
            self.assertTrue(terminal["model_inference_executed"])
            self.assertTrue(terminal["actual_udp_executed"])
            self.assertFalse(terminal["successor_registry_emitted"])
            terminals = sorted(path.name for path in output.glob("UE_A2_*.json"))
            self.assertEqual(terminals, ["UE_A2_PASSED.json"])

    def test_full_cli_is_fail_closed_and_cannot_emit_a_false_pass(self) -> None:
        script = ROOT / "rl_agent/ue_a2_technical_smoke.py"
        result = subprocess.run(
            [sys.executable, str(script), "full"],
            cwd="/tmp",
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stderr)
        self.assertEqual(payload["error"], "DEPRECATED_FULL_COMMAND")
        self.assertNotIn("PASSED", result.stdout)


if __name__ == "__main__":
    unittest.main()
