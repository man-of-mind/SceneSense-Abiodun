from __future__ import annotations

import copy
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl_agent.ue_split_wire_contract import (
    A1_REGISTRY_CSV_SHA256,
    DEFAULT_REGISTRY_CSV,
    LAUNCH_BINDING_SCHEMA,
    ROOT,
    SplitWireContractError,
    WIRE_IDENTITY_SCHEMA,
    action_contract_sha256,
    build_wire_identity,
    build_launch_binding,
    load_registered_profiles,
    resolve_registered_profile,
    sha256_file,
    validate_declared_feature_shapes,
    validate_feature_payload,
    validate_runtime_binding,
    validate_serialized_feature_headers,
    validate_wire_identity,
    verify_registry_row_fingerprint,
)


def _profile(family: str, quant: str, q: str):
    matches = [
        profile
        for profile in load_registered_profiles()
        if profile.row["model_family"] == family
        and profile.row["quantization_mode"] == quant
        and profile.row["roi_drop_fraction"] == q
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one profile, found {len(matches)}")
    return matches[0]


def _runtime_args(profile, *, role: str = "back") -> SimpleNamespace:
    row = profile.row
    values = {
        "role": role,
        "ue_profile_id": profile.profile_id,
        "quantization_mode": row["quantization_mode"],
        "roi_threshold": float(row["roi_drop_fraction"]),
        "entropy_coder": row["entropy_coder"],
        "zstd_level": int(row["entropy_level"]),
        "chunk_bytes": int(row["udp_chunk_bytes"]),
        "model_input_width": int(row["input_width"]),
        "model_input_height": int(row["input_height"]),
        "ae_checkpoint": "",
    }
    if role in {"back", "loopback"}:
        values.update(
            {
                "object_score_threshold": float(row["object_score_threshold"]),
                "object_nms_radius_px": int(row["object_nms_radius_px"]),
                "topk_objects": int(row["topk_objects"]),
                "max_objects_drawn": int(row["max_objects_published"]),
            }
        )
    return SimpleNamespace(**values)


def _wire_field_lengths(shape: tuple[int, ...], bits: int) -> tuple[int, int]:
    _, channels, height, width = shape
    ranges = channels * 2 * 4
    total = channels * height * width
    if bits == 8:
        data = total
    elif bits == 6:
        data = ((total + 3) // 4) * 3
    elif bits == 4:
        data = (total + 1) // 2
    else:
        raise AssertionError(bits)
    return ranges, data


def _serialized_features(profile) -> dict[str, dict[str, bytes]]:
    bits = int(profile.row["quantization_bits"])
    result: dict[str, dict[str, bytes]] = {}
    for level, shape in profile.expected_wire_shapes.items():
        _, channels, height, width = shape
        ranges, data = _wire_field_lengths(shape, bits)
        result[level] = {
            "header": struct.pack("!IIIB", channels, height, width, bits),
            "ranges": bytes(ranges),
            "data": bytes(data),
        }
    return result


class UESplitWireContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_registered_profiles()
        cls.profile = _profile("ae32", "per_channel_uint4", "0.9")

    def assertContractCode(self, code: str, callback) -> None:  # noqa: N802
        with self.assertRaises(SplitWireContractError) as raised:
            callback()
        self.assertEqual(raised.exception.code, code, str(raised.exception))

    def test_a1_registry_is_hash_sealed_and_exactly_72_unique_actions(self) -> None:
        self.assertEqual(sha256_file(DEFAULT_REGISTRY_CSV), A1_REGISTRY_CSV_SHA256)
        self.assertEqual(len(self.profiles), 72)
        self.assertEqual(len({profile.profile_id for profile in self.profiles}), 72)
        self.assertEqual(
            len({profile.action_contract_sha256 for profile in self.profiles}),
            72,
        )
        self.assertEqual(
            len({json.dumps(profile.wire_identity, sort_keys=True) for profile in self.profiles}),
            72,
        )

    def test_exact_profile_resolution_and_unknown_profile_rejection(self) -> None:
        resolved = resolve_registered_profile(self.profile.profile_id)
        self.assertEqual(resolved.profile_id, self.profile.profile_id)
        self.assertEqual(resolved.row["action_index"], self.profile.row["action_index"])
        self.assertContractCode(
            "PROFILE_NOT_FOUND",
            lambda: resolve_registered_profile("not-a-registered-profile"),
        )

    def test_a1_row_fingerprint_is_verified_before_contract_use(self) -> None:
        verify_registry_row_fingerprint(self.profile.row)
        tampered = dict(self.profile.row)
        tampered["roi_drop_fraction"] = "0.7"
        self.assertContractCode(
            "REGISTRY_ROW_FINGERPRINT_MISMATCH",
            lambda: verify_registry_row_fingerprint(tampered),
        )

    def test_action_contract_hash_excludes_mutable_evidence_status_fields(self) -> None:
        baseline = action_contract_sha256(self.profile.row)
        annotated = dict(self.profile.row)
        annotated.update(
            {
                "wire_smoke_status": "PASS",
                "technical_validity_status": "TECHNICALLY_VALID",
                "technical_invalid_reason": "",
                "quality_mask_applied": "True",
                "runtime_sha256": "f" * 64,
                "front_profile_launch_args_json": "[]",
            }
        )
        self.assertEqual(action_contract_sha256(annotated), baseline)

        operational_change = dict(self.profile.row)
        operational_change["roi_drop_fraction"] = "0.7"
        self.assertNotEqual(action_contract_sha256(operational_change), baseline)

    def test_wire_identity_has_frozen_operational_fields(self) -> None:
        identity = self.profile.wire_identity
        self.assertEqual(identity["schema"], WIRE_IDENTITY_SCHEMA)
        self.assertEqual(identity["profile_id"], self.profile.profile_id)
        self.assertEqual(
            identity["action_contract_sha256"], self.profile.action_contract_sha256
        )
        self.assertEqual(identity["checkpoint_sha256"], self.profile.row["checkpoint_sha256"])
        self.assertEqual(identity["roi_drop_fraction"], "0.9")
        self.assertEqual(identity["quantization_mode"], "per_channel_uint4")
        self.assertEqual(identity["entropy_coder"], "zstd")
        self.assertEqual(identity["entropy_level"], 3)
        self.assertEqual(identity["udp_chunk_bytes"], 60000)
        self.assertIs(type(identity["udp_chunk_bytes"]), int)
        with self.assertRaises(TypeError):
            self.profile.row["roi_drop_fraction"] = "0.7"  # type: ignore[index]
        self.assertContractCode(
            "ACTION_CONTRACT_SHA256_MISMATCH",
            lambda: build_wire_identity(self.profile.row, "0" * 64),
        )

    def test_identity_comparison_is_exact_typed_and_rejects_every_mutation(self) -> None:
        expected = self.profile.wire_identity
        self.assertEqual(validate_wire_identity(dict(expected), expected), expected)

        for key, original in expected.items():
            with self.subTest(key=key):
                changed = dict(expected)
                if isinstance(original, int):
                    changed[key] = original + 1
                else:
                    changed[key] = f"{original}-changed"
                self.assertContractCode(
                    "WIRE_IDENTITY_VALUE_MISMATCH",
                    lambda changed=changed: validate_wire_identity(changed, expected),
                )

        wrong_type = dict(expected)
        wrong_type["entropy_level"] = "3"
        self.assertContractCode(
            "WIRE_IDENTITY_VALUE_MISMATCH",
            lambda: validate_wire_identity(wrong_type, expected),
        )
        wrong_chunk_type = dict(expected)
        wrong_chunk_type["udp_chunk_bytes"] = "60000"
        self.assertContractCode(
            "WIRE_IDENTITY_VALUE_MISMATCH",
            lambda: validate_wire_identity(wrong_chunk_type, expected),
        )
        missing = dict(expected)
        missing.pop("profile_id")
        self.assertContractCode(
            "WIRE_IDENTITY_KEYS_MISMATCH",
            lambda: validate_wire_identity(missing, expected),
        )
        extra = {**expected, "unversioned_hint": "ignored?"}
        self.assertContractCode(
            "WIRE_IDENTITY_KEYS_MISMATCH",
            lambda: validate_wire_identity(extra, expected),
        )
        self.assertContractCode(
            "WIRE_IDENTITY_MISSING",
            lambda: validate_wire_identity(None, expected),
        )

    def test_runtime_binding_verifies_front_and_edge_against_actual_checkpoint(self) -> None:
        checkpoint = ROOT / self.profile.row["checkpoint_path"]
        front = validate_runtime_binding(
            self.profile,
            _runtime_args(self.profile, role="front"),
            checkpoint_path=checkpoint,
        )
        edge = validate_runtime_binding(
            self.profile,
            _runtime_args(self.profile, role="back"),
            checkpoint_path=checkpoint,
        )
        self.assertEqual(front["checkpoint_sha256"], self.profile.row["checkpoint_sha256"])
        self.assertEqual(edge["action_contract_sha256"], self.profile.action_contract_sha256)
        self.assertEqual(edge["profile_identity"], self.profile.wire_identity)

    def test_runtime_binding_rejects_knob_decoder_and_external_ae_mismatches(self) -> None:
        checkpoint = ROOT / self.profile.row["checkpoint_path"]
        mutations = {
            "quantization_mode": "per_channel_uint8",
            "roi_threshold": 0.7,
            "entropy_coder": "zlib",
            "zstd_level": 2,
            "chunk_bytes": 59999,
            "model_input_width": 767,
            "model_input_height": 431,
            "object_score_threshold": 0.05,
            "object_nms_radius_px": 4,
            "topk_objects": 80,
            "max_objects_drawn": 30,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                args = vars(_runtime_args(self.profile, role="back")).copy()
                args[field] = value
                self.assertContractCode(
                    "RUNTIME_BINDING_MISMATCH",
                    lambda args=args: validate_runtime_binding(
                        self.profile, args, checkpoint_path=checkpoint
                    ),
                )

        args = vars(_runtime_args(self.profile, role="back")).copy()
        args["ae_checkpoint"] = "/tmp/external-ae.pt"
        self.assertContractCode(
            "EXTERNAL_AE_OVERRIDE_FORBIDDEN",
            lambda: validate_runtime_binding(self.profile, args, checkpoint_path=checkpoint),
        )

    def test_runtime_binding_rejects_wrong_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-wire-contract-") as directory:
            wrong = Path(directory) / "wrong.pt"
            wrong.write_bytes(b"not the registered checkpoint")
            self.assertContractCode(
                "CHECKPOINT_HASH_MISMATCH",
                lambda: validate_runtime_binding(
                    self.profile,
                    _runtime_args(self.profile, role="front"),
                    checkpoint_path=wrong,
                ),
            )

    def test_declared_shapes_validate_wire_native_and_edge_decoded_stages(self) -> None:
        for stage, shapes in (
            ("wire", self.profile.expected_wire_shapes),
            ("native", self.profile.expected_native_shapes),
            ("edge_decoded", self.profile.expected_edge_decoded_shapes),
        ):
            with self.subTest(stage=stage):
                observed = {key: list(value) for key, value in shapes.items()}
                self.assertEqual(
                    validate_declared_feature_shapes(self.profile, observed, stage=stage),
                    shapes,
                )

        wrong = {key: list(value) for key, value in self.profile.expected_wire_shapes.items()}
        wrong["high"][1] += 1
        self.assertContractCode(
            "FEATURE_SHAPE_MISMATCH",
            lambda: validate_declared_feature_shapes(self.profile, wrong),
        )
        extra = {
            **{key: list(value) for key, value in self.profile.expected_wire_shapes.items()},
            "extra": [1, 1, 1, 1],
        }
        self.assertContractCode(
            "FEATURE_LEVELS_MISMATCH",
            lambda: validate_declared_feature_shapes(self.profile, extra),
        )

    def test_serialized_per_channel_headers_and_lengths_validate_before_decode(self) -> None:
        serialized = _serialized_features(self.profile)
        headers = validate_serialized_feature_headers(self.profile, serialized)
        self.assertEqual(headers["high"], (32, 27, 48, 4))
        self.assertEqual(headers["low"], (40, 54, 96, 4))

        bad_shape = copy.deepcopy(serialized)
        bad_shape["high"]["header"] = struct.pack("!IIIB", 33, 27, 48, 4)
        self.assertContractCode(
            "FEATURE_WIRE_HEADER_MISMATCH",
            lambda: validate_serialized_feature_headers(self.profile, bad_shape),
        )

        bad_bits = copy.deepcopy(serialized)
        bad_bits["low"]["header"] = struct.pack("!IIIB", 40, 54, 96, 6)
        self.assertContractCode(
            "FEATURE_WIRE_HEADER_MISMATCH",
            lambda: validate_serialized_feature_headers(self.profile, bad_bits),
        )

        truncated = copy.deepcopy(serialized)
        truncated["high"]["data"] = truncated["high"]["data"][:-1]
        self.assertContractCode(
            "FEATURE_WIRE_LENGTH_MISMATCH",
            lambda: validate_serialized_feature_headers(self.profile, truncated),
        )

        extra_field = copy.deepcopy(serialized)
        extra_field["low"]["scale"] = b""
        self.assertContractCode(
            "FEATURE_WIRE_FIELDS_MISMATCH",
            lambda: validate_serialized_feature_headers(self.profile, extra_field),
        )

    def test_header_contract_covers_all_registered_bit_packers(self) -> None:
        cases = (
            ("per_channel_uint8", 8),
            ("per_channel_uint6", 6),
            ("per_channel_uint4", 4),
        )
        for quantizer, bits in cases:
            with self.subTest(quantizer=quantizer):
                profile = _profile("ae32", quantizer, "0.9")
                headers = validate_serialized_feature_headers(
                    profile, _serialized_features(profile)
                )
                self.assertEqual(headers["low"][-1], bits)
                self.assertEqual(headers["high"][-1], bits)

    def test_full_payload_validation_checks_identity_before_tensor_metadata(self) -> None:
        payload = {
            "profile_identity": dict(self.profile.wire_identity),
            "feature_shapes": {
                key: list(value) for key, value in self.profile.expected_wire_shapes.items()
            },
            "features": _serialized_features(self.profile),
            "batch_size": 1,
            "model_input_size": [768, 432],
        }
        evidence = validate_feature_payload(self.profile, payload)
        self.assertEqual(evidence["profile_identity"], self.profile.wire_identity)

        identity_first = dict(payload)
        identity_first["profile_identity"] = {
            **self.profile.wire_identity,
            "profile_id": "wrong-profile",
        }
        identity_first["features"] = "would fail if inspected"
        self.assertContractCode(
            "WIRE_IDENTITY_VALUE_MISMATCH",
            lambda: validate_feature_payload(self.profile, identity_first),
        )

    def test_launch_binding_is_fixed_json_data_with_host_and_container_argv(self) -> None:
        binding = build_launch_binding(self.profile)
        self.assertEqual(binding["schema"], LAUNCH_BINDING_SCHEMA)
        self.assertTrue(Path(binding["checkpoint_paths"]["host"]).is_absolute())
        self.assertTrue(binding["checkpoint_paths"]["container"].startswith("/work/abiodun/"))
        self.assertTrue(binding["registry_paths"]["container"].startswith("/work/abiodun/"))
        for key in ("front_args", "edge_args"):
            argv = binding[key]
            self.assertIsInstance(argv, list)
            self.assertTrue(all(isinstance(value, str) for value in argv))
            self.assertIn("--ue-profile-id", argv)
            self.assertIn("--ue-profile-registry-csv", argv)
            self.assertIn("--require-ue-profile-binding", argv)
            self.assertNotIn("--ae-checkpoint", argv)
        self.assertNotIn("--object-score-threshold", binding["front_args"])
        self.assertIn("--object-score-threshold", binding["edge_args"])
        self.assertEqual(
            binding["edge_args"][binding["edge_args"].index("--object-score-threshold") + 1],
            "0.2",
        )

    def test_direct_cli_emits_one_machine_readable_json_document(self) -> None:
        script = ROOT / "rl_agent/ue_split_wire_contract.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "resolve",
                "--profile-id",
                self.profile.profile_id,
            ],
            cwd="/tmp",
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 1)
        document = json.loads(lines[0])
        self.assertEqual(document["profile_id"], self.profile.profile_id)
        self.assertEqual(document["schema"], LAUNCH_BINDING_SCHEMA)
        self.assertEqual(document["registry_sha256"], A1_REGISTRY_CSV_SHA256)

    def test_registry_file_hash_mismatch_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-wire-registry-") as directory:
            copied = Path(directory) / "registry.csv"
            copied.write_bytes(DEFAULT_REGISTRY_CSV.read_bytes() + b"\n")
            self.assertNotEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), A1_REGISTRY_CSV_SHA256)
            self.assertContractCode(
                "REGISTRY_FILE_HASH_MISMATCH",
                lambda: load_registered_profiles(copied),
            )


if __name__ == "__main__":
    unittest.main()
