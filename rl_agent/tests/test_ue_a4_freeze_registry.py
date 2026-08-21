from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rl_agent.ue_a4_freeze_registry import (
    CONFIG_SCHEMA,
    DEFAULT_CONFIG,
    TECHNICAL_REGISTRY_FIELDS,
    TechnicalRegistryError,
    _load_inputs,
    _read_csv,
    _validate_a2_bundle,
    assemble,
    join_and_promote,
    load_config,
    technical_row_fingerprint,
    validate_bundle,
)
from rl_agent.ue_split_wire_contract import action_contract_sha256


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UEA4FreezeRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = Path(tempfile.mkdtemp(prefix="ue-a4-registry-tests-"))
        cls.output = cls.temp / "registry_a"
        assemble(DEFAULT_CONFIG, cls.output, now="2026-08-20T20:00:00+00:00")
        cls.rows = _read_csv(cls.output / "ue_split_technical_registry.csv")
        cls.manifest = json.loads((cls.output / "manifest.json").read_text())
        config = load_config(DEFAULT_CONFIG)
        loaded = _load_inputs(config)
        cls.a1_rows = _read_csv(loaded["a1_registry"])
        cls.a2_rows = _read_csv(loaded["a2_profiles_path"])
        cls.launch_bindings = loaded["certified_launch_bindings"]
        cls.launch_digests = loaded["certified_launch_digests"]
        cls.config = config
        cls.loaded = loaded

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp, ignore_errors=True)

    def test_exact_unfiltered_72_action_grid_is_frozen(self) -> None:
        self.assertEqual(len(self.rows), 72)
        self.assertEqual(len({row["profile_id"] for row in self.rows}), 72)
        self.assertEqual(len({row["action_contract_sha256"] for row in self.rows}), 72)
        self.assertEqual(
            {row["model_family"] for row in self.rows}, {"noae", "ae32", "ae64", "ae128"}
        )
        self.assertEqual(
            {row["quantization_mode"] for row in self.rows},
            {"per_channel_uint8", "per_channel_uint6", "per_channel_uint4"},
        )
        self.assertEqual(
            {float(row["roi_drop_fraction"]) for row in self.rows},
            {0.0, 0.3, 0.5, 0.7, 0.9, 0.98},
        )
        self.assertEqual({row["technical_validity_status"] for row in self.rows}, {"TECHNICALLY_VALID"})
        self.assertEqual({row["quality_mask_applied"] for row in self.rows}, {"False"})
        self.assertEqual({row["technical_invalid_reason"] for row in self.rows}, {""})

    def test_a1_operational_identity_and_a2_contract_are_preserved(self) -> None:
        a1 = {row["profile_id"]: row for row in self.a1_rows}
        a2 = {row["profile_id"]: row for row in self.a2_rows}
        for row in self.rows:
            source = a1[row["profile_id"]]
            evidence = a2[row["profile_id"]]
            self.assertEqual(row["registry_schema"], source["registry_schema"])
            self.assertEqual(row["registry_id"], source["registry_id"])
            self.assertEqual(row["source_row_fingerprint_sha256"], source["row_fingerprint_sha256"])
            self.assertEqual(row["action_contract_sha256"], action_contract_sha256(source))
            self.assertEqual(row["action_contract_sha256"], evidence["action_contract_sha256"])
            self.assertEqual(row["a2_profile_row_sha256"], hashlib.sha256(
                json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
            ).hexdigest())

    def test_codec_boundary_is_explicit_and_not_an_action(self) -> None:
        self.assertEqual({row["entropy_coder"] for row in self.rows}, {"zstd"})
        self.assertEqual({row["entropy_level"] for row in self.rows}, {"3"})
        self.assertEqual(self.manifest["codec_boundary"]["ue_to_edge"]["coder"], "zstd")
        self.assertEqual(self.manifest["codec_boundary"]["edge_to_map"]["coder"], "zlib")
        self.assertEqual(self.manifest["codec_boundary"]["edge_to_map"]["level"], 1)

    def test_every_a2_gate_is_promoted_without_runtime_retarget(self) -> None:
        for row in self.rows:
            gates = json.loads(row["a2_gate_statuses_json"])
            self.assertEqual(set(gates), {
                "registry_binding_status", "fixture_status", "production_codec_status",
                "in_memory_wire_status", "wire_shape_status", "map_schema_status",
                "model_front_status", "model_edge_status", "roi_execution_status",
                "ae_execution_status", "tail_execution_status", "actual_udp_status",
            })
            self.assertTrue(all(value.startswith("PASS") for value in gates.values()))
            self.assertEqual(row["wire_profile_identity_present"], "True")
            self.assertEqual(row["wire_mismatch_rejection_present"], "True")
            self.assertEqual(row["runtime_path"], row["certified_runtime_path"])
            self.assertEqual(row["runtime_sha256"], row["certified_runtime_sha256"])
            self.assertNotEqual(row["runtime_path"], row["a1_declared_runtime_path"])
            self.assertEqual(
                row["a2_launch_binding_sha256"], self.launch_digests[row["profile_id"]]
            )
            front_args = json.loads(row["front_profile_launch_args_json"])
            edge_args = json.loads(row["edge_profile_launch_args_json"])
            self.assertIn("--require-ue-profile-binding", front_args)
            self.assertIn("--require-ue-profile-binding", edge_args)
        self.assertFalse(self.manifest["authority"]["runtime_retarget_authorized"])
        self.assertEqual(
            self.manifest["gates"]["runtime_retarget"], "NOT_AUTHORIZED_NOT_PERFORMED"
        )

    def test_row_and_bundle_seals_validate(self) -> None:
        self.assertEqual(tuple(self.rows[0]), TECHNICAL_REGISTRY_FIELDS)
        for row in self.rows:
            self.assertEqual(row["technical_row_fingerprint_sha256"], technical_row_fingerprint(row))
        validated = validate_bundle(self.output)
        self.assertEqual(validated["counts"], {
            "profiles": 72,
            "technically_valid": 72,
            "technically_invalid": 0,
            "quality_masked": 0,
        })
        terminal = json.loads((self.output / "UE_A4_TECHNICAL_REGISTRY_FROZEN.json").read_text())
        self.assertEqual(terminal["manifest_sha256"], sha256(self.output / "manifest.json"))
        self.assertEqual(terminal["next_checklist_item"], "UE-N1")
        self.assertFalse(terminal["runtime_retargeted"])

    def test_manifest_commits_all_output_bytes(self) -> None:
        for record in self.manifest["outputs"]:
            path = self.output / record["path"]
            self.assertEqual(sha256(path), record["sha256"])
            self.assertEqual(path.stat().st_size, record["bytes"])
            if "rows" in record:
                self.assertEqual(len(_read_csv(path)), record["rows"])

    def test_registry_is_create_only_and_deterministic(self) -> None:
        with self.assertRaisesRegex(TechnicalRegistryError, "refusing to overwrite"):
            assemble(DEFAULT_CONFIG, self.output)
        other = self.temp / "registry_b"
        assemble(DEFAULT_CONFIG, other, now="2026-08-20T20:00:00+00:00")
        for name in ("ue_split_technical_registry.csv", "REPORT.md", "resolved_config.json", "manifest.json"):
            self.assertEqual(sha256(self.output / name), sha256(other / name), name)

    def test_bundle_tampering_is_rejected(self) -> None:
        tampered = self.temp / "tampered"
        shutil.copytree(self.output, tampered)
        registry = tampered / "ue_split_technical_registry.csv"
        text = registry.read_text()
        registry.write_text(text.replace("TECHNICALLY_VALID", "INVALID", 1))
        with self.assertRaisesRegex(TechnicalRegistryError, "output seal mismatch"):
            validate_bundle(tampered)

    def test_self_consistent_one_row_reseal_is_rejected(self) -> None:
        tampered = self.temp / "one_row_resealed"
        shutil.copytree(self.output, tampered)
        registry = tampered / "ue_split_technical_registry.csv"
        rows = _read_csv(registry)[:1]
        with registry.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TECHNICAL_REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = tampered / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record = next(
            item for item in manifest["outputs"] if item["path"] == registry.name
        )
        record.update({"sha256": sha256(registry), "bytes": registry.stat().st_size, "rows": 1})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        terminal_path = tampered / "UE_A4_TECHNICAL_REGISTRY_FROZEN.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["manifest_sha256"] = sha256(manifest_path)
        terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(TechnicalRegistryError, "row count"):
            validate_bundle(tampered)

    def test_self_consistent_gate_downgrade_reseal_is_rejected(self) -> None:
        tampered = self.temp / "gate_resealed"
        shutil.copytree(self.output, tampered)
        registry = tampered / "ue_split_technical_registry.csv"
        rows = _read_csv(registry)
        gates = json.loads(rows[0]["a2_gate_statuses_json"])
        gates["actual_udp_status"] = "FAIL"
        rows[0]["a2_gate_statuses_json"] = json.dumps(
            gates, sort_keys=True, separators=(",", ":")
        )
        rows[0]["technical_row_fingerprint_sha256"] = technical_row_fingerprint(rows[0])
        with registry.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TECHNICAL_REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = tampered / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record = next(
            item for item in manifest["outputs"] if item["path"] == registry.name
        )
        record.update({"sha256": sha256(registry), "bytes": registry.stat().st_size})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        terminal_path = tampered / "UE_A4_TECHNICAL_REGISTRY_FROZEN.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["manifest_sha256"] = sha256(manifest_path)
        terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(TechnicalRegistryError, "reconstructed A1/A2"):
            validate_bundle(tampered)

    def test_join_rejects_missing_duplicate_factor_and_status_failures(self) -> None:
        cases: list[tuple[list[dict[str, str]], str]] = []
        cases.append(([dict(row) for row in self.a2_rows[:-1]], "exactly 72"))
        duplicate = [dict(row) for row in self.a2_rows]
        duplicate[-1]["profile_id"] = duplicate[0]["profile_id"]
        cases.append((duplicate, "not unique"))
        mismatch = [dict(row) for row in self.a2_rows]
        mismatch[0]["quantization_bits"] = "2"
        cases.append((mismatch, "quantization_bits mismatch"))
        failed = [dict(row) for row in self.a2_rows]
        failed[0]["actual_udp_status"] = "FAIL"
        cases.append((failed, "A2 gate failed"))
        blocked = [dict(row) for row in self.a2_rows]
        blocked[0]["blocking_codes"] = "UDP_TIMEOUT"
        cases.append((blocked, "blocking codes"))
        for rows, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TechnicalRegistryError, message):
                    join_and_promote(
                        self.config,
                        self.a1_rows,
                        rows,
                        self.launch_bindings,
                        self.launch_digests,
                    )

    def test_join_rejects_quality_mask_and_action_contract_tampering(self) -> None:
        a1_masked = [dict(row) for row in self.a1_rows]
        a1_masked[0]["quality_mask_applied"] = "True"
        with self.assertRaisesRegex(TechnicalRegistryError, "fingerprint|quality mask"):
            join_and_promote(
                self.config,
                a1_masked,
                self.a2_rows,
                self.launch_bindings,
                self.launch_digests,
            )
        a2_tampered = [dict(row) for row in self.a2_rows]
        a2_tampered[0]["action_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(TechnicalRegistryError, "action-contract mismatch"):
            join_and_promote(
                self.config,
                self.a1_rows,
                a2_tampered,
                self.launch_bindings,
                self.launch_digests,
            )

    def test_a2_completion_gates_ignore_only_stale_manifest_environment(self) -> None:
        loaded = dict(self.loaded)
        manifest = json.loads(json.dumps(self.loaded["a2_manifest"]))
        self.assertFalse(manifest["environment"]["actual_udp_executed"])
        self.assertFalse(manifest["environment"]["model_inference_executed"])
        loaded["a2_manifest"] = manifest
        _validate_a2_bundle(self.config, loaded)
        terminal = dict(self.loaded["a2_terminal"])
        terminal["actual_udp_executed"] = False
        loaded["a2_terminal"] = terminal
        with self.assertRaisesRegex(TechnicalRegistryError, "terminal gate failed"):
            _validate_a2_bundle(self.config, loaded)

    def test_config_rejects_zlib_feature_migration_quality_filter_and_a2_01(self) -> None:
        mutations = []
        codec = json.loads(DEFAULT_CONFIG.read_text())
        codec["repository_root"] = str(ROOT)
        codec["transport_decision"]["ue_to_edge"]["coder"] = "zlib"
        mutations.append((codec, "zstd level 3"))
        quality = json.loads(DEFAULT_CONFIG.read_text())
        quality["repository_root"] = str(ROOT)
        quality["authority"]["quality_filter_authorized"] = True
        mutations.append((quality, "evidence-only"))
        old = json.loads(DEFAULT_CONFIG.read_text())
        old["repository_root"] = str(ROOT)
        old["a2"]["required_bundle_name"] = "20260820_cuda_model_smoke_01"
        mutations.append((old, "only the authoritative"))
        for index, (value, message) in enumerate(mutations):
            path = self.temp / f"bad_config_{index}.json"
            path.write_text(json.dumps(value))
            with self.subTest(message=message):
                with self.assertRaisesRegex(TechnicalRegistryError, message):
                    load_config(path)

    def test_direct_script_cli_bootstraps_repo_imports(self) -> None:
        output = self.temp / "direct_cli"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "rl_agent/ue_a4_freeze_registry.py"),
                "--output-dir",
                str(output),
            ],
            cwd=self.temp,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "UE_A4_TECHNICAL_REGISTRY_FROZEN.json").is_file())
        validate_bundle(output)

    def test_config_schema_is_versioned(self) -> None:
        self.assertEqual(json.loads(DEFAULT_CONFIG.read_text())["schema"], CONFIG_SCHEMA)


if __name__ == "__main__":
    unittest.main()
