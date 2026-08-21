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

from rl_agent.ue_split_profile_registry import (
    ProfileRegistryError,
    _build_rows,
    _read_evidence_rows,
    assemble,
    canonical_profile_id,
    load_config,
    registry_row_fingerprint,
    validate_registry_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "rl_agent/configs/ue_split_profile_registry_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UESplitProfileRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = Path(tempfile.mkdtemp(prefix="ue-split-registry-tests-"))
        cls.output = cls.temp / "registry_a"
        assemble(CONFIG, cls.output, now="2026-08-20T12:00:00+00:00")
        with (cls.output / "ue_split_profile_registry.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.manifest = json.loads((cls.output / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp, ignore_errors=True)

    def test_exact_unfiltered_72_action_grid(self) -> None:
        self.assertEqual(len(self.rows), 72)
        self.assertEqual(len({row["profile_id"] for row in self.rows}), 72)
        self.assertEqual(set(row["model_family"] for row in self.rows), {"noae", "ae32", "ae64", "ae128"})
        self.assertEqual(
            set(row["quantization_mode"] for row in self.rows),
            {"per_channel_uint8", "per_channel_uint6", "per_channel_uint4"},
        )
        self.assertEqual(
            {float(row["roi_drop_fraction"]) for row in self.rows},
            {0.0, 0.3, 0.5, 0.7, 0.9, 0.98},
        )
        self.assertEqual(set(row["quality_mask_applied"] for row in self.rows), {"False"})
        self.assertEqual(
            set(row["technical_validity_status"] for row in self.rows),
            {"REGISTERED_PENDING_SMOKE"},
        )

    def test_checkpoints_are_strict_and_integrated_ae_shapes_are_bound(self) -> None:
        self.assertEqual(
            set(row["checkpoint_strict_structure_status"] for row in self.rows),
            {"PASS"},
        )
        expected = {
            "noae": "1x960x27x48",
            "ae32": "1x32x27x48",
            "ae64": "1x64x27x48",
            "ae128": "1x128x27x48",
        }
        for family, shape in expected.items():
            family_rows = [row for row in self.rows if row["model_family"] == family]
            self.assertEqual(len(family_rows), 18)
            self.assertEqual(set(row["expected_high_wire_shape"] for row in family_rows), {shape})
            self.assertEqual(
                set(row["expected_high_after_edge_decode_shape"] for row in family_rows),
                {"1x960x27x48"},
            )
            self.assertEqual(
                set(row["external_ae_override_allowed"] for row in family_rows),
                {"False"},
            )

    def test_decoder_binding_overrides_incompatible_live_defaults(self) -> None:
        for row in self.rows:
            self.assertEqual(float(row["object_score_threshold"]), 0.2)
            self.assertEqual(int(row["object_nms_radius_px"]), 2)
            self.assertEqual(int(row["topk_objects"]), 120)
            self.assertEqual(int(row["max_objects_published"]), 120)
            args = json.loads(row["edge_profile_launch_args_json"])
            self.assertEqual(args[args.index("--object-score-threshold") + 1], "0.2")
            self.assertEqual(args[args.index("--object-nms-radius-px") + 1], "2")
            self.assertEqual(args[args.index("--topk-objects") + 1], "120")
            self.assertEqual(args[args.index("--max-objects-drawn") + 1], "120")
            self.assertEqual(args[args.index("--chunk-bytes") + 1], "60000")
            self.assertNotIn("--ae-checkpoint", args)
            self.assertTrue(row["edge_container_checkpoint_path"].startswith("/work/abiodun/"))
            self.assertEqual(
                row["edge_launcher_propagation_status"],
                "PENDING_UE_A2_DECODER_OVERRIDE_INTEGRATION",
            )
            front_args = json.loads(row["front_profile_launch_args_json"])
            self.assertFalse(front_args[front_args.index("--fusion-checkpoint") + 1].startswith("/"))
        self.assertEqual(
            self.manifest["runtime_audit"]["profile_evidence_decoder_binding"],
            {
                "object_score_threshold": 0.2,
                "object_nms_radius_px": 2,
                "topk_objects": 120,
            },
        )
        input_paths = {item["path"] for item in self.manifest["inputs"]}
        self.assertIn("rl_agent/density_knob/raw/eval_settings.json", input_paths)
        self.assertIn("rl_agent/density_knob/density_knob_eval.py", input_paths)

    def test_wire_identity_gap_is_explicit_not_false_pass(self) -> None:
        self.assertEqual(set(row["wire_profile_identity_present"] for row in self.rows), {"False"})
        self.assertEqual(set(row["wire_mismatch_rejection_present"] for row in self.rows), {"False"})
        self.assertEqual(set(row["wire_smoke_status"] for row in self.rows), {"PENDING_UE_A2"})
        self.assertIn("WIRE_PROFILE_IDENTITY_ABSENT", self.manifest["known_gaps"])
        self.assertEqual(self.manifest["counts"]["technically_valid"], 0)
        self.assertEqual(
            self.manifest["gates"]["current_edge_launcher_profile_override_propagation"],
            "PENDING_UE_A2",
        )
        self.assertIn(
            "EDGE_LAUNCHER_DECODER_OVERRIDES_NOT_PROPAGATED",
            self.manifest["known_gaps"],
        )

    def test_row_fingerprints_are_reproducible_from_csv_values(self) -> None:
        for row in self.rows:
            self.assertEqual(row["row_fingerprint_sha256"], registry_row_fingerprint(row))

    def test_manifest_and_terminal_seals_validate(self) -> None:
        for output in self.manifest["outputs"]:
            path = self.output / output["path"]
            self.assertEqual(sha256(path), output["sha256"])
            if path.suffix == ".csv":
                with path.open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), output["rows"])
        terminal = json.loads(
            (self.output / "UE_A1_STATIC_BINDING_VERIFIED.json").read_text()
        )
        self.assertEqual(terminal["manifest_sha256"], sha256(self.output / "manifest.json"))
        self.assertFalse(terminal["technical_validity_frozen"])
        self.assertEqual(terminal["next_checklist_item"], "UE-A2")
        self.assertEqual(
            {key: value for key, value in terminal.items() if key != "manifest_sha256"},
            self.manifest["terminal_decision_payload"],
        )
        validate_registry_bundle(self.output)

    def test_terminal_decision_tamper_breaks_manifest_commitment(self) -> None:
        tampered = self.temp / "registry_terminal_tamper"
        shutil.copytree(self.output, tampered)
        path = tampered / "UE_A1_STATIC_BINDING_VERIFIED.json"
        terminal = json.loads(path.read_text())
        terminal["technical_validity_frozen"] = True
        path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ProfileRegistryError, "manifest-committed payload"):
            validate_registry_bundle(tampered)

    def test_registry_is_create_only(self) -> None:
        with self.assertRaisesRegex(ProfileRegistryError, "refusing to overwrite"):
            assemble(CONFIG, self.output)

    def test_registry_rows_are_deterministic(self) -> None:
        other = self.temp / "registry_b"
        assemble(CONFIG, other, now="2026-08-20T12:00:00+00:00")
        for name in ("ue_split_profile_registry.csv", "REPORT.md", "resolved_config.json"):
            self.assertEqual(sha256(self.output / name), sha256(other / name), name)

    def test_direct_script_cli_resolves_repository_imports(self) -> None:
        output = self.temp / "registry_direct_cli"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "rl_agent/ue_split_profile_registry.py"),
                "--output-dir",
                str(output),
            ],
            cwd=self.temp,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "UE_A1_STATIC_BINDING_VERIFIED.json").is_file())

    def test_canonical_profile_id_binds_every_action_factor(self) -> None:
        value = canonical_profile_id(
            "ae64",
            "per_channel_uint6",
            0.98,
            3,
            "c6a2362c7c2d72ff31825508ae7532c0796ec063a8556317d47d8d30fad99480",
        )
        self.assertEqual(value, "ae64__u6__q0.98__zstd3__ckptc6a2362c7c2d")

    def test_config_rejects_quality_filter_authority(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-registry-bad-config-") as directory:
            config = json.loads(CONFIG.read_text())
            config["repository_root"] = str(ROOT)
            config["authority"]["profile_quality_filter"] = True
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ProfileRegistryError, "static-only"):
                load_config(path)

    def test_hash_drift_fails_and_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-registry-hash-drift-") as directory:
            config = json.loads(CONFIG.read_text())
            config["repository_root"] = str(ROOT)
            config["runtime_contract"]["feature_codec_sha256"] = "0" * 64
            path = Path(directory) / "bad.json"
            output = Path(directory) / "output"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ProfileRegistryError, "hash drift"):
                assemble(path, output)
            self.assertFalse(output.exists())

    def test_duplicate_evidence_factor_row_fails_closed(self) -> None:
        config, root = load_config(CONFIG)
        rows, evidence_path = _read_evidence_rows(config, root)
        with self.assertRaisesRegex(ProfileRegistryError, "duplicate evidence factor row"):
            _build_rows(config, root, [*rows, dict(rows[0])], evidence_path)


if __name__ == "__main__":
    unittest.main()
