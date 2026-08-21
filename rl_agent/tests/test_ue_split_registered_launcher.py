from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "uplink_only_spatial_map_pipeline"
    / "run_track1_oai_default106_ttracer_10fps_v2.sh"
)
LEGACY_LAUNCHER = (
    ROOT
    / "uplink_only_spatial_map_pipeline"
    / "run_track1_oai_default106_ttracer_10fps.sh"
)
REGISTRY = (
    ROOT
    / "rl_agent"
    / "registries"
    / "ue_split_profile_registry_v1"
    / "ue_split_profile_registry.csv"
)
LEGACY_LAUNCHER_SHA256 = (
    "4bd64a5992a50daeb33b5da45a41495350c91ef599f9b105fe4e7f67fc7c09da"
)

PROFILE_OVERRIDE_ENV = {
    "AE_CHECKPOINT",
    "AE_CHECKPOINT_CONTAINER",
    "CHECKPOINT",
    "CHECKPOINT_CONTAINER",
    "QUANTIZATION_MODE",
    "ENTROPY_CODER",
    "ZSTD_LEVEL",
    "ROI_THRESHOLD",
    "CHUNK_BYTES",
    "OBJECT_SCORE_THRESHOLD",
    "OBJECT_NMS_RADIUS_PX",
    "TOPK_OBJECTS",
    "MAX_OBJECTS_DRAWN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_launcher_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in {
        *PROFILE_OVERRIDE_ENV,
        "UE_SPLIT_PROFILE_ID",
        "UE_SPLIT_PROFILE_REGISTRY_CSV",
        "UE_PROFILE_BINDING_ONLY",
    }:
        env.pop(name, None)
    return env


def _run_launcher(**updates: str) -> subprocess.CompletedProcess[str]:
    env = _clean_launcher_env()
    env.update(updates)
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _arg_value(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise AssertionError(f"missing value after {flag}")
    return argv[index + 1]


class UESplitRegisteredLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with REGISTRY.open(newline="", encoding="utf-8") as handle:
            cls.row = next(csv.DictReader(handle))

    def test_launcher_is_valid_bash_and_legacy_v1_is_untouched(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertEqual(_sha256(LEGACY_LAUNCHER), LEGACY_LAUNCHER_SHA256)

    def test_profile_id_and_registry_are_both_required(self) -> None:
        missing_profile = _run_launcher(
            UE_SPLIT_PROFILE_REGISTRY_CSV=str(REGISTRY),
            UE_PROFILE_BINDING_ONLY="1",
        )
        self.assertEqual(missing_profile.returncode, 2)
        self.assertIn("UE_SPLIT_PROFILE_ID is required", missing_profile.stderr)

        missing_registry = _run_launcher(
            UE_SPLIT_PROFILE_ID=self.row["profile_id"],
            UE_PROFILE_BINDING_ONLY="1",
        )
        self.assertEqual(missing_registry.returncode, 2)
        self.assertIn("UE_SPLIT_PROFILE_REGISTRY_CSV is required", missing_registry.stderr)

    def test_external_ae_and_independent_profile_overrides_fail_closed(self) -> None:
        common = {
            "UE_SPLIT_PROFILE_ID": self.row["profile_id"],
            "UE_SPLIT_PROFILE_REGISTRY_CSV": str(REGISTRY),
            "UE_PROFILE_BINDING_ONLY": "1",
        }
        external_ae = _run_launcher(**common, AE_CHECKPOINT="unexpected.pt")
        self.assertEqual(external_ae.returncode, 2)
        self.assertIn("external AE overrides are forbidden", external_ae.stderr)

        raw_knob = _run_launcher(**common, QUANTIZATION_MODE="per_channel_uint4")
        self.assertEqual(raw_knob.returncode, 2)
        self.assertIn("QUANTIZATION_MODE is not accepted", raw_knob.stderr)

    def test_binding_only_resolves_one_row_into_exact_front_and_edge_argv(self) -> None:
        result = _run_launcher(
            UE_SPLIT_PROFILE_ID=self.row["profile_id"],
            UE_SPLIT_PROFILE_REGISTRY_CSV=str(REGISTRY),
            UE_PROFILE_BINDING_ONLY="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        binding = json.loads(result.stdout)
        front = binding["front_args"]
        edge = binding["edge_args"]

        self.assertEqual(binding["profile_id"], self.row["profile_id"])
        self.assertEqual(binding["registry_id"], self.row["registry_id"])
        self.assertEqual(binding["registry_paths"]["host"], str(REGISTRY.resolve()))
        self.assertEqual(
            binding["checkpoint_paths"]["host"],
            str((ROOT / self.row["checkpoint_path"]).resolve()),
        )
        self.assertEqual(
            binding["checkpoint_paths"]["container"],
            self.row["edge_container_checkpoint_path"],
        )

        expected_common = {
            "--quantization-mode": self.row["quantization_mode"],
            "--entropy-coder": self.row["entropy_coder"],
            "--zstd-level": self.row["entropy_level"],
            "--roi-threshold": self.row["roi_drop_fraction"],
            "--chunk-bytes": self.row["udp_chunk_bytes"],
            "--model-input-width": self.row["input_width"],
            "--model-input-height": self.row["input_height"],
        }
        for flag, expected in expected_common.items():
            with self.subTest(role="front", flag=flag):
                self.assertEqual(_arg_value(front, flag), expected)
            with self.subTest(role="edge", flag=flag):
                self.assertEqual(_arg_value(edge, flag), expected)

        self.assertEqual(
            _arg_value(front, "--fusion-checkpoint"),
            binding["checkpoint_paths"]["host"],
        )
        self.assertEqual(
            _arg_value(edge, "--fusion-checkpoint"),
            binding["checkpoint_paths"]["container"],
        )
        self.assertEqual(
            _arg_value(front, "--ue-profile-registry-csv"),
            binding["registry_paths"]["host"],
        )
        self.assertEqual(
            _arg_value(edge, "--ue-profile-registry-csv"),
            binding["registry_paths"]["container"],
        )
        for argv in (front, edge):
            self.assertEqual(_arg_value(argv, "--ue-profile-id"), self.row["profile_id"])
            self.assertIn("--require-ue-profile-binding", argv)
            self.assertNotIn("--ae-checkpoint", argv)

        expected_decoder = {
            "--object-score-threshold": self.row["object_score_threshold"],
            "--object-nms-radius-px": self.row["object_nms_radius_px"],
            "--topk-objects": self.row["topk_objects"],
            "--max-objects-drawn": self.row["max_objects_published"],
        }
        for flag, expected in expected_decoder.items():
            self.assertEqual(_arg_value(edge, flag), expected)
            self.assertNotIn(flag, front)

    def test_launcher_consumes_arrays_without_eval_and_selects_v2_on_both_roles(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("eval ", source)
        self.assertEqual(
            source.count("carla_fusion_staleness_scenario_uplink_only_v2.py"),
            2,
        )
        self.assertIn('"${FRONT_PROFILE_ARGS[@]}"', source)
        self.assertIn('extra_args+=("${EDGE_PROFILE_EXTRA_ARGS[@]}")', source)
        self.assertIn('FUSION_BACK_CHECKPOINT="${CHECKPOINT_CONTAINER}"', source)
        self.assertIn('FUSION_QUANTIZATION_MODE="${QUANTIZATION_MODE}"', source)
        self.assertIn('FUSION_ENTROPY_CODER="${ENTROPY_CODER}"', source)
        self.assertNotIn("--ae-checkpoint", source)


if __name__ == "__main__":
    unittest.main()
